from collections import OrderedDict
import multiprocessing
import os
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import TypedDict
import numpy as np
import time
from numpy.typing import NDArray
import importlib.machinery
import importlib.util
import types
import random

from mindspore.mindrecord import FileWriter

from .dataset import DistributedDataset, SimpleDataset
from .tokenizers import CPMBeeTokenizer

class _MixedDatasetConfig(TypedDict):
    weight: float
    path: str
    transforms: Union[List[Dict[str, Any]], str]
    task_name: str
    dataset_name: str
    incontext_weight: List[float]

    lines: int
    dataset: DistributedDataset


CPMBeeInputType = Union[str, Dict[str, "CPMBeeInputType"]]


class _DictTree(TypedDict):
    value: str
    children: List["_DictTree"]
    depth: int
    segment_id: int
    need_predict: bool


class _PrevExtTableStates(TypedDict):
    ext_table: Dict[int, str]
    token_id_table: Dict[str, Dict[int, int]]


class _TransformFuncDict(TypedDict):
    loader: importlib.machinery.SourceFileLoader
    module: types.ModuleType
    last_m: float


_TransformFunction = Callable[[CPMBeeInputType, int, random.Random], CPMBeeInputType]

class CPMBeeBatch(TypedDict):
    inputs: NDArray[np.int32]
    inputs_sub: NDArray[np.int32]
    length: NDArray[np.int32]
    context: NDArray[np.bool_]
    sample_ids: NDArray[np.int32]
    num_segments: NDArray[np.int32]
    segment_ids: NDArray[np.int32]
    segment_rel_offset: NDArray[np.int32]
    segment_rel: NDArray[np.int32]
    spans: NDArray[np.int32]
    target: NDArray[np.int32]
    ext_ids: NDArray[np.int32]
    ext_sub: NDArray[np.int32]
    task_ids: NDArray[np.int32]
    task_names: List[str]
    raw_data: List[Any]


def rel_to_bucket(n_up: int, n_down: int, max_depth: int = 8):
    ret = n_up * max_depth + n_down
    if ret == 0:
        return ret
    else:
        # bucket 1 is reserved for incontext samples
        return ret + 1


def convert_data_to_id(
    tokenizer: CPMBeeTokenizer,
    data: Any,
    prev_ext_states: Optional[_PrevExtTableStates] = None,
    shuffle_answer: bool = True,
    max_depth: int = 8,
):
    root: _DictTree = {
        "value": "<root>",
        "children": [],
        "depth": 0,
        "segment_id": 0,
        "need_predict": False,
    }

    segments = [root]

    def _build_dict_tree(data: CPMBeeInputType, depth: int, need_predict: bool) -> List[_DictTree]:
        if isinstance(data, dict):
            ret_list: List[_DictTree] = []
            curr_items = list(data.items())
            if need_predict and shuffle_answer:
                access_idx = np.arange(len(curr_items))
                np.random.shuffle(access_idx)
                curr_items = [curr_items[idx] for idx in access_idx]
            for k, v in curr_items:
                child_info: _DictTree = {
                    "value": k,
                    "children": [],
                    "depth": depth,
                    "segment_id": len(segments),
                    "need_predict": False,  # only leaves are contexts
                }
                segments.append(child_info)
                child_info["children"] = _build_dict_tree(
                    v, depth + 1, need_predict or (depth == 1 and k == "<ans>")
                )  # elements in <root>.<ans>

                ret_list.append(child_info)
            return ret_list
        else:
            assert isinstance(data, str), "Invalid data {}".format(data)
            ret: _DictTree = {
                "value": data,
                "children": [],
                "depth": depth,
                "segment_id": len(segments),
                "need_predict": need_predict,
            }
            segments.append(ret)
            return [ret]

    root["children"] = _build_dict_tree(data, 1, False)

    num_segments = len(segments)
    segment_rel = np.zeros((num_segments * num_segments,), dtype=np.int32)

    def _build_segment_rel(node: _DictTree) -> List[Tuple[int, int]]:
        ret: List[Tuple[int, int]] = [(node["segment_id"], node["depth"])]
        for child in node["children"]:
            sub = _build_segment_rel(child)
            for seg_id_1, depth_1 in sub:
                for seg_id_2, depth_2 in ret:
                    n_up = min(depth_1 - node["depth"], max_depth - 1)
                    n_down = min(depth_2 - node["depth"], max_depth - 1)
                    segment_rel[seg_id_1 * num_segments + seg_id_2] = rel_to_bucket(
                        n_up, n_down, max_depth=max_depth
                    )
                    segment_rel[seg_id_2 * num_segments + seg_id_1] = rel_to_bucket(
                        n_down, n_up, max_depth=max_depth
                    )
            ret.extend(sub)
        return ret

    _build_segment_rel(root)

    input_ids: List[int] = []
    input_id_subs: List[int] = []
    segment_bound: List[Tuple[int, int]] = []

    ext_table: Dict[int, str] = {}
    token_id_table: Dict[str, Dict[int, int]] = {}

    if prev_ext_states is not None:
        ext_table = prev_ext_states["ext_table"]
        token_id_table = prev_ext_states["token_id_table"]

    for seg in segments:
        tokens, ext_table = tokenizer.encode(seg["value"], ext_table)

        token_id_subs = []
        reid_token_ids = []
        for idx in tokens:
            if idx in ext_table:
                # unk or special token
                token = ext_table[idx]
                if token.startswith("<") and token.endswith(">"):
                    # special token
                    if "_" in token:
                        token_name = token[1:-1].split("_", maxsplit=1)[0]
                    else:
                        token_name = token[1:-1]
                    token_name = "<{}>".format(token_name)
                else:
                    token_name = "<unk>"

                if token_name not in token_id_table:
                    token_id_table[token_name] = {}
                if idx not in token_id_table[token_name]:
                    token_id_table[token_name][idx] = len(token_id_table[token_name])
                if token_name not in tokenizer.encoder:
                    raise ValueError("Invalid token {}".format(token))
                reid_token_ids.append(tokenizer.encoder[token_name])
                token_id_subs.append(token_id_table[token_name][idx])
            else:
                reid_token_ids.append(idx)
                token_id_subs.append(0)
        tokens = [tokenizer.bos_id] + reid_token_ids
        token_id_subs = [0] + token_id_subs
        if not seg["need_predict"]:
            tokens = tokens + [tokenizer.eos_id]
            token_id_subs = token_id_subs + [0]
        else:
            # no eos
            pass
        begin = len(input_ids)
        input_ids.extend(tokens)
        input_id_subs.extend(token_id_subs)
        end = len(input_ids)
        segment_bound.append((begin, end))

    ids = np.array(input_ids, dtype=np.int32)
    id_subs = np.array(input_id_subs, dtype=np.int32)
    segs = np.zeros((ids.shape[0],), dtype=np.int32)
    context = np.zeros((ids.shape[0],), dtype=np.int8)
    for i, (begin, end) in enumerate(segment_bound):
        if not segments[i]["need_predict"]:
            context[begin:end] = 1
        segs[begin:end] = i

    curr_ext_table_states: _PrevExtTableStates = {
        "ext_table": ext_table,
        "token_id_table": token_id_table,
    }
    return ids, id_subs, context, segs, segment_rel, num_segments, curr_ext_table_states


def _dataset_identity(c: _MixedDatasetConfig):
    return "{}.{}".format(c["task_name"], c["dataset_name"])


class _MixedDatasetSaver:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        tokenizer: CPMBeeTokenizer,
        max_depth: int = 16,
    ) -> None:
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_depth = max_depth
        self.tokenizer = tokenizer
        self._transform_func_table: Dict[str, _TransformFuncDict] = {}

        self._inputs: List[NDArray[np.int32]] = []
        self._inputs_sub: List[NDArray[np.int32]] = []
        self._context: List[NDArray[np.int8]] = []
        self._sample_ids: List[NDArray[np.int32]] = []
        self._segments: List[NDArray[np.int32]] = []
        self._num_segments: List[NDArray[np.int32]] = []
        self._segment_rel_offset: List[NDArray[np.int32]] = []
        self._segment_rel: List[NDArray[np.int32]] = []
        self._spans: List[List[int]] = []
        self._task_ids: List[List[str]] = []
        self._raw_data: List[List[Any]] = []

    def __len__(self):
        return len(self._inputs)

    def apply_transform(
        self,
        data: CPMBeeInputType,
        transform: Union[Dict[str, Any], Callable[[CPMBeeInputType], CPMBeeInputType], None],
    ) -> CPMBeeInputType:
        if transform is None:
            return data
        if not isinstance(transform, dict):
            # transform function
            return transform(data)

        mapping_list: List[Tuple[str, str]] = []

        def _walk_transform_dict(data: Union[Dict[str, Any], str], prefix: str = ""):
            if isinstance(data, dict):
                for k, v in data.items():
                    if len(prefix) > 0:
                        _walk_transform_dict(v, prefix + "." + k)
                    else:
                        _walk_transform_dict(v, k)
            else:
                assert isinstance(data, str), "Invalid transform {}".format(data)
                mapping_list.append((prefix, data))

        _walk_transform_dict(transform)

        expanded_mapping_list: List[Tuple[str, Any]] = []

        def _expand_mapping(
            data: CPMBeeInputType, stars: List[str], path: List[str], target: List[str]
        ):
            if len(path) == 0:
                num_stars = 0
                for it in target:
                    if it == "*":
                        num_stars += 1
                if num_stars != len(stars):
                    raise ValueError("Invalid transform {}".format(".".join(target)))

                nw_tgt = []
                num_stars = 0
                for it in target:
                    if it == "*":
                        nw_tgt.append(stars[num_stars])
                        num_stars += 1
                    else:
                        nw_tgt.append(it)
                expanded_mapping_list.append((".".join(nw_tgt), data))
            else:
                if not isinstance(data, dict):
                    raise ValueError("Invalid data {}".format(data))
                if path[0] == "*":
                    for k, v in data.items():
                        _expand_mapping(v, stars + [k], path[1:], target)
                else:
                    _expand_mapping(data[path[0]], stars, path[1:], target)

        # expand mapping list
        for tgt, src in mapping_list:
            if src.startswith("$"):
                # copy from src
                _expand_mapping(data, [], src[1:].split("."), tgt.split("."))
            else:
                if "*" in tgt:
                    raise ValueError("Constant value is not allowed to have `*` in prefix")
                expanded_mapping_list.append((tgt, src))

        ret = {}
        for tgt, val in expanded_mapping_list:
            tgt = tgt.split(".")
            cur = ret
            while len(tgt) > 1:
                cur = cur[tgt[0]]
                tgt = tgt[1:]
            cur[tgt[0]] = val
        return ret

    def data_to_id(
        self,
        data: Any,
        prev_ext_states: Optional[_PrevExtTableStates] = None,
        shuffle_answer: bool = True,
    ):
        return convert_data_to_id(
            self.tokenizer, data, prev_ext_states, shuffle_answer, self._max_depth
        )

    def _ensure_transform_function(
        self, module_name: str, transform_script_path: str
    ) -> _TransformFunction:
        module_name = "cpm_live.transforms.{}".format(module_name)
        if transform_script_path not in self._transform_func_table:
            loader = importlib.machinery.SourceFileLoader(module_name, transform_script_path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            if spec is None:
                raise RuntimeError("spec is none! {}".format(module_name))
            mod = importlib.util.module_from_spec(spec)
            self._transform_func_table[transform_script_path] = {
                "loader": loader,
                "module": mod,
                "last_m": 0,
            }

        transform_script_info = self._transform_func_table[transform_script_path]
        curr_m_time = float(
            transform_script_info["loader"].path_stats(transform_script_path)["mtime"]
        )
        if curr_m_time > transform_script_info["last_m"]:
            transform_script_info["last_m"] = curr_m_time
            transform_script_info["loader"].exec_module(transform_script_info["module"])
        transform_func = getattr(transform_script_info["module"], "transform", None)
        if transform_func is None:

            def _empty_transform_func(data: CPMBeeInputType, num_sample: int, r: random.Random):
                raise NotImplementedError(
                    "Transform func for dataset {} not implemented".format(module_name)
                )

            return _empty_transform_func
        else:
            return transform_func

    def build_instance(self, config: _MixedDatasetConfig):
        _sample_weight = np.array(config["incontext_weight"], dtype=np.float32)
        _sample_weight = _sample_weight / _sample_weight.sum()
        num_incontext = np.random.choice(_sample_weight.shape[0], p=_sample_weight)
        ds = config["dataset"]
        transforms = config["transforms"]
        if isinstance(transforms, str):
            while True:
                try:
                    if not os.path.exists(transforms):
                        raise RuntimeError(
                            "transform script file {} not exists".format(transforms)
                        )
                    # load transform script
                    transform_func = self._ensure_transform_function(
                        _dataset_identity(config), transforms
                    )
                    seed = random.random()
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)

            def _transform(data: CPMBeeInputType):
                r = random.Random(seed)
                return transform_func(data, num_incontext, r)
            transform = _transform
        elif len(transforms) == 0:
            transform = None
        else:
            transform = transforms[np.random.choice(len(transforms))]

        raw_data = {}
        while True:
            inp = ds.read()
            inp = self.apply_transform(inp, transform)

            (
                input_ids,
                input_id_subs,
                context,
                segment_ids,
                segment_rel,
                n_segments,
                table_states,
            ) = self.data_to_id(inp)
            if input_ids.shape[0] > self._max_length:
                # too long
                continue
            input_ids = input_ids[: self._max_length]
            context = context[: self._max_length]
            segment_ids = segment_ids[: self._max_length]
            raw_data["input"] = inp
            raw_data["samples"] = []
            break

        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)

        for i in range(num_incontext):
            if input_ids.shape[0] >= self._max_length:
                # early break
                break

            sample = ds.read()
            sample = self.apply_transform(sample, transform)
            (
                sample_input_ids,
                sample_id_subs,
                _,
                sample_segments,
                sample_rel,
                n_segments,
                table_states,
            ) = self.data_to_id(sample, table_states)

            if input_ids.shape[0] + sample_input_ids.shape[0] > self._max_length:
                # too long, break
                break
            raw_data["samples"].append(sample)
            input_ids = np.concatenate([input_ids, sample_input_ids], axis=0)
            input_id_subs = np.concatenate([input_id_subs, sample_id_subs], axis=0)
            context = np.concatenate(
                [context, np.ones(sample_input_ids.shape, dtype=np.int8)], axis=0
            )
            segment_ids = np.concatenate([segment_ids, sample_segments], axis=0)
            segment_rel_offset = np.concatenate(
                [
                    segment_rel_offset,
                    np.full(sample_input_ids.shape, segment_rel.shape[0], dtype=np.int32),
                ],
                axis=0,
            )
            segment_rel = np.concatenate([segment_rel, sample_rel], axis=0)
            sample_ids = np.concatenate(
                [sample_ids, np.full(sample_input_ids.shape, i + 1, dtype=np.int32)], axis=0
            )
            num_segments = np.concatenate(
                [num_segments, np.full(sample_input_ids.shape, n_segments, dtype=np.int32)], axis=0
            )
        return (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel_offset,
            segment_rel,
            sample_ids,
            num_segments,
            raw_data,
        )

    def pack_batch(self, force: bool = False) -> CPMBeeBatch:
        # pack batch
        if len(self._inputs) < self._batch_size:
            if not force:
                raise RuntimeError("Batch insufficient")
            batch_size = len(self._inputs)
        else:
            batch_size = self._batch_size
        inputs = np.zeros((batch_size, self._max_length), dtype=np.int32)
        inputs_sub = np.zeros((batch_size, self._max_length), dtype=np.int32)
        context = np.zeros((batch_size, self._max_length), dtype=np.int8)
        sample_ids = np.zeros((batch_size, self._max_length), dtype=np.int32)
        segments = np.zeros((batch_size, self._max_length), dtype=np.int32)
        num_segments = np.zeros((batch_size, self._max_length), dtype=np.int32)
        segment_rel_offset = np.zeros((batch_size, self._max_length), dtype=np.int32)
        tgt = np.full((batch_size, self._max_length), -100, dtype=np.int32)

        max_rel = 0
        for i in range(batch_size):
            max_rel = max(max_rel, self._segment_rel[i].shape[0])
        segment_rel = np.zeros((batch_size, max_rel), dtype=np.int32)
        spans = np.zeros((batch_size, self._max_length), dtype=np.int32)
        length = np.zeros((batch_size,), dtype=np.int32)
        task_ids = np.zeros((batch_size, self._max_length), dtype=np.int32)

        all_task_names: Set[str] = set()
        for i in range(batch_size):
            for task_name in self._task_ids[i]:
                all_task_names.add(task_name)
        task_names: List[str] = list(all_task_names)
        task_name_to_id = {name: i for i, name in enumerate(task_names)}

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []
        raw_data_list: List[Any] = []
        for i in range(batch_size):
            instance_length = self._inputs[i].shape[0]
            rel_size = self._segment_rel[i].shape[0]
            inputs[i, :instance_length] = self._inputs[i]
            inputs_sub[i, :instance_length] = self._inputs_sub[i]
            context[i, :instance_length] = self._context[i]
            sample_ids[i, :instance_length] = self._sample_ids[i]
            segments[i, :instance_length] = self._segments[i]
            num_segments[i, :instance_length] = self._num_segments[i]
            segment_rel_offset[i, :instance_length] = self._segment_rel_offset[i]
            segment_rel[i, :rel_size] = self._segment_rel[i]

            span_begin = 0
            for span_id, (span_end, task_name) in enumerate(zip(self._spans[i], self._task_ids[i])):
                spans[i, span_begin:span_end] = span_id
                task_ids[i, span_begin:span_end] = task_name_to_id[task_name]
                span_begin = span_end
            length[i] = instance_length
            raw_data_list.extend(self._raw_data[i])

            for j in range(instance_length):
                idx, idx_sub = self._inputs[i][j], self._inputs_sub[i][j]
                tgt_idx = idx
                if idx_sub > 0:
                    # need to be in ext table
                    if (idx, idx_sub) not in batch_ext_table_map:
                        batch_ext_table_map[(idx, idx_sub)] = len(batch_ext_table_map)
                        batch_ext_table_ids.append(idx)
                        batch_ext_table_sub.append(idx_sub)
                    tgt_idx = batch_ext_table_map[(idx, idx_sub)] + self.tokenizer.vocab_size
                if j > 1 and context[i, j - 1] == 0:
                    if idx != self.tokenizer.bos_id:
                        tgt[i, j - 1] = tgt_idx
                    else:
                        tgt[i, j - 1] = self.tokenizer.eos_id
            if context[i, instance_length - 1] == 0:
                tgt[i, instance_length - 1] = self.tokenizer.eos_id

        if len(batch_ext_table_map) == 0:
            # placeholder
            batch_ext_table_ids.append(0)
            batch_ext_table_sub.append(1)

        self._inputs = self._inputs[batch_size:]
        self._inputs_sub = self._inputs_sub[batch_size:]
        self._context = self._context[batch_size:]
        self._sample_ids = self._sample_ids[batch_size:]
        self._segments = self._segments[batch_size:]
        self._num_segments = self._num_segments[batch_size:]
        self._segment_rel_offset = self._segment_rel_offset[batch_size:]
        self._segment_rel = self._segment_rel[batch_size:]
        self._spans = self._spans[batch_size:]
        self._task_ids = self._task_ids[batch_size:]
        self._raw_data = self._raw_data[batch_size:]
        return {
            "inputs": inputs,
            "inputs_sub": inputs_sub,
            "length": length,
            "context": context > 0,
            "sample_ids": sample_ids,
            "num_segments": num_segments,
            "segment_ids": segments,
            "segment_rel_offset": segment_rel_offset,
            "segment_rel": segment_rel,
            "spans": spans,
            "target": tgt,
            "ext_ids": np.array(batch_ext_table_ids, dtype=np.int32),
            "ext_sub": np.array(batch_ext_table_sub, dtype=np.int32),
            "task_ids": task_ids,
            "task_names": task_names,
            # "raw_data": raw_data_list,
        }

    def add_data(self, config: _MixedDatasetConfig) -> Optional[CPMBeeBatch]:
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel_offset,
            segment_rel,
            sample_ids,
            num_segments,
            raw_data,
        ) = self.build_instance(config)

        # add to batch
        best_fit: Union[None, int] = None
        best_fit_space: Union[None, int] = None
        for i in range(len(self._inputs)):
            space = self._max_length - self._inputs[i].shape[0]
            if input_ids.shape[0] <= space:
                if best_fit_space is None:
                    best_fit = i
                    best_fit_space = space
                elif best_fit_space > space:
                    best_fit = i
                    best_fit_space = space
        if best_fit is None:
            # add a new instance
            self._inputs.append(input_ids)
            self._inputs_sub.append(input_id_subs)
            self._context.append(context)
            self._sample_ids.append(sample_ids)
            self._segments.append(segment_ids)
            self._num_segments.append(num_segments)
            self._segment_rel_offset.append(segment_rel_offset)
            self._segment_rel.append(segment_rel)
            self._spans.append([input_ids.shape[0]])
            self._task_ids.append([config["task_name"]])
            self._raw_data.append([raw_data])
        else:
            # add to existing instance
            self._inputs[best_fit] = np.concatenate([self._inputs[best_fit], input_ids], axis=0)
            self._inputs_sub[best_fit] = np.concatenate(
                [self._inputs_sub[best_fit], input_id_subs], axis=0
            )
            self._context[best_fit] = np.concatenate([self._context[best_fit], context], axis=0)
            self._sample_ids[best_fit] = np.concatenate(
                [self._sample_ids[best_fit], sample_ids], axis=0
            )
            self._segments[best_fit] = np.concatenate(
                [self._segments[best_fit], segment_ids], axis=0
            )
            self._num_segments[best_fit] = np.concatenate(
                [self._num_segments[best_fit], num_segments], axis=0
            )
            self._segment_rel_offset[best_fit] = np.concatenate(
                [
                    self._segment_rel_offset[best_fit],
                    segment_rel_offset + self._segment_rel[best_fit].shape[0],
                ],
                axis=0,
            )
            self._segment_rel[best_fit] = np.concatenate(
                [self._segment_rel[best_fit], segment_rel], axis=0
            )
            self._spans[best_fit].append(self._inputs[best_fit].shape[0])
            self._task_ids[best_fit].append(config["task_name"])
            self._raw_data[best_fit].append(raw_data)

        if len(self._inputs) > self._batch_size:
            return self.pack_batch()
        else:
            # not ready
            return None


def save_mindrecord(dataset_path, save_path, batch_size=32, max_length=2048, max_depth=8, shard_num=1, drop_remainder=False):
    nlp_schema = {
        "inputs": {"type": "int32", "shape": [-1]},
        "inputs_sub": {"type": "int32", "shape": [-1]},
        "length": {"type": "int32"},
        "context": {"type": "int32", "shape": [-1]},
        "sample_ids": {"type": "int32", "shape": [-1]},
        "num_segments": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "segment_rel_offset": {"type": "int32", "shape": [-1]},
        "segment_rel": {"type": "int32", "shape": [-1]},
        "spans": {"type": "int32", "shape": [-1]},
        "target": {"type": "int32", "shape": [-1]},
        "ext_ids": {"type": "int32", "shape": [-1]},
        "ext_sub": {"type": "int32", "shape": [-1]},
        "task_ids": {"type": "int32", "shape": [-1]},
    }

    writer = FileWriter(save_path, shard_num=shard_num, overwrite=True)
    writer.add_schema(nlp_schema, "Preprocessed cpm-bee dataset.")

    tokenizer = CPMBeeTokenizer()
    ds = SimpleDataset(dataset_path, shuffle=False)
    packer = _MixedDatasetSaver(
            batch_size, max_length, tokenizer, max_depth
        )
    ds_cfg: _MixedDatasetConfig = {
    "weight": 1.0,
    "path": dataset_path,
    "transforms": [],
    "task_name": 'test_task',
    "dataset_name": "finetune",
    "incontext_weight": [1.0],
    "lines": len(ds),
    "dataset": ds,
    }


    def process_data(batch, batch_size):
        data = []
        for i in range(batch_size):
            sample = {
                "inputs": batch['inputs'][i],
                "inputs_sub": batch['inputs_sub'][i],
                "length": int(batch['length'][i]),
                "context": batch['context'][i],
                "sample_ids": batch['sample_ids'][i],
                "num_segments": batch['num_segments'][i],
                "segment_ids": batch['segment_ids'][i],
                "segment_rel_offset": batch['segment_rel_offset'][i],
                "segment_rel": batch['segment_rel'][i],
                "spans": batch['spans'][i],
                "target": batch['target'][i],
                "ext_ids": batch['ext_ids'],
                "ext_sub": batch['ext_sub'],
                "task_ids": batch['task_ids'][i],
            }
            data.append(sample)
        return data

    while True:
        try:
            batch = packer.add_data(ds_cfg)
        except EOFError:
            break
        if batch is None:
            continue
        else:
            data = process_data(batch, batch_size)
            writer.write_raw_data(data)

    remain = len(packer)
    if remain > 0:
        batch = packer.pack_batch(force=True)
        if not drop_remainder:
            data = process_data(batch, remain)
            writer.write_raw_data(data)

    writer.commit()
