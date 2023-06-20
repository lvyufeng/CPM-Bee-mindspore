import numpy as np
from src.dataset import SimpleDataset
from src.tokenizers import CPMBeeTokenizer
from src.data_converter import _MixedDatasetConfig, _MixedDatasetSaver, save_mindrecord
from src.models import CPMBeeSimple

def test_simple_dataset():
    dataset_path = '/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train'
    ds = SimpleDataset(dataset_path, shuffle=False)
    inp = ds.read()
    print(inp)

def test_map_saver():
    dataset_path = '/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train'
    ds = SimpleDataset(dataset_path, shuffle=False)

    batch_size = 32
    max_length = 2048
    tokenizer = CPMBeeTokenizer()
    max_depth = 8
    _packer = _MixedDatasetSaver(
            batch_size, max_length, tokenizer, max_depth
        )
    _ds_cfg: _MixedDatasetConfig = {
    "weight": 1.0,
    "path": dataset_path,
    "transforms": [],
    "task_name": 'test_task',
    "dataset_name": "finetune",
    "incontext_weight": [1.0],
    "lines": len(ds),
    "dataset": ds,
    }

    count = 0
    while True:
        try:
            batch = _packer.add_data(_ds_cfg)
        except EOFError:
            break
        if batch is None:
            continue
        else:
            count += 1
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    print(k, v.shape, v.dtype)
        if count % 2 == 0:
            break

def test_save_mindrecord():
    dataset_path = '/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train'
    save_path = dataset_path + '/test.mindrecord'
    save_mindrecord(dataset_path, save_path)


def test_prepare_dataset():
    dataset_path = '/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train'
    ds = SimpleDataset(dataset_path, shuffle=False)

    batch_size = 32
    max_length = 2048
    tokenizer = CPMBeeTokenizer()
    max_depth = 8
    _packer = _MixedDatasetSaver(
            batch_size, max_length, tokenizer, max_depth
        )
    _ds_cfg: _MixedDatasetConfig = {
    "weight": 1.0,
    "path": dataset_path,
    "transforms": [],
    "task_name": 'test_task',
    "dataset_name": "finetune",
    "incontext_weight": [1.0],
    "lines": len(ds),
    "dataset": ds,
    }

    while True:
        try:
            batch = _packer.add_data(_ds_cfg)
        except EOFError:
            break
        if batch is None:
            continue
        else:
            break
    
    import time
    s = time.time()
    prepared_tensors = CPMBeeSimple.prepare_data(
        batch['inputs'],  # (batch, seqlen) int32
        batch['inputs_sub'],  # (batch, seqlen) int32
        batch['length'],  # (batch) int32
        batch['context'],  # (batch, seqlen) bool
        batch['sample_ids'],  # (batch, seq_len) int32
        batch['num_segments'],  # (batch, seq_len) int32
        batch['segment_ids'],  # (batch, seqlen) int32
        batch['segment_rel_offset'],  # (batch, seq_len) int32
        batch['segment_rel'],  # (batch, num_segment_bucket) int32
        batch['spans'],  # (batch, seqlen) int32
        batch['ext_ids'],  # (ext_table_size) int32
        batch['ext_sub'],  # (ext_table_size) int32
    )
    t = time.time()
    print(t - s)
    print(prepared_tensors)

