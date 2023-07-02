import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, value_and_grad
from mindspore.communication import init
from mindspore.dataset import MindDataset, GeneratorDataset
from mindspore.dataset.transforms import TypeCast
from mindspore.train import Model, LossMonitor
from mindspore.amp import DynamicLossScaleManager

from src.models import CPMBeeConfig, BeeForward, TrainOneStep
from src.data_converter import save_mindrecord
from src.lr_scheduler import Noam


def get_dataset(dataset_path, batch_size=32, max_length=2048, max_depth=8):
    dataset = MindDataset(dataset_path, ["inputs", "inputs_sub", "length", "context",
                "sample_ids", "num_segments", "segment", "segment_rel_offset",
                "segment_rel", "span", "ext_table_ids", "ext_table_sub","label"], shuffle=False)
    type_cast_op = TypeCast(ms.bool_)
    dataset = dataset.map(type_cast_op, 'context')
    dataset = dataset.batch(batch_size)
    return dataset


from src.dataset import SimpleDataset
from src.tokenizers import CPMBeeTokenizer
from src.data_converter import _MixedDatasetConfig, _MixedDatasetSaver, save_mindrecord

def get_prepare_dataset():
    dataset_path = '/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train'
    ds = SimpleDataset(dataset_path, shuffle=False)

    batch_size = 2
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

    def generate():
        for i in range(100):
            pad_length = 7000
            ext_pad_length = 10
            max_len = 0
            while True:
                try:
                    batch = _packer.add_data(_ds_cfg)
                except EOFError:
                    break
                if batch is None:
                    continue
                else:
                    break
            cur_len = batch['segment_rel'].shape[1]

            yield (
                batch['inputs'],  # (batch, seqlen) int32
                batch['inputs_sub'],  # (batch, seqlen) int32
                batch['length'],  # (batch) int32
                batch['context'],  # (batch, seqlen) bool
                batch['sample_ids'],  # (batch, seq_len) int32
                batch['num_segments'],  # (batch, seq_len) int32
                batch['segment_ids'],  # (batch, seqlen) int32
                batch['segment_rel_offset'],  # (batch, seq_len) int32
                np.pad(batch['segment_rel'], ((0, 0), (0, pad_length - batch['segment_rel'].shape[1])), 'constant'),  # (batch, num_segment_bucket) int32
                batch['spans'],  # (batch, seqlen) int32
                np.pad(batch['ext_ids'], (0, ext_pad_length - batch['ext_ids'].shape[0]), 'constant'),  # (ext_table_size) int32
                np.pad(batch['ext_sub'], (0, ext_pad_length - batch['ext_ids'].shape[0]), 'constant'),  # (ext_table_size) int32
                batch['target']
            )

    return generate

cpm_2b_config = {
    "vocab_size": 86592,
    "dim_model": 4096,
    "dim_ff" : 5120,
    "num_layers" : 48,
    "num_heads": 32,
    "dim_head" : 64,
    "dropout_p" : 0.0,
    "position_bias_num_buckets" : 256,
    "position_bias_num_segment_buckets": 256,
    "position_bias_max_distance" : 2048,
    "eps" : 1e-6,
    "half" : True,
    "mask_modules": [[False, False], [True, False], [False, False], [True, False], [True, True], [True, False], [True, True], [True, True], [False, False], [False, False], [True, True], [True, False], [True, False], [True, True], [False, False], [True, True], [False, False], [False, True], [True, False], [True, True], [False, False], [False, True], [True, True], [True, True], [False, False], [True, True], [False, False], [True, True], [True, True], [False, False], [True, True], [False, False], [True, True], [False, False], [True, True], [True, False], [True, True], [True, True], [True, True], [False, False], [True, True], [False, False], [True, True], [True, True], [False, False], [True, True], [False, False], [False, False]]
}

def test_cpm_bee_cell():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, \
                                 search_mode="sharding_propagation", \
                                 dataset_strategy="full_batch", enable_parallel_optimizer=True)

    init("nccl")

    # 随机构造数据集
    # dataset = get_dataset('/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train/test.mindrecord',
    #                       batch_size=2)
    dataset = GeneratorDataset(get_prepare_dataset(), ["inputs", "inputs_sub", "length", "context",
                "sample_ids", "num_segments", "segment", "segment_rel_offset",
                "segment_rel", "span", "ext_table_ids", "ext_table_sub","label"], shuffle=False)
    print(next(dataset.create_tuple_iterator()))

    config = CPMBeeConfig(**cpm_2b_config)
    model = BeeForward(config)
    model.shard(1, 4)

    total = [np.prod(param.shape) for param in model.trainable_params()]
    print('total: ', sum(total) // 1000000)

    lr_scheduler = Noam(1e-4, 1, 2000)
    epoch_size = 5
    optimizer = nn.AdamWeightDecay(model.trainable_params(), lr_scheduler, weight_decay=0.01)

    loss_scale_manager = DynamicLossScaleManager()
    loss_monitor = LossMonitor()

    train_step = TrainOneStep(model, optimizer, loss_scale_manager.get_update_cell())
    trainer = Model(train_step)

    trainer.train(epoch_size, dataset, callbacks=[loss_monitor], dataset_sink_mode=False)
