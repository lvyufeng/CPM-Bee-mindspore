import os
import mindspore as ms
from mindspore import Tensor, nn, value_and_grad
from mindspore.communication import init
from mindspore.dataset import MindDataset
from mindspore.dataset.transforms import TypeCast
from mindspore.train import Model, LossMonitor
from mindspore.amp import DynamicLossScaleManager

from src.models import CPMBeeConfig, BeeForward
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
                                 dataset_strategy="data_parallel", enable_parallel_optimizer=True)

    init("nccl")

    # 随机构造数据集
    dataset = get_dataset('/mnt/code/lvyufeng/CPM-Bee/tutorials/basic_task_finetune/bin_data/train/test.mindrecord',
                          batch_size=4)

    print(next(dataset.create_tuple_iterator()))

    config = CPMBeeConfig(**cpm_2b_config)
    model = BeeForward(config)
    model.shard(4, 1)

    lr_scheduer = Noam(1e-4, 100, 2000)
    epoch_size = 5
    optimizer = nn.AdamWeightDecay(model.trainable_params(), lr_scheduer)

    loss_scale_manager = DynamicLossScaleManager()
    loss_monitor = LossMonitor()
    trainer = Model(model, optimizer=optimizer, loss_scale_manager=loss_scale_manager)

    trainer.train(epoch_size, dataset, callbacks=[loss_monitor])
