import pytest
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor, nn, value_and_grad
from mindspore import jit as ms_jit
from mindspore.communication import init, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean
from mindspore.train import Model
from src.models import CPMBee, CPMBeeConfig, CPMBeeSimple

def get_dataset(batch, seqlen, num_segment_bucket, ext_table_size, step_per_epoch):
    """
    """
    input = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    label = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    input_sub = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    length = np.random.randint(1, seqlen, (batch,)).astype(np.int32)
    context = np.full((batch, seqlen), 1).astype(np.bool_)
    sample_ids = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    num_segments = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    segment = np.random.randint(0, 1, (batch, seqlen)).astype(np.int32)
    segment_rel_offset = np.random.randint(0, 1, (batch, seqlen)).astype(np.int32)
    segment_rel = np.random.randint(0, 1, (batch, num_segment_bucket)).astype(np.int32)
    span = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    ext_table_ids = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)
    ext_table_sub = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)

    def generate():
        for _ in range(step_per_epoch):
            yield Tensor(input), Tensor(input_sub), Tensor(length), Tensor(context), Tensor(sample_ids), \
                  Tensor(num_segments), Tensor(segment), Tensor(segment_rel), Tensor(segment_rel_offset), \
                  Tensor(span), Tensor(ext_table_ids), Tensor(ext_table_sub), Tensor(label)

    return generate


def get_dataset(batch, seqlen, ext_table_size, step_per_epoch):
    """
    """
    input = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    label = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    input_sub = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    position = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    segment_bucket = np.full((batch, seqlen, seqlen), 1).astype(np.int32)
    attention_mask = np.full((batch, seqlen, seqlen), 1).astype(np.bool_)
    ext_table_ids = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)
    ext_table_sub = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)

    def generate():
        for _ in range(step_per_epoch):
            yield Tensor(input), Tensor(input_sub), Tensor(position), Tensor(segment_bucket), \
                Tensor(attention_mask), Tensor(ext_table_ids), Tensor(ext_table_sub), Tensor(label)

    return generate

cpm_1b_config = {
    # "vocab_size": 86583,
    "vocab_size": 40000,
    "dim_model": 4096,
    "dim_ff" : 1024,
    "num_layers" : 8,
    "num_heads": 32,
    "dim_head" : 40,
    "dropout_p" : 0.0,
    "position_bias_num_buckets" : 256,
    "position_bias_num_segment_buckets": 256,
    "position_bias_max_distance" : 2048,
    "eps" : 1e-6,
    "half" : True,
}

def test_cpm_bee_simple_backward():
    var_single_batch_size = 2

    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", save_graphs=2, save_graphs_path="./saved_graph")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, \
                                 search_mode="sharding_propagation", \
                                 dataset_strategy="data_parallel")

    # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, \
    #                              search_mode="dynamic_programming", \
    #                              dataset_strategy="data_parallel")
    

    group = GlobalComm.WORLD_COMM_GROUP
    init("nccl")
    rank_id = get_rank()

    # 随机构造数据集
    fake_dataset = get_dataset(var_single_batch_size, 256, 64, 100)
    """"
    Tensor(input), Tensor(input_sub), Tensor(position), Tensor(segment_bucket), \
                Tensor(attention_mask), Tensor(ext_table_ids), Tensor(ext_table_sub), Tensor(label)

    """
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "input_sub", "position", "segment_bucket", "attention_mask",
                                                 "ext_table_ids", "ext_table_sub", "label"])

    config = CPMBeeConfig(**cpm_1b_config)
    model = CPMBeeSimple(config)
    model.shard(1, 4)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.4
    epoch_size = 5
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate)
    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree, group=group)

    def forward(inputs):
        logits, _ = model(*inputs[:-1])
        loss = loss_fn(logits.view((-1, logits.shape[-1])), inputs[-1].view(-1))
        return loss

    grad_fn = value_and_grad(forward, None, model.trainable_params())

    @ms_jit
    def train_step(inputs):
        loss, grads = grad_fn(inputs)
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss

    data_iter = dataset.create_tuple_iterator()
    for epoch in range(epoch_size):
        for idx, data in enumerate(data_iter):
            loss = train_step(data)
            if idx % 100 == 0:
                print(f"Epoch_{epoch}/Step_{idx}: Loss:{loss}.")


@pytest.mark.skipif(True, reason='has error')
def test_cpm_bee_backward():
    var_single_batch_size = 2

    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", save_graphs=2, save_graphs_path="./saved_graph")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, \
                                 search_mode="sharding_propagation", \
                                 dataset_strategy="data_parallel")
    
    group = GlobalComm.WORLD_COMM_GROUP
    init("nccl")
    rank_id = get_rank()

    # 随机构造数据集
    fake_dataset = get_dataset(var_single_batch_size, 256, cpm_1b_config["position_bias_num_segment_buckets"], 64, 100)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "input_sub", "length", "context", "sample_ids", "num_segments",
                                                 "segment", "segment_rel", "segment_rel_offset", "span",
                                                 "ext_table_ids", "ext_table_sub", "label"])

    config = CPMBeeConfig(**cpm_1b_config)
    model = CPMBee(config)
    model.shard(1, 4)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.4
    epoch_size = 5
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate)
    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree, group=group)

    def forward(inputs):
        logits, _ = model(*inputs[:-1])
        loss = loss_fn(logits.view((-1, logits.shape[-1])), inputs[-1].view(-1))
        return loss

    grad_fn = value_and_grad(forward, None, model.trainable_params())

    @ms_jit
    def train_step(inputs):
        loss, grads = grad_fn(inputs)
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss

    data_iter = dataset.create_tuple_iterator()
    for epoch in range(epoch_size):
        for idx, data in enumerate(data_iter):
            loss = train_step(data)
            if idx % 100 == 0:
                print(f"Epoch_{epoch}/Step_{idx}: Loss:{loss}.")
