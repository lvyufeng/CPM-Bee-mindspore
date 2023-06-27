import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, value_and_grad
from mindspore import jit as ms_jit
from src.models import CPMBee, CPMBeeConfig, BeeForward, init_weights
from src.dataset import SimpleDataset
from src.tokenizers import CPMBeeTokenizer
from src.data_converter import _MixedDatasetConfig, _MixedDatasetSaver

def make_tensor(batch, seqlen, num_segment_bucket, ext_table_size):
    # input: Tensor,  # (batch, seqlen) int32
    # input_sub: Tensor,  # (batch, seqlen) int32
    # length: Tensor,  # (batch) int32
    # context: Tensor,  # (batch, seqlen) bool
    # sample_ids: Tensor,  # (batch, seq_len) int32
    # num_segments: Tensor,  # (batch, seq_len) int32
    # segment: Tensor,  # (batch, seqlen) int32
    # segment_rel_offset: Tensor,  # (batch, seq_len) int32
    # segment_rel: Tensor,  # (batch, num_segment_bucket) int32
    # span: Tensor,  # (batch, seqlen) int32
    # ext_table_ids: Tensor,  # (ext_table_size) int32
    # ext_table_sub: Tensor,  # (ext_table_size) int32
    input = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
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
    return (Tensor(input), Tensor(input_sub), Tensor(length), Tensor(context), Tensor(sample_ids), \
        Tensor(num_segments), Tensor(segment), Tensor(segment_rel), Tensor(segment_rel_offset), \
        Tensor(span), Tensor(ext_table_ids), Tensor(ext_table_sub)), Tensor(input)

def make_real_tensor():
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

    while True:
        try:
            batch = _packer.add_data(_ds_cfg)
        except EOFError:
            break
        if batch is None:
            continue
        else:
            break

    print(batch['segment_rel'].shape)
    return (
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
        batch['ext_sub'],  # (ext_table_size) int32,
        batch['target']
    )

@pytest.mark.parametrize('jit', [True, False])
def test_cpm_bee_forward(jit):
    config = CPMBeeConfig(2000, 128, 8, 8, 512, 4, position_bias_num_segment_buckets=32)
    model = CPMBee(config)

    inputs, _ = make_tensor(4, 32, config.position_bias_num_segment_buckets, 32)

    def forward(inputs):
        outputs = model(*inputs)
        return outputs
    
    if jit:
        forward = ms_jit(forward)
    
    outputs = forward(inputs)

@pytest.mark.parametrize('jit', [True])
def test_cpm_bee_backward(jit):
    config = CPMBeeConfig(86580, 128, 8, 8, 512, 4, position_bias_num_segment_buckets=2048, half=True)
    model = CPMBee(config)
    model.apply(init_weights)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # inputs, targets = make_tensor(4, -32, config.position_bias_num_segment_buckets, 32)

    inputs_np = make_real_tensor()

    inputs_ms = [Tensor(i) for i in inputs_np]

    def forward(inputs, target):
        logits, _ = model(*inputs)
        logits = logits.astype(mindspore.float32)
        loss = loss_fn(logits.view((-1, logits.shape[-1])), target.view(-1))
        return loss

    grad_fn = value_and_grad(forward, None, model.trainable_params())

    def train_step(inputs, targets):
        loss, grads = grad_fn(inputs, targets)
        return loss

    if jit:
        train_step = ms_jit(forward)
    
    outputs = train_step(inputs_ms[:-1], inputs_ms[-1])
    print(outputs)


def test_cpm_bee_backward_cell():
    config = CPMBeeConfig(86580, 128, 8, 8, 512, 4, position_bias_num_segment_buckets=2048, half=True)
    forward = BeeForward(config)
    forward.apply(init_weights)

    inputs_np = make_real_tensor()

    inputs_ms = [Tensor(i) for i in inputs_np]

    grad_fn = value_and_grad(forward, None, forward.trainable_params())

    def train_step(inputs):
        loss, grads = grad_fn(*inputs)
        return loss
    
    outputs = train_step(inputs_ms)
    print(outputs)
