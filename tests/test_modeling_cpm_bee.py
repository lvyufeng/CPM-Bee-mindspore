import pytest
import numpy as np
from mindspore import Tensor, nn, value_and_grad
from mindspore import jit as ms_jit
from src.models import CPMBee, CPMBeeConfig, init_weights

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

@pytest.mark.parametrize('jit', [True, False])
def test_cpm_bee_backward(jit):
    config = CPMBeeConfig(2000, 128, 8, 8, 512, 4, position_bias_num_segment_buckets=32)
    model = CPMBee(config)
    model.apply(init_weights)
    loss_fn = nn.CrossEntropyLoss()

    inputs, targets = make_tensor(4, 32, config.position_bias_num_segment_buckets, 32)

    def forward(inputs, target):
        logits, _ = model(*inputs)
        loss = loss_fn(logits.view((-1, logits.shape[-1])), target.view(-1))
        return loss

    grad_fn = value_and_grad(forward, None, model.trainable_params())

    def train_step(inputs, targets):
        loss, grads = grad_fn(inputs, targets)
        return loss, grads

    if jit:
        train_step = ms_jit(forward)
    
    outputs = train_step(inputs, targets)
