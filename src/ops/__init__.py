from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim

def masked_fill(inputs, mask, value):
    masked = ops.full_like(inputs, value, dtype=inputs.dtype)
    outputs = ops.select(mask, masked, inputs)
    return outputs
