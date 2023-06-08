from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, Constant


class LayerNorm(nn.Cell):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: mstype.float_ = mstype.half,
        eps: float = 1e-6,
        init_var: float = 1.0,
    ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = Parameter(initializer(Constant(init_var), (dim_norm,), dtype=dtype), 'weight')

    def construct(self, x: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.shape[-1] == self.dim_norm
        old_dtype = x.dtype
        variance = x.astype(mstype.float32).pow(2).mean(axis=-1, keep_dims=True)
        x = (x * ops.rsqrt(variance + self.eps)).astype(old_dtype)
        return x * self.weight
