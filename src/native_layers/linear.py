# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union
from mindspore import nn, ops
from mindspore import Parameter, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Initializer, initializer


class Linear(nn.Cell):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: mstype.tensor = mstype.half,
        param_init: Union[str, Initializer] = 'normal',
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before

        self.weight = Parameter(initializer(param_init, (dim_out, dim_in), dtype=dtype), 'weight')
        self.matmul = ops.MatMul()

    def construct(self, x: Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        x_shape = x.shape
        x = x.astype(self.weight.dtype)
        if self.scale_before:
            x = x / ops.sqrt(ops.scalar_to_tensor(self.dim_in))
            x = self.matmul(x.reshape(-1, x_shape[-1]), self.weight.swapaxes(0, 1))
        else:
            x = self.matmul(x.reshape(-1, x_shape[-1]), self.weight.swapaxes(0, 1))
            x = x / ops.sqrt(ops.scalar_to_tensor(self.dim_in))
        x = x.reshape(x_shape[:-1] + (x.shape[-1],))
        return x

    def shard(self, dp, mp):
        self.matmul.shard(((dp, mp), (mp, 1)))
