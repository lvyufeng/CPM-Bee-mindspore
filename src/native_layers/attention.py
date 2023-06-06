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

from typing import Optional, Tuple
import math
import mindspore
from mindspore import nn, ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from .linear import Linear


class Attention(nn.Cell):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype: mstype.tensor = mstype.half,
        dropout_p: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_k = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_v = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)

        self.attention_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype)

        self.softmax = nn.Softmax(axis=-1)

        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def construct(
        self,
        hidden_q: Tensor,
        hidden_kv: Tensor,
        attention_mask: Tensor,
        position_bias: Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Args:
            hidden_q (:obj:`Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.shape[0]
        len_q = hidden_q.shape[1]
        len_k = hidden_kv.shape[1]

        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_kv)
        h_v = self.project_v(hidden_kv)

        h_q = h_q.view((batch_size, len_q, self.num_heads, self.dim_head)).transpose(0, 2, 1, 3)
        h_k = h_k.view((batch_size, len_k, self.num_heads, self.dim_head)).transpose(0, 2, 1, 3)
        h_v = h_v.view((batch_size, len_k, self.num_heads, self.dim_head)).transpose(0, 2, 1, 3)

        if past_kv is not None:
            h_k = ops.cat([past_kv[0], h_k], axis=-2)
            h_v = ops.cat([past_kv[1], h_v], axis=-2)
            len_k = h_k.shape[-2]

        # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
        score = ops.matmul(h_q, h_k.swapaxes(-1, -2)) / ops.sqrt(ops.scalar_to_tensor(self.dim_head))
        score = score + position_bias
        score = ops.masked_fill(
            score,
            attention_mask.view((batch_size, 1, len_q, len_k)) == False,
            ops.scalar_to_tensor(float("-inf"), dtype=score.dtype),
        )

        score = self.softmax(score)

        score = ops.masked_fill(
            score,
            attention_mask.view((batch_size, 1, len_q, len_k)) == False,
            ops.scalar_to_tensor(0, dtype=score.dtype),
        )

        if self.dropout is not None:
            score = self.dropout(score)

        # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
        score = ops.matmul(score, h_v)

        score = score.view((batch_size, self.num_heads, len_q, self.dim_head)).transpose(0, 2, 1, 3)
        score = score.view((batch_size, len_q, self.num_heads * self.dim_head))

        score = self.attention_out(score)
        if use_cache:
            return score, (h_k, h_v)
        else:
            return score
