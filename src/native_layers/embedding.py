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

from typing import Optional, Union
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Initializer, initializer

from .position_embedding import RotaryEmbedding


class Embedding(nn.Cell):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: mstype.tensor = mstype.half,
        param_init: Union[str, Initializer] = 'normal',
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.weight = Parameter(
            initializer(param_init, (vocab_size, embedding_size), dtype=dtype),
            'weight'
        )

    def construct(self, ids: Tensor):
        """
        Args:
            ids (:obj:`Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = ops.gather(self.weight, ids, 0) / ops.sqrt(ops.scalar_to_tensor(self.dim_model))

        return embeds

    def projection(self, x: Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        logits = ops.matmul(x / ops.sqrt(ops.scalar_to_tensor(self.dim_model)), self.weight)
        return logits


class EmbeddingExt(nn.Cell):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        distance_scale: int = 16,
        dtype: mstype.tensor = mstype.half,
        param_init: Union[str, Initializer] = 'normal',
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.rotary_emb = RotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype
        )

        self.weight = Parameter(
            initializer(param_init, (vocab_size, embedding_size), dtype=dtype),
            'weight'
        )

    def construct(self, ids: Tensor, ids_sub: Tensor):
        """
        Args:
            ids (:obj:`Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = ops.gather(self.weight, ids, 0) / ops.sqrt(ops.scalar_to_tensor(self.dim_model))
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: Tensor, ext_table: Optional[Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        logits = ops.matmul(x / ops.sqrt(ops.scalar_to_tensor(self.dim_model)), self.weight.swapaxes(0, 1))
        if ext_table is not None:
            logits_ext = ops.matmul(x, ext_table.swapaxes(0, 1))
            logits = ops.cat([logits, logits_ext], axis=-1)
        return logits
