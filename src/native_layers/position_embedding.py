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
import numpy as np
from typing import Union
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Initializer, initializer


class SegmentPositionEmbedding(nn.Cell):
    def __init__(
        self,
        num_heads,
        num_segments=1,
        num_buckets=32,
        max_distance=128,
        bidirectional=False,
        dtype=mstype.half,
        param_init: Union[str, Initializer] = 'normal',
    ):

        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments

        self.relative_attention_bias = Parameter(
            initializer(param_init, (num_segments * num_segments + num_buckets, num_heads), dtype=dtype),
            'relative_attention_bias'
        )

    def construct(
        self,
        key_pos: Tensor,
        query_pos: Tensor,
        key_segment: Tensor,
        query_segment: Tensor,
    ):
        batch = key_pos.shape[0]
        keylen = key_pos.shape[1]
        querylen = query_pos.shape[1]

        assert key_pos.shape[0] == query_pos.shape[1]
        assert keylen == key_segment.shape[1] and querylen == query_segment.shape[1]

        key_pos = key_pos.view((batch, -1, keylen))
        query_pos = query_pos.view((batch, querylen, -1))
        key_segment = key_segment.view((batch, -1, keylen))
        query_segment = query_segment.view((batch, querylen, -1))

        relative_position_bucket = self._segment_relative_position_bucket(
            query_segment, key_segment
        )
        relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

        # b*q*k
        absolute_position_bucket = self._position_bucket(
            ops.arange(keylen, dtype=mstype.int32)[None, :]
            - ops.arange(querylen, dtype=mstype.int32)[:, None],
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_position_bucket = ops.where(
            (key_segment == query_segment),
            absolute_position_bucket[None, :, :],
            relative_position_bucket,
        )
        # (batch, len_q, len_k)
        relative_position_bucket = ops.stop_gradient(relative_position_bucket)

        # (batch, len_q, len_k, num_heads)
        embeds = ops.gather(self.relative_attention_bias, relative_position_bucket, 0)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.transpose(0, 3, 1, 2)
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment

    def _position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).astype(mstype.int32) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, ops.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / ops.log(ops.scalar_to_tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).astype(mstype.int32)
        relative_postion_if_large = ops.minimum(
            relative_postion_if_large,
            ops.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += ops.where(
            is_small, relative_position.to(mstype.int32), relative_postion_if_large
        )
        return relative_buckets


class BucketPositionBias(nn.Cell):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        num_segment_bucket: int = 32,
        max_distance: int = 128,
        dtype: mstype.float_ = mstype.half,
        param_init: Union[str, Initializer] = 'normal',
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.num_segment_bucket = num_segment_bucket
        self.max_distance = max_distance

        self.relative_attention_bias = Parameter(
            initializer(param_init, (num_buckets + num_segment_bucket, num_heads), dtype=dtype),
            'relative_attention_bias'
        )

    def construct(
        self,
        query_pos: Tensor,  # (batch, len_q)
        key_pos: Tensor,  # (batch, len_k)
        rel_buckets: Tensor,  # (batch, len_q, len_k)
    ):

        batch = key_pos.shape[0]
        keylen = key_pos.shape[1]
        querylen = query_pos.shape[1]

        assert key_pos.shape[0] == query_pos.shape[0]
        assert (
            rel_buckets.shape[0] == batch
            and rel_buckets.shape[1] == querylen
            and rel_buckets.shape[2] == keylen
        )

        relative_position_bucket = rel_buckets - 1 + self.num_buckets  # 与相对位置编码区间不重叠

        # b*q*k
        inner_segment_bucket = self._position_bucket(
            key_pos[..., None, :] - query_pos[..., :, None],
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_position_bucket = ops.where(
            rel_buckets == 0,
            inner_segment_bucket,
            relative_position_bucket,
        )
        # (batch, len_q, len_k)
        relative_position_bucket = ops.stop_gradient(relative_position_bucket)

        # (batch, len_q, len_k, num_heads)
        embeds = ops.gather(self.relative_attention_bias, relative_position_bucket, 0)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.transpose(0, 3, 1, 2)
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets = (relative_position > 0).astype(mstype.int32) * num_buckets
        relative_position = ops.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / ops.log(ops.scalar_to_tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).astype(mstype.int32)
        relative_postion_if_large = ops.minimum(
            relative_postion_if_large,
            ops.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += ops.where(
            is_small, relative_position.to(mstype.int32), relative_postion_if_large
        )
        return relative_buckets


class RotaryEmbedding(nn.Cell):
    def __init__(
        self,
        dim,
        base=10000,
        distance_scale: Union[int, float] = 1,
        dtype: mstype.float_ = mstype.half,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        self.distance_scale = distance_scale
        self.dtype = dtype
        self.inv_freq = Tensor(inv_freq, dtype)

    def construct(self, x: Tensor, x_pos: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`Tensor` of shape ``(...)``): Positions of inputs.
        """
        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].astype(self.dtype) * self.inv_freq[None, :]  # (..., dim/2)

        # the same implementation as sat
        emb = ops.cat((freqs, freqs), axis=-1)  # (..., dim)
        emb_cos = ops.cos(emb)  # (..., dim)
        emb_sin = ops.sin(emb)  # (..., dim)

        rotate_x = ops.cat(
            [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], axis=-1
        )  # (..., dim)

        return x * emb_cos + rotate_x * emb_sin
