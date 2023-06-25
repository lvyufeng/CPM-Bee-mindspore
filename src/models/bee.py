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

from typing import Optional, Tuple, List
from typing_extensions import TypedDict

import numpy as np

from mindspore import nn, ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, Normal, Constant
from mindspore.communication.management import GlobalComm
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from ..native_layers import Encoder, EmbeddingExt, BucketPositionBias, Linear, LayerNorm, SegmentPositionEmbedding
from ..utils import Config
from ..ops import masked_fill

class CPMBeeInferenceState(TypedDict):
    buffer_position: Tensor
    buffer_context: Tensor
    buffer_sample_ids: Tensor
    buffer_num_segments: Tensor
    buffer_segments: Tensor
    buffer: List[Tuple[Tensor, Tensor]]


class CPMBeeConfig(Config):
    def __init__(
        self,
        vocab_size=30720,
        dim_model=4096,
        num_heads=64,
        dim_head=64,
        dim_ff=10240,
        num_layers=32,
        dropout_p=0.0,
        position_bias_num_buckets=256,
        position_bias_num_segment_buckets=256,
        position_bias_max_distance=2048,
        eps=1e-6,
        half: bool = True,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_num_segment_buckets = position_bias_num_segment_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        if half:
            self.dtype = mstype.half
        else:
            self.dtype = mstype.single
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules


class CPMBee(nn.Cell):
    def __init__(self, config: CPMBeeConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            mask_modules=config.mask_modules,
        )

        self.input_embedding = EmbeddingExt(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            param_init=Normal(sigma=0.02),
        )

        self.position_bias = BucketPositionBias(
            num_heads=config.num_heads,
            num_buckets=config.position_bias_num_buckets,
            num_segment_bucket=config.position_bias_num_segment_buckets,
            max_distance=config.position_bias_max_distance,
            dtype=config.dtype,
        )

    def construct(
        self,
        input: Tensor,  # (batch, seqlen) int32
        input_sub: Tensor,  # (batch, seqlen) int32
        length: Tensor,  # (batch) int32
        context: Tensor,  # (batch, seqlen) bool
        sample_ids: Tensor,  # (batch, seq_len) int32
        num_segments: Tensor,  # (batch, seq_len) int32
        segment: Tensor,  # (batch, seqlen) int32
        segment_rel_offset: Tensor,  # (batch, seq_len) int32
        segment_rel: Tensor,  # (batch, num_segment_bucket) int32
        span: Tensor,  # (batch, seqlen) int32
        ext_table_ids: Tensor,  # (ext_table_size) int32
        ext_table_sub: Tensor,  # (ext_table_size) int32
    ):
        batch = input.shape[0]
        seqlen = input.shape[1]
        # processing masks and position bias bucket

        # calc segment bucket
        segment_rel_2d = masked_fill(
            segment[:, :, None] * num_segments[:, :, None]
            + segment[:, None, :]
            + segment_rel_offset[:, :, None],
            ~(
                (sample_ids[:, :, None] == sample_ids[:, None, :])
                & (span[:, None, :] == span[:, :, None])
            ),  # not in the same span or sample
            0,  # avoid torch.gather overflow
        ).view((batch, seqlen * seqlen))

        segment_bucket = ops.gather_elements(
            input=segment_rel,
            dim=1,
            index=segment_rel_2d.astype(mstype.int32),
        ).view((batch, seqlen, seqlen))

        segment_bucket = masked_fill(
            segment_bucket,
            ~(
                (sample_ids[:, :, None] == sample_ids[:, None, :])
                & (span[:, None, :] == span[:, :, None])
            ),  # not in the same span or sample
            1,  # bucket is used for in-context samples
        )

        # directional mask
        directional_mask_2d = ops.arange(seqlen) <= ops.arange(seqlen).view((-1, 1))
        # sample mask
        sample_mask_2d = (sample_ids[:, :, None] == 0) | (
            sample_ids[:, :, None] == sample_ids[:, None, :]
        )
        # context mask
        attention_mask = context[:, None, :] | (
            ops.logical_not(context[:, :, None]) & directional_mask_2d.view((1, seqlen, seqlen))
        )
        # span mask
        attention_mask = (
            attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
        )
        # length mask
        mask_1d = (
            ops.arange(seqlen)[None, :].tile((batch, 1)) < length[:, None]
        )
        attention_mask = (
            mask_1d.view((batch, seqlen, 1)) & mask_1d.view((batch, 1, seqlen)) & attention_mask
        )
        position = ops.arange(seqlen, dtype=mstype.int32).broadcast_to((batch, seqlen))

        position = ops.stop_gradient(position)
        segment_bucket = ops.stop_gradient(segment_bucket)
        attention_mask = ops.stop_gradient(attention_mask)

        hidden_states = self.input_embedding(input, input_sub)
        position_bias = self.position_bias(position, position, segment_bucket)
        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)

        ext_table = self.input_embedding(ext_table_ids, ext_table_sub)

        logits = self.input_embedding.projection(hidden_states, ext_table)

        return logits, hidden_states

    def shard(self, dp, mp):
        # self.input_embedding.shard(dp, mp)
        self.encoder.shard(dp, mp)


class CPMBeeSimple(nn.Cell):
    def __init__(self, config: CPMBeeConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            mask_modules=config.mask_modules,
        )

        self.input_embedding = EmbeddingExt(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            param_init=Normal(sigma=0.02),
        )

        self.position_bias = BucketPositionBias(
            num_heads=config.num_heads,
            num_buckets=config.position_bias_num_buckets,
            num_segment_bucket=config.position_bias_num_segment_buckets,
            max_distance=config.position_bias_max_distance,
            dtype=config.dtype,
        )


    @staticmethod
    def prepare_data(
        input,  # (batch, seqlen) int32
        input_sub,  # (batch, seqlen) int32
        length,  # (batch) int32
        context,  # (batch, seqlen) bool
        sample_ids,  # (batch, seq_len) int32
        num_segments,  # (batch, seq_len) int32
        segment,  # (batch, seqlen) int32
        segment_rel_offset,  # (batch, seq_len) int32
        segment_rel,  # (batch, num_segment_bucket) int32
        span,  # (batch, seqlen) int32
        ext_table_ids,  # (ext_table_size) int32
        ext_table_sub,  # (ext_table_size) int32
    ):
        batch = input.shape[0]
        seqlen = input.shape[1]
        # processing masks and position bias bucket
        def masked_fill_np(inputs, mask, value):
            masked = np.full_like(inputs, value, inputs.dtype)
            outputs = np.where(mask, masked, inputs)
            return outputs

        # calc segment bucket
        segment_rel_2d = masked_fill_np(
            segment[:, :, None] * num_segments[:, :, None]
            + segment[:, None, :]
            + segment_rel_offset[:, :, None],
            ~(
                (sample_ids[:, :, None] == sample_ids[:, None, :])
                & (span[:, None, :] == span[:, :, None])
            ),  # not in the same span or sample
            0,  # avoid torch.gather overflow
        ).reshape((batch, seqlen * seqlen))

        print(segment_rel_2d.shape)
        print(segment_rel_2d.dtype)
        segment_bucket = np.take_along_axis(
            segment_rel,
            segment_rel_2d.astype(np.int32),
            axis=1
        ).reshape((batch, seqlen, seqlen))

        segment_bucket = masked_fill_np(
            segment_bucket,
            ~(
                (sample_ids[:, :, None] == sample_ids[:, None, :])
                & (span[:, None, :] == span[:, :, None])
            ),  # not in the same span or sample
            1,  # bucket is used for in-context samples
        )

        # directional mask
        directional_mask_2d = np.arange(seqlen) <= np.arange(seqlen).reshape((-1, 1))
        # sample mask
        sample_mask_2d = (sample_ids[:, :, None] == 0) | (
            sample_ids[:, :, None] == sample_ids[:, None, :]
        )
        # context mask
        attention_mask = context[:, None, :] | (
            np.logical_not(context[:, :, None]) & directional_mask_2d.reshape((1, seqlen, seqlen))
        )
        # span mask
        attention_mask = (
            attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
        )
        # length mask
        mask_1d = (
            np.tile(np.arange(seqlen)[None, :], (batch, 1)) < length[:, None]
        )
        attention_mask = (
            mask_1d.reshape((batch, seqlen, 1)) & mask_1d.reshape((batch, 1, seqlen)) & attention_mask
        )
        position = np.broadcast_to(np.arange(seqlen, dtype=np.int32), (batch, seqlen))

        return_list = [input, input_sub, position, segment_bucket, attention_mask, ext_table_ids, ext_table_sub]
        return tuple(Tensor(i) for i in return_list)

    def construct(
        self,
        input: Tensor,  # (batch, seqlen) int32
        input_sub: Tensor,  # (batch, seqlen) int32
        position: Tensor,  # (batch, seqlen) int32
        segment_bucket: Tensor,  # (batch, seqlen, seqlen) int32
        attention_mask: Tensor,  # (batch, seqlen, seqlen) bool
        ext_table_ids: Tensor,  # (ext_table_size) int32
        ext_table_sub: Tensor,  # (ext_table_size) int32
    ):
        hidden_states = self.input_embedding(input, input_sub)
        position_bias = self.position_bias(position, position, segment_bucket)

        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)

        ext_table = self.input_embedding(ext_table_ids, ext_table_sub)

        logits = self.input_embedding.projection(hidden_states, ext_table)

        return logits, hidden_states

    def shard(self, dp, mp):
        # self.input_embedding.shard(dp, mp)
        self.encoder.shard(dp, mp)

class Forward(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.model = CPMBeeSimple(config)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def construct(
        self,
        input: Tensor,  # (batch, seqlen) int32
        input_sub: Tensor,  # (batch, seqlen) int32
        position: Tensor,  # (batch, seqlen) int32
        segment_bucket: Tensor,  # (batch, seqlen, seqlen) int32
        attention_mask: Tensor,  # (batch, seqlen, seqlen) bool
        ext_table_ids: Tensor,  # (ext_table_size) int32
        ext_table_sub: Tensor,  # (ext_table_size) int32
        label: Tensor
        ):
        logits, _ = self.model(input, input_sub, position, segment_bucket, attention_mask, ext_table_ids, ext_table_sub)
        loss = self.loss_fn(logits.view((-1, logits.shape[-1])), label.view(-1))
        return loss

    def shard(self, dp, mp):
        self.model.shard(dp, mp)

class TrainStep(nn.Cell):
    def __init__(self, forward_fn, optimizer):
        super().__init__()
        self.grad_fn = ops.value_and_grad(forward_fn, None, forward_fn.trainable_params())
        self.optimizer = optimizer

        group = GlobalComm.WORLD_COMM_GROUP
        mean = _get_gradients_mean()
        degree = _get_device_num()
        self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree, group=group)

    def construct(
        self,
        input: Tensor,  # (batch, seqlen) int32
        input_sub: Tensor,  # (batch, seqlen) int32
        position: Tensor,  # (batch, seqlen) int32
        segment_bucket: Tensor,  # (batch, seqlen, seqlen) int32
        attention_mask: Tensor,  # (batch, seqlen, seqlen) bool
        ext_table_ids: Tensor,  # (ext_table_size) int32
        ext_table_sub: Tensor,  # (ext_table_size) int32
        label: Tensor
        ):
        loss, grads = self.grad_fn(input, input_sub, position, segment_bucket,
                                   attention_mask, ext_table_ids, ext_table_sub, label)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss

def init_weights(cell):
    if isinstance(cell, Linear):
        cell.weight.set_data(initializer(Normal(1.0), cell.weight.shape, cell.weight.dtype))
    elif isinstance(cell, LayerNorm):
        cell.weight.set_data(initializer(Constant(1.0), cell.weight.shape, cell.weight.dtype))
    elif isinstance(cell, EmbeddingExt):
        cell.weight.set_data(initializer(Normal(0.02), cell.weight.shape, cell.weight.dtype))
