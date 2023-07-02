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

from ..native_layers import Encoder, EmbeddingExt, BucketPositionBias, Linear, LayerNorm
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

        self.logical_and = ops.LogicalAnd()

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
                self.logical_and(
                (sample_ids[:, :, None] == sample_ids[:, None, :]),
                (span[:, None, :] == span[:, :, None]))
            ),  # not in the same span or sample
            0,  # avoid torch.gather overflow
        ).view((batch, seqlen * seqlen))

        # segment_bucket = ops.gather_elements(
        #     input=segment_rel,
        #     dim=1,
        #     index=segment_rel_2d.astype(mstype.int32),
        # ).view((batch, seqlen, seqlen))
        segment_bucket = ops.ones((batch, seqlen, seqlen), mstype.int32)

        segment_bucket = masked_fill(
            segment_bucket,
            ~(
                self.logical_and((sample_ids[:, :, None] == sample_ids[:, None, :]),
                (span[:, None, :] == span[:, :, None]))
            ),  # not in the same span or sample
            1,  # bucket is used for in-context samples
        )

        # directional mask
        directional_mask_2d = ops.arange(seqlen) <= ops.arange(seqlen).view((-1, 1))
        # sample mask
        sample_mask_2d = ops.bitwise_or((sample_ids[:, :, None] == 0).astype(mstype.int32),
            (sample_ids[:, :, None] == sample_ids[:, None, :]).astype(mstype.int32)
        ).astype(mstype.bool_)

        # context mask
        attention_mask = ops.logical_or(context[:, None, :], 
            (self.logical_and(ops.logical_not(context[:, :, None]),
                            directional_mask_2d.view((1, seqlen, seqlen)))
        ))
        # span mask
        attention_mask = (
            attention_mask.astype(mstype.int32) & sample_mask_2d.astype(mstype.int32) & \
                            (span[:, None, :] == span[:, :, None]).astype(mstype.int32)
        )
        # length mask
        mask_1d = (
            ops.arange(seqlen, dtype=mstype.int32)[None, :].tile((batch, 1)) < length[:, None]
        )

        mask_1d_and = self.logical_and(mask_1d.view((batch, seqlen, 1)), mask_1d.view((batch, 1, seqlen)))
        attention_mask = mask_1d_and.astype(mstype.int32) & attention_mask

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
        self.input_embedding.shard(dp, mp)
        self.encoder.shard(dp, mp)
        self.logical_and.shard(((1, 1, 1), (1, 1, 1)))


class BeeForward(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.model = CPMBee(config)
        # self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(True, 'mean')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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
        label: Tensor
        ):
        # ext_table_ids = ext_table_ids[0]
        # ext_table_sub = ext_table_sub[0]
        logits, _ = self.model(input, input_sub, length, context, sample_ids,
                               num_segments, segment, segment_rel_offset, segment_rel, span,
                               ext_table_ids, ext_table_sub)
        logits = logits.astype(mstype.float32)
        loss = self.loss_fn(logits.view((-1, logits.shape[-1])), label.view(-1))
        return loss

    def shard(self, dp, mp):
        self.model.shard(dp, mp)

from mindspore.nn.wrap.loss_scale import _grad_scale

class TrainOneStep(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_sense):
        super().__init__(network, optimizer, scale_sense)
    
    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status = Tensor([0] * 8, mstype.int32)

        scaling_sens_filled = ops.ones_like(loss) * scaling_sens.astype(loss.dtype)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            grads = ops.clip_by_global_norm(grads, 1.0)
            loss = ops.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens

def init_weights(cell):
    if isinstance(cell, Linear):
        cell.weight.set_data(initializer(Normal(1.0), cell.weight.shape, cell.weight.dtype))
    elif isinstance(cell, LayerNorm):
        cell.weight.set_data(initializer(Constant(1.0), cell.weight.shape, cell.weight.dtype))
    elif isinstance(cell, EmbeddingExt):
        cell.weight.set_data(initializer(Normal(0.02), cell.weight.shape, cell.weight.dtype))
