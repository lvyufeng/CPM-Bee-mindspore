import torch
import numpy as np
import mindspore as ms
from src.models import CPMBeeConfig, CPMBee
from src.dataset import SimpleDataset
from src.tokenizers import CPMBeeTokenizer
from src.data_converter import _MixedDatasetConfig, _MixedDatasetSaver

from cpm_live.models import CPMBeeTorch
from cpm_live.models import CPMBeeConfig as CPMBeeTorchConfig


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
        batch['ext_sub'],  # (ext_table_size) int32
    )

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
    length = np.random.randint(10, seqlen, (batch,)).astype(np.int32)
    context = np.full((batch, seqlen), 1).astype(np.bool_)
    sample_ids = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    num_segments = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    segment = np.random.randint(0, 1, (batch, seqlen)).astype(np.int32)
    segment_rel_offset = np.random.randint(0, 1, (batch, seqlen)).astype(np.int32)
    segment_rel = np.random.randint(0, 1, (batch, num_segment_bucket)).astype(np.int32)
    span = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    ext_table_ids = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)
    ext_table_sub = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)
    return input, input_sub, length, context, sample_ids, \
        num_segments, segment, segment_rel, segment_rel_offset, \
        span, ext_table_ids, ext_table_sub

def test_ms_pt_cmp():
    # default CPM-Bee 10b
    # config = CPMBeeConfig()
    config = CPMBeeConfig(86580, 128, 8, 8, 512, 4, position_bias_num_segment_buckets=2048, half=True)
    pt_config = CPMBeeTorchConfig(86580, 128, 8, 8, 512, 4, position_bias_num_segment_buckets=2048, half=True)
    ms_model = CPMBee(config)
    pt_model = CPMBeeTorch(pt_config).cuda()

    # pt_states = torch.load('/mnt/data1/cpm-bee/pytorch_model.bin')
    # pt_model.load_state_dict(pt_states)

    pt_states = pt_model.state_dict()
    for name, param in ms_model.parameters_and_names():
        if 'gamma' in name:
            name = name.replace('gamma', 'weight')
        if 'beta' in name:
            name = name.replace('beta', 'bias')
        if 'embedding_table' in name:
            name = name.replace('embedding_table', 'weight')
        
        pt_states[name] = torch.tensor(param.asnumpy())
        # param.set_data(ms.Tensor(pt_states[name].cpu().detach().numpy()))

    pt_model.load_state_dict(pt_states)

    # ms_dict = ms.load_checkpoint('/mnt/data1/cpm-bee/mindspore_model.ckpt')
    # ms.load_param_into_net(ms_model, ms_dict)
    pt_model.eval()
    ms_model.set_train(False)

    inputs_np = make_tensor(2, 2048, config.position_bias_num_segment_buckets, 256)
    # inputs_np = make_real_tensor()

    inputs_ms = [ms.Tensor(i) for i in inputs_np]
    inputs_pt = [torch.tensor(i).cuda() for i in inputs_np]

    logits_pt, _ = pt_model(*inputs_pt)
    logits_ms, _ = ms_model(*inputs_ms)

    print(logits_pt, logits_ms)
    print('\nlogits ms:', logits_ms.shape)
    print('logits pt:', logits_pt.shape)
    assert np.allclose(logits_ms.asnumpy(),
                       logits_pt.cpu().detach().numpy(),
                       1e-3, 1e-3)
