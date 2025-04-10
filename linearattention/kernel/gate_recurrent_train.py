import time
from typing import Optional

import torch
import triton
import triton.language as tl

torch.backends.cudnn.allow_tf32 = True


@triton.jit
def _fwd_recurrence(
        S, d,
        O,
        NUM_HEAD, NUM_BLOCK,
        D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
        BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
        last_kv: Optional[tl.tensor]
):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0,
                                                                                                             BLOCK_MODEL_K)[
                                                                                                   :,
                                                                                                   None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
        0, BLOCK_MODEL_V)[None, :]

    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0,
                                                                                                             BLOCK_MODEL_K)[
                                                                                                   :,
                                                                                                   None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
        0, BLOCK_MODEL_V)[None, :]

    if last_kv is not None:
        last_kv = last_kv + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0,
                                                                                                                 BLOCK_MODEL_K)[
                                                                                                       :,
                                                                                                       None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
            0, BLOCK_MODEL_V)[None, :]
        acc = tl.load(last_kv).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V
    d = d + offset_bh * NUM_BLOCK
    for i in range(NUM_BLOCK - 1):
        d_i = tl.load(d)
        S_i = tl.load(S)
        acc = acc * d_i + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        d += 1
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V

    ## NUM_SPLIT_K/V. K/V dimension split into NUM_SPLIT_K/V parts with equal size BLOCK_MODEL


@triton.jit
def _bwd_recurrence(
        S, d,
        DI, DG, DL, DS,
        NUM_HEAD, NUM_BLOCK,
        D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
        BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,

):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    # offset_h = offset_bh % NUM_HEAD
    NUM_K = D_MODEL_K // BLOCK_MODEL_K
    NUM_V = D_MODEL_V // BLOCK_MODEL_V
    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0,
                                                                                                             BLOCK_MODEL_K)[
                                                                                                   :,
                                                                                                   None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
        0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0,
                                                                                                               BLOCK_MODEL_K)[
                                                                                                     :,
                                                                                                     None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
        0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0,
                                                                                                               BLOCK_MODEL_K)[
                                                                                                     :,
                                                                                                     None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
        0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    DG = DG + offset_bh * NUM_BLOCK * NUM_K * NUM_V + offset_d * NUM_V + offset_s + (NUM_BLOCK - 2) * NUM_K * NUM_V

    d = d + offset_bh * NUM_BLOCK + (NUM_BLOCK - 1)

    Dacc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        d_i = tl.load(d)
        Dacc = Dacc * d_i + DS_i
        DG_i = tl.sum(Dacc * S_i.to(tl.float32))

        tl.store(DG, DG_i.to(DG.dtype.element_ty))
        tl.store(DI, Dacc.to(DI.dtype.element_ty))

        S -= D_MODEL_K * D_MODEL_V
        DI -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
        DG -= NUM_K * NUM_V
        d -= 1

    DL = DL + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:,
                                                                                         None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(
        0, BLOCK_MODEL_V)[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(d)
    Dacc = Dacc * d_i + DS_i
    tl.store(DL, Dacc.to(DL.dtype.element_ty))


class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        cross_decay = cross_decay.contiguous()
        kv = kv.contiguous()

        B, H, N, D_k, D_v = kv.shape
        output = torch.empty_like(kv)
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16

        assert D_k % BLOCK_MODEL_K == 0
        assert D_v % BLOCK_MODEL_V == 0

        grid = (B * H, D_k // BLOCK_MODEL_K, D_v // BLOCK_MODEL_V)
        ctx.grid = grid
        ctx.have_last_kv = last_kv is not None
        ctx.BLOCK_MODEL_K = BLOCK_MODEL_K
        ctx.BLOCK_MODEL_V = BLOCK_MODEL_V

        _fwd_recurrence[grid](
            kv,
            cross_decay,
            output,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N, NUM_HEAD=H,
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
            last_kv=last_kv
        )

        ctx.save_for_backward(output, cross_decay)
        return output

    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()

        output, cross_decay = ctx.saved_tensors

        B, H, N, D_k, D_v = output.shape

        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16

        grid = (B * H, D_k // BLOCK_MODEL_K, D_v // BLOCK_MODEL_V)

        DI = torch.empty_like(DO)
        DG = torch.empty(B * H, N, D_k // BLOCK_MODEL_K, D_v // BLOCK_MODEL_V, device=cross_decay.device,
                         dtype=cross_decay.dtype)
        DL = torch.empty(B, H, D_k, D_v, device=output.device, dtype=output.dtype)
        _bwd_recurrence[grid](
            output, cross_decay,
            DI, DG, DL, DO,
            NUM_HEAD=H, NUM_BLOCK=N,
            D_MODEL_K=D_k,
            D_MODEL_V=D_v,
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
        )

        DI[:, :, -1] = 0
        DG[:, -1] = 0
        DG = DG.view(B, H, N, -1).sum(dim=-1)
        return DI, DG, DL if ctx.have_last_kv else None


def cross_chunk(q, k, v, g, last_hidden_state=None):
    #kv = k.transpose(-1, -2) @ (v * (-g + g[:, :, :, -1, None]).exp()[..., None].to(v.dtype))
    # Remove the intra-chunk init decay.
    kv = k.transpose(-1, -2) @ v
    #cross_decay = g[:, :, :, -1].exp().to(kv.dtype)
    cross_decay = g.mean(dim=-1).exp().to(kv.dtype)
    S = chunk_gate_recurrent(kv, cross_decay, last_hidden_state).float()
    g_cumsum = g.float().cumsum(-1)
    cross = (q * g_cumsum[..., None].exp()) @ S
    
    # contains_nan = torch.isinf(cross).any()
    # #contains_nan = torch.isnan(o).any()
    # if contains_nan :
    #     import pdb
    #     pdb.set_trace()
    return cross

@torch.compile
def inner_chunk(q, k, v):
    attn = q @ k.transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    inner = attn @ v
    return inner


def chunk_gate_retention(q, k, v, g, chunk_size=64, last_hidden_state=None):
    bsz, num_head, tgt_len, key_dim = q.shape
    head_dim = v.shape[-1]
    num_chunk = tgt_len // chunk_size
    q = q.view(bsz, num_head, num_chunk, chunk_size, key_dim)
    k = k.view(bsz, num_head, num_chunk, chunk_size, key_dim) * (key_dim ** -0.5)
    v = v.view(bsz, num_head, num_chunk, chunk_size, head_dim)
    g = g.view(bsz, num_head, num_chunk, chunk_size)
    #g = g.float().cumsum(-1) #remove it
    q = q.float()
    k = k.float()
    v = v.float()
    cross = cross_chunk(q, k, v, g, last_hidden_state=last_hidden_state)
    inner = inner_chunk(q, k, v)
    o = cross + inner

    return o.view(bsz, num_head, tgt_len, head_dim)


# for long sequence parallelism
def hier_chunk_gate_retention(q, k, v, g, chunk_size=64, hier_chunk_size=16384):
    bsz, num_head, tgt_len, key_dim = q.shape
    head_dim = v.shape[-1]
    num_hier_chunk = tgt_len // hier_chunk_size
    assert tgt_len == num_hier_chunk * hier_chunk_size

    q = q.view(bsz, num_head, num_hier_chunk, hier_chunk_size, key_dim)
    k = k.view(bsz, num_head, num_hier_chunk, hier_chunk_size, key_dim)
    v = v.view(bsz, num_head, num_hier_chunk, hier_chunk_size, head_dim)
    g = g.view(bsz, num_head, num_hier_chunk, hier_chunk_size)
    hier_cross = cross_chunk(q, k * (key_dim ** -0.5), v, g.float().cumsum(-1)).view(bsz, num_head, tgt_len, head_dim)

    qi = q.transpose(1, 2).reshape(bsz * num_hier_chunk, num_head, hier_chunk_size, key_dim)
    ki = k.transpose(1, 2).reshape(bsz * num_hier_chunk, num_head, hier_chunk_size, key_dim)
    vi = v.transpose(1, 2).reshape(bsz * num_hier_chunk, num_head, hier_chunk_size, head_dim)
    gi = g.transpose(1, 2).reshape(bsz * num_hier_chunk, num_head, hier_chunk_size)
    inner_cross = chunk_gate_retention(qi, ki, vi, gi, chunk_size)

    inner_cross = inner_cross.view(bsz, num_hier_chunk, num_head, hier_chunk_size, head_dim).transpose(1, 2).reshape(
        bsz, num_head, tgt_len, head_dim)
    o = hier_cross + inner_cross
    return o


def recurrent_gate_retention(q, k, v, g, incremental_state):
    bsz, num_head, _, key_dim = q.shape
    k *= key_dim ** -0.5
    g = g.view(bsz, num_head, 1, 1).float().exp()
    kv = k.transpose(-1, -2) * v
    if "last_hidden_state" in incremental_state:
        prev_kv = incremental_state["last_hidden_state"]
        kv += prev_kv * g.to(prev_kv.dtype)

    incremental_state["last_hidden_state"] = kv
    o = q @ kv
    return o


def parallel_gate_retention(q, k, v, g):
    k = k * (q.shape[-1] ** -0.5)
    causal_mask = torch.full([q.shape[-2], q.shape[-2]], float("-inf"), device=q.device).triu(1).type_as(q)
    g = g.float().cumsum(-1)
    mask = g[..., None] - g[..., None, :] + causal_mask
    mask = mask.exp()

    attn = q @ k.transpose(-1, -2)
    attn = attn * mask.to(attn.dtype)
    o = attn @ v
    return o


def naive_kv_recurrent(kv, cross_decay, last_kv=None):
    BSZ, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = kv.shape
    kv_recurrent = []
    kv_state = torch.zeros(BSZ, NUM_HEAD, D_MODEL_K, D_MODEL_V, dtype=kv.dtype,
                           device="cuda") if last_kv is None else last_kv
    # accumulate kv by loop
    for i in range(NUM_BLOCK):
        kv_recurrent.append(kv_state)
        kv_state = kv_state * cross_decay[:, :, i, None, None] + kv[:, :, i]

    kv_recurrent = torch.stack(kv_recurrent, dim=2)
    return kv_recurrent


chunk_gate_recurrent = ChunkGateRecurrent.apply
