import torch
import torch.nn as nn
import torch.nn.functional as F
from linearattention.rms_norm import RMSNorm
from linearattention.kernel.gate_recurrent_train import chunk_gate_retention, recurrent_gate_retention
from linearattention.kernel.swiglu import swiglu


class GateRetention(nn.Module):
    def __init__(
            self,
            dim,
            n_self_heads,
            norm_eps,
            model_parallel_size=1,
            gate_logit_normalizer: int = 16,
            chunk_size=256,
    ):
        super().__init__()
        self.embed_dim = dim
        self.chunk_size = int(chunk_size)
        self.num_heads = n_self_heads // model_parallel_size
        self.head_dim = dim // n_self_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.gt_proj = nn.Linear(dim, n_self_heads, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.query_ln = RMSNorm(self.head_dim, elementwise_affine=False, eps=norm_eps)
        self.key_ln = RMSNorm(self.head_dim, elementwise_affine=False, eps=norm_eps)
        self.value_ln = RMSNorm(self.head_dim, elementwise_affine=False, eps=norm_eps)
        self.subln = RMSNorm(self.head_dim, elementwise_affine=False, eps=norm_eps)
        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
            self,
            x,
            c,
    ):
        
        bsz, tgt_len, _ = x.size()
        q = self.q_proj(x).float()
        k = self.k_proj(x).float()
        v = self.v_proj(x).float()        
        
        q = self.query_ln(q).float()
        k = self.key_ln(k).float()
        v = self.value_ln(v).float()
        g = self.g_proj(x).float()
        gt = self.gt_proj(x+c).float()

        qr = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        kr = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        gt = gt.view(bsz, tgt_len, self.num_heads).transpose(1, 2)

        gt = (F.logsigmoid(gt) / self.gate_logit_normalizer)
        o = chunk_gate_retention(qr, kr, v, gt, chunk_size=self.chunk_size)

        o = self.subln(o).transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        o = swiglu(g, o)
        o = self.out_proj(o)

        return o
