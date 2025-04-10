import torch
import torch.nn as nn
import torch.nn.functional as F
from linearattention.kernel.gate_recurrent_infer import chunk_gate_retention
from linearattention.rms_norm import RMSNorm
from linearattention.kernel.swiglu import swiglu

# from kernel.lingche3 import chunk_gate_retention
# from rms_norm import RMSNorm
# from kernel.swiglu import swiglu

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
            c=None,
            t=None,
            incremental_state=None,
    ):

        bsz, tgt_len, _ = x.size()

        # Project inputs
        q = self.q_proj(x).float()
        k = self.k_proj(x).float()
        v = self.v_proj(x).float()
        g = self.g_proj(x).float()
        
        q = self.query_ln(q).float()
        k = self.key_ln(k).float()
        v = self.value_ln(v).float()
        # Handle conditional input
        if c is not None:
            gt = self.gt_proj(x + c).float()
        else:
            gt = self.gt_proj(x).float()

        # Reshape for attention computation
        qr = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        kr = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        gt = gt.view(bsz, tgt_len, self.num_heads).transpose(1, 2)

        # Apply gate logit normalization
        gt = (F.logsigmoid(gt) / self.gate_logit_normalizer)

        if incremental_state is not None:
            # Retrieve or initialize the last hidden state
            last_hidden_state = incremental_state.get("last_hidden_state", None)

                # Pre-filling cache: process the input in chunks
            o, last_hidden_state_out = chunk_gate_retention(
                qr, kr, v, gt, chunk_size=self.chunk_size, last_hidden_state=last_hidden_state
            )
            # Update the incremental state with the new hidden state
            incremental_state["last_hidden_state"] = last_hidden_state_out
        else:
            # Standard processing without incremental state
            o, _ = chunk_gate_retention(qr, kr, v, gt, chunk_size=self.chunk_size, last_hidden_state=None)
            
        # Post-processing
        o = self.subln(o).transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        o = swiglu(g, o)
        o = self.out_proj(o)
        return o


def create_gate_retention_model(dim=768, n_heads=12, norm_eps=1e-6):
    model = GateRetention(
        dim=dim,
        n_self_heads=n_heads,
        norm_eps=norm_eps
    )
    return model

class MultiLayerGateRetention(nn.Module):
    def __init__(self, num_layers, dim, n_heads, norm_eps):
        super().__init__()
        self.layers = nn.ModuleList([
            GateRetention(dim=dim, n_self_heads=n_heads, norm_eps=norm_eps)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, incremental_state=None):

        for i, layer in enumerate(self.layers):
            # Handle incremental_state for each layer
            if incremental_state is None:
                layer_incremental_state = None
            else:
                # Ensure incremental_state has an entry for each layer
                assert len(incremental_state) == self.num_layers, "Incremental state must match number of layers"
                layer_incremental_state = incremental_state[i]

            x = layer(x, incremental_state=layer_incremental_state)
        return x


# Example usage and verification
if __name__ == "__main__":
    # Parameters
    num_layers = 5
    dim = 768
    n_heads = 12
    batch_size = 16
    seq_length = 640  # Example sequence length
    chunk_size = 64  # Process 64 tokens at a time

    # Create the multi-layer model
    model = MultiLayerGateRetention(
        num_layers=num_layers,
        dim=dim,
        n_heads=n_heads,
        norm_eps=1e-6
    ).cuda()

    # Create sample input
    x = torch.randn(batch_size, seq_length, dim).cuda()

    # Test standard forward pass (processing the full sequence at once)
    output_full = model(x)
    print(f"Standard output shape: {output_full.shape}")

    # Initialize incremental state for caching for each layer
    incremental_state = [{} for _ in range(num_layers)]

    # Process the sequence in chunks
    num_chunks = seq_length // chunk_size  # Should be 10 for seq_length=640 and chunk_size=64
    outputs = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        x_chunk = x[:, start:end, :]
        # Process each chunk using the incremental state
        output_chunk = model(x_chunk, incremental_state=incremental_state)
        outputs.append(output_chunk)

        print(f"Processed chunk {i+1}/{num_chunks}, output shape: {output_chunk.shape}")

    # Concatenate outputs from all chunks
    output_concat = torch.cat(outputs, dim=1)
    print(f"Concatenated output shape: {output_concat.shape}")

    # Verify that the concatenated output matches the full sequence output
    assert torch.allclose(output_full, output_concat, atol=1e-5), "Outputs differ too much!"
    print("Output verified and passed all checks!")