from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearOProj,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.layers.linear import _LinearTPImpl
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


# ---------------------------------------------------------------------------
# GDN (Gated Delta Net) components
# ---------------------------------------------------------------------------


class DepthwiseConv1d(BaseOP):
    """Depthwise 1D convolution for GDN layers (no bias)."""

    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GDNAttn(BaseOP):
    """Gated Delta Net attention with pure PyTorch recurrent implementation."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id

        hidden_size = config.hidden_size
        self._num_q_heads = config.linear_num_key_heads
        self._num_k_heads = config.linear_num_key_heads
        self._num_v_heads = config.linear_num_value_heads
        self._head_k_dim = config.linear_key_head_dim
        self._head_v_dim = config.linear_value_head_dim
        self._conv_kernel_dim = config.linear_conv_kernel_dim

        q_dim = self._num_q_heads * self._head_k_dim
        k_dim = self._num_k_heads * self._head_k_dim
        v_dim = self._num_v_heads * self._head_v_dim
        self._q_dim = q_dim
        self._k_dim = k_dim
        self._v_dim = v_dim
        self._qkv_dim = q_dim + k_dim + v_dim
        self._out_dim = v_dim

        # Projections
        self.in_proj_qkv = LinearReplicated(hidden_size, self._qkv_dim, has_bias=False)
        self.in_proj_z = LinearReplicated(hidden_size, self._out_dim, has_bias=False)
        self.in_proj_b = LinearReplicated(hidden_size, self._num_v_heads, has_bias=False)
        self.in_proj_a = LinearReplicated(hidden_size, self._num_v_heads, has_bias=False)
        self.out_proj = LinearReplicated(self._out_dim, hidden_size, has_bias=False)

        # Conv1d (depthwise, no bias)
        self.conv1d = DepthwiseConv1d(self._qkv_dim, self._conv_kernel_dim)

        # Parameters
        self.A_log = torch.empty(self._num_v_heads)
        self.dt_bias = torch.empty(self._num_v_heads)

        # Norm: per-head RMS norm with weight [head_v_dim]
        self.norm = RMSNorm(self._head_v_dim, config.rms_norm_eps)

        # State tensors (allocated later via allocate_states)
        self._conv_state: torch.Tensor | None = None
        self._recurrent_state: torch.Tensor | None = None

    def allocate_states(self, max_batch_size: int, device: torch.device, dtype: torch.dtype):
        self._conv_state = torch.zeros(
            max_batch_size, self._qkv_dim, self._conv_kernel_dim,
            device=device, dtype=dtype,
        )
        self._recurrent_state = torch.zeros(
            max_batch_size, self._num_v_heads, self._head_k_dim, self._head_v_dim,
            device=device, dtype=dtype,
        )

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalization matching the FLA library implementation."""
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

    def _apply_gated_norm(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Per-head RMS norm with SiLU gating: rms_norm(o_per_head) * silu(z_per_head)."""
        B = o.shape[0]
        o = o.view(B, self._num_v_heads, self._head_v_dim)
        z = z.view(B, self._num_v_heads, self._head_v_dim)
        # Apply per-head norm using the shared norm weights
        o = o.reshape(-1, self._head_v_dim)
        o = self.norm.forward(o)
        o = o.view(B, self._num_v_heads, self._head_v_dim)
        # Compute silu(z) in float32 for precision
        o = o * F.silu(z.float()).to(o.dtype)
        return o.reshape(B, -1)

    def _gdn_decode(
        self,
        qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        req_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Batch decode: one token per request."""
        assert self._conv_state is not None and self._recurrent_state is not None
        B = qkv.shape[0]

        # Conv1d update: shift left, insert new, convolve (no bias)
        conv_state = self._conv_state[req_indices]
        conv_state = torch.roll(conv_state, -1, dims=-1)
        conv_state[:, :, -1] = qkv
        qkv_conv = (conv_state * self.conv1d.weight.squeeze(1)).sum(-1)
        qkv_conv = F.silu(qkv_conv)
        self._conv_state[req_indices] = conv_state

        # Split QKV
        q, k, v = qkv_conv.split([self._q_dim, self._k_dim, self._v_dim], dim=-1)
        q = q.view(B, self._num_q_heads, self._head_k_dim).float()
        k = k.view(B, self._num_k_heads, self._head_k_dim).float()
        v = v.view(B, self._num_v_heads, self._head_v_dim).float()

        # L2 normalize Q and K, then scale Q
        q = self._l2norm(q, dim=-1)
        k = self._l2norm(k, dim=-1)
        scale = 1.0 / (self._head_k_dim ** 0.5)
        q = q * scale

        # Gating (computed in fp32 for numerical stability)
        g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())
        beta = torch.sigmoid(b.float())

        # Recurrent update
        h = self._recurrent_state[req_indices].float()
        h = h * torch.exp(g).unsqueeze(-1).unsqueeze(-1)
        delta_v = v - torch.einsum("bhkv,bhk->bhv", h, k)
        delta_v = delta_v * beta.unsqueeze(-1)
        h = h + torch.einsum("bhk,bhv->bhkv", k, delta_v)
        o = torch.einsum("bhkv,bhk->bhv", h, q)
        self._recurrent_state[req_indices] = h.to(self._recurrent_state.dtype)

        return o.to(qkv.dtype).reshape(B, -1)

    def _gdn_prefill_single(
        self,
        qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        req_idx: int,
    ) -> torch.Tensor:
        """Sequential prefill for one request."""
        assert self._conv_state is not None and self._recurrent_state is not None
        T = qkv.shape[0]
        W = self._conv_kernel_dim

        # Reset state for fresh prefill
        self._conv_state[req_idx].zero_()
        self._recurrent_state[req_idx].zero_()

        # Causal conv1d via F.conv1d with left padding (no bias)
        qkv_t = qkv.unsqueeze(0).transpose(1, 2)
        qkv_padded = F.pad(qkv_t, (W - 1, 0))
        qkv_conv = F.conv1d(qkv_padded, self.conv1d.weight, bias=None, groups=self._qkv_dim)
        qkv_conv = F.silu(qkv_conv).squeeze(0).transpose(0, 1)

        # Save conv state (last W raw input values)
        if T >= W:
            self._conv_state[req_idx] = qkv[-W:].T.contiguous()
        else:
            self._conv_state[req_idx].zero_()
            self._conv_state[req_idx, :, W - T:] = qkv.T.contiguous()

        # Split QKV
        q, k, v = qkv_conv.split([self._q_dim, self._k_dim, self._v_dim], dim=-1)
        q = q.view(T, self._num_q_heads, self._head_k_dim).float()
        k = k.view(T, self._num_k_heads, self._head_k_dim).float()
        v = v.view(T, self._num_v_heads, self._head_v_dim).float()

        # L2 normalize Q and K, then scale Q
        q = self._l2norm(q, dim=-1)
        k = self._l2norm(k, dim=-1)
        scale = 1.0 / (self._head_k_dim ** 0.5)
        q = q * scale

        # Gating
        g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())
        beta = torch.sigmoid(b.float())

        # Sequential recurrence
        h = torch.zeros(
            self._num_v_heads, self._head_k_dim, self._head_v_dim,
            device=qkv.device, dtype=torch.float32,
        )
        outputs = []
        for t in range(T):
            h = h * torch.exp(g[t]).unsqueeze(-1).unsqueeze(-1)
            delta_v = v[t] - torch.einsum("hkv,hk->hv", h, k[t])
            delta_v = delta_v * beta[t].unsqueeze(-1)
            h = h + torch.einsum("hk,hv->hkv", k[t], delta_v)
            o = torch.einsum("hkv,hk->hv", h, q[t])
            outputs.append(o)

        self._recurrent_state[req_idx] = h.to(self._recurrent_state.dtype)
        return torch.stack(outputs, dim=0).to(qkv.dtype).reshape(T, -1)

    @nvtx_annotate("GDN_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch

        qkv = self.in_proj_qkv.forward(x)
        z = self.in_proj_z.forward(x)
        b_val = self.in_proj_b.forward(x)
        a_val = self.in_proj_a.forward(x)

        if batch.is_decode:
            req_indices = torch.tensor(
                [r.table_idx for r in batch.padded_reqs],
                dtype=torch.long, device=x.device,
            )
            o = self._gdn_decode(qkv, a_val, b_val, req_indices)
        else:
            outputs = []
            offset = 0
            for req in batch.padded_reqs:
                length = req.extend_len
                o_req = self._gdn_prefill_single(
                    qkv[offset:offset + length],
                    a_val[offset:offset + length],
                    b_val[offset:offset + length],
                    req.table_idx,
                )
                outputs.append(o_req)
                offset += length
            o = torch.cat(outputs, dim=0)

        # Gated per-head RMS norm
        o = self._apply_gated_norm(o, z)
        return self.out_proj.forward(o)


# ---------------------------------------------------------------------------
# Qwen3.5 attention with output gate (q_proj includes gate)
# ---------------------------------------------------------------------------


class Qwen3_5RopeAttn(BaseOP):
    """Attention with fused q+gate projection and per-head Q/K norms."""

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        num_qo = config.num_qo_heads
        num_kv = config.num_kv_heads

        # q_proj in checkpoint is [2*num_qo*head_dim, hidden] (Q + output gate interleaved per head)
        # After merge: qkv_proj = cat([q_gate_interleaved, k, v], dim=0)
        q_gate_dim = 2 * num_qo * head_dim
        k_dim = num_kv * head_dim
        v_dim = num_kv * head_dim
        total_dim = q_gate_dim + k_dim + v_dim

        self.qkv_proj = _LinearTPImpl(
            full_isize=config.hidden_size,
            full_osize=total_dim,
            local_isize=config.hidden_size,
            local_osize=total_dim,
            has_bias=False,
        )

        self._q_gate_dim = q_gate_dim
        self._q_dim = num_qo * head_dim
        self._k_dim = k_dim
        self._v_dim = v_dim
        self._num_qo = num_qo
        self._head_dim = head_dim

        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=num_qo,
            num_kv_heads=num_kv,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )

        self.o_proj = LinearOProj(
            head_dim * num_qo,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x

        # Split: q_gate (interleaved per head), k, v
        q_gate, k, v = qkv.split(
            [self._q_gate_dim, self._k_dim, self._v_dim], dim=-1
        )
        del qkv

        # De-interleave q and gate: [B, num_heads, head_dim*2] -> chunk -> [B, num_heads, head_dim] each
        q_gate = q_gate.view(-1, self._num_qo, self._head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)  # each [B, num_heads, head_dim]
        gate = gate.reshape(-1, self._q_dim)  # [B, num_heads * head_dim]
        q = q.reshape(-1, self._q_dim)

        # Recombine q+k+v for AttentionLayer (it expects merged qkv)
        qkv_for_attn = torch.cat([q, k, v], dim=-1)
        del q, k, v

        o = self.attn.forward(qkv_for_attn)
        del qkv_for_attn

        # Apply output gate
        o = o * torch.sigmoid(gate)
        del gate

        return self.o_proj.forward(o)


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class Qwen3_5GDNDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.linear_attn = GDNAttn(config, layer_id)
        self.mlp = GatedMLP(config, layer_id=layer_id)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.linear_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3_5AttentionDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3_5RopeAttn(config, layer_id)
        self.mlp = GatedMLP(config, layer_id=layer_id)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


# ---------------------------------------------------------------------------
# Model & CausalLM wrapper
# ---------------------------------------------------------------------------


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        layers = []
        block_types = config.layers_block_type
        for layer_id in range(config.num_layers):
            if block_types[layer_id] == "attention":
                layers.append(Qwen3_5AttentionDecoderLayer(config, layer_id))
            else:
                layers.append(Qwen3_5GDNDecoderLayer(config, layer_id))
        self.layers = OPList(layers)
        self.norm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._config = config
        super().__init__()

    def allocate_gdn_states(
        self, max_batch_size: int, device: torch.device, dtype: torch.dtype
    ):
        for layer in self.model.layers.op_list:
            if isinstance(layer, Qwen3_5GDNDecoderLayer):
                layer.linear_attn.allocate_states(max_batch_size, device, dtype)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3_5ForCausalLM"]
