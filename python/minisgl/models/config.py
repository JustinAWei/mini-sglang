from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # Qwen3.5 hybrid GDN/attention fields
    full_attention_interval: int = 0  # 0 = all layers are standard attention
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def is_hybrid(self) -> bool:
        return self.full_attention_interval > 0

    @property
    def layers_block_type(self) -> List[str]:
        if self.full_attention_interval <= 0:
            return ["attention"] * self.num_layers
        result = []
        for i in range(self.num_layers):
            if (i + 1) % self.full_attention_interval == 0:
                result.append("attention")
            else:
                result.append("linear_attention")
        return result

    @property
    def num_kv_layers(self) -> int:
        """Number of layers that use KV cache (attention layers only)."""
        return sum(1 for t in self.layers_block_type if t == "attention")

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # For VLM models (e.g. Qwen3.5), extract text_config and propagate
        # top-level attrs that may not be on the nested config.
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Partial rotary support (Qwen3.5 uses partial_rotary_factor=0.25)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial_rotary_factor)

        # Llama/Qwen: rope_theta is a direct attr; Mistral: inside rope_scaling;
        # transformers 5.x: inside rope_parameters
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(config, "rope_parameters", rope_scaling)
            if isinstance(rope_params, dict):
                rope_theta = rope_params.get("rope_theta", 10000.0)
            else:
                rope_theta = 10000.0
        # Skip multimodal RoPE scaling (mrope_sections)
        if isinstance(rope_scaling, dict) and (
            "mrope_sections" in rope_scaling or "mrope_section" in rope_scaling
        ):
            rope_scaling = None

        # Qwen3.5 hybrid GDN/attention config
        full_attention_interval = getattr(config, "full_attention_interval", 0)
        linear_num_key_heads = getattr(config, "linear_num_key_heads", 0)
        linear_num_value_heads = getattr(config, "linear_num_value_heads", 0)
        linear_key_head_dim = getattr(config, "linear_key_head_dim", 0)
        linear_value_head_dim = getattr(config, "linear_value_head_dim", 0)
        linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            full_attention_interval=full_attention_interval,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
        )
