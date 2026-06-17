from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from configuration import ModelConfig, load_config
except ImportError:  # pragma: no cover
    from .configuration import ModelConfig, load_config


class FeaturePositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_input: int,
        embed_dim: int,
        num_frequencies: int,
        sigma: float,
    ) -> None:
        super().__init__()
        self.num_features = d_input
        self.embed_dim = embed_dim
        self.frequencies = nn.Parameter(torch.randn(d_input, num_frequencies) * sigma)
        self.linear = nn.Linear(2 * num_frequencies, embed_dim)
        self.spatial_pe = nn.Parameter(torch.randn(1, 1, d_input, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        angles = x * self.frequencies * 2 * math.pi
        periodic_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        projected = self.linear(periodic_features)
        return self.spatial_pe + projected


class RotaryEmbeddingBase(nn.Module):
    def __init__(self, head_dim: int, num_heads: int, base: int) -> None:
        super().__init__()
        if head_dim % 2 != 0:  # even head_dim required for RoPE pairing
            raise ValueError("head_dim must be even for rotary embeddings.")  # fail fast on invalid RoPE dimensions
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.base = base

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return x * cos + x_rot * sin


class DiscreteRotaryEmbedding(RotaryEmbeddingBase):
    def __init__(self, head_dim: int, num_heads: int, base: int) -> None:
        super().__init__(head_dim=head_dim, num_heads=num_heads, base=base)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, sequence_length, _, _ = q.shape
        positions = torch.arange(sequence_length, device=q.device, dtype=q.dtype).reshape(1, sequence_length, 1, 1)
        angles = positions * self.inv_freq.reshape(1, 1, 1, -1).to(dtype=q.dtype)
        embedding = angles.repeat_interleave(2, dim=-1)
        cos = embedding.cos()
        sin = embedding.sin()
        return self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)


class ContinuousRotaryEmbedding(RotaryEmbeddingBase):
    def __init__(self, head_dim: int, num_heads: int, base: int) -> None:
        super().__init__(head_dim=head_dim, num_heads=num_heads, base=base)
        self.time_scale = nn.Parameter(torch.ones(self.head_dim // 2))
        self.log_freqs = nn.Parameter(torch.randn(num_heads, head_dim // 2))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.unsqueeze(-1).unsqueeze(-1)
        frequencies = torch.exp(self.log_freqs).clamp(min=1e-4, max=1e2).unsqueeze(0).unsqueeze(0)
        time_scale = self.time_scale.reshape(1, 1, 1, -1)  # reshape avoids contiguous-memory assumptions
        angles = t * time_scale * frequencies
        embedding = angles.repeat_interleave(2, dim=-1)
        cos = embedding.cos()
        sin = embedding.sin()
        return self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)


class HybridContinuousRotaryEmbedding(RotaryEmbeddingBase):
    def __init__(self, head_dim: int, num_heads: int, base: int) -> None:
        super().__init__(head_dim=head_dim, num_heads=num_heads, base=base)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.time_scale = nn.Parameter(torch.ones(self.head_dim // 2))
        self.log_freqs = nn.Parameter(torch.randn(num_heads, head_dim // 2))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, sequence_length, _, _ = q.shape
        t_expanded = t.unsqueeze(-1).unsqueeze(-1)
        frequencies = torch.exp(self.log_freqs).clamp(min=1e-4, max=1e2).unsqueeze(0).unsqueeze(0)
        time_scale = self.time_scale.reshape(1, 1, 1, -1)  # reshape avoids contiguous-memory assumptions
        angles_time = t_expanded * time_scale * frequencies

        positions = torch.arange(sequence_length, device=q.device, dtype=q.dtype).reshape(1, sequence_length, 1, 1)  # reshape avoids contiguous-memory assumptions
        angles_pos = positions * self.inv_freq.reshape(1, 1, 1, -1)  # reshape avoids contiguous-memory assumptions
        angles = angles_time + angles_pos

        embedding = angles.repeat_interleave(2, dim=-1)
        cos = embedding.cos()
        sin = embedding.sin()
        return self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)


class RotaryEmbeddingFactory:
    @staticmethod
    def create(config: ModelConfig, head_dim: int) -> RotaryEmbeddingBase:
        rope_type = config.rope_type.lower()
        if rope_type == "rope":
            return DiscreteRotaryEmbedding(head_dim=head_dim, num_heads=config.num_heads, base=config.rope_base)
        if rope_type == "crope":
            return ContinuousRotaryEmbedding(head_dim=head_dim, num_heads=config.num_heads, base=config.rope_base)
        if rope_type in {"hybrid_crope", "hybrid-crope", "hybrid"}:
            return HybridContinuousRotaryEmbedding(
                head_dim=head_dim,
                num_heads=config.num_heads,
                base=config.rope_base,
            )
        raise ValueError(f"Unsupported rope type: {config.rope_type}")


class ContinuousTimeAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = config.d_model
        self.num_heads = config.num_heads  #same nb of heads for spatial/timewise
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.attention_dropout
        if config.max_dt is None:
            raise ValueError(
                "model.max_dt must be resolved before model construction. "
                "scripts/run_training.py derives it from train sequence time spans."
            )
        self.max_dt = float(config.max_dt)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.rotary = RotaryEmbeddingFactory.create(config, self.head_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:  # validate x has shape [batch, sequence, d_model]
            raise ValueError(f"x must have shape [batch, sequence, d_model], got {tuple(x.shape)}.")
        if t.ndim != 2:  # validate t has shape [batch, sequence]
            raise ValueError(f"t must have shape [batch, sequence], got {tuple(t.shape)}.")
        if x.shape[:2] != t.shape:  # validate x and t share batch and sequence dimensions
            raise ValueError(f"x and t must share [batch, sequence], got {tuple(x.shape[:2])} and {tuple(t.shape)}.")
        batch_size, sequence_length, hidden_size = x.shape
        if hidden_size != self.d_model:  # validate temporal attention hidden dimension
            raise ValueError(f"x last dimension must be d_model={self.d_model}, got {hidden_size}.")
        qkv = self.qkv(x).reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)  # reshape avoids contiguous-memory assumptions
        q, k, v = qkv.unbind(dim=2)
        q, k = self.rotary(q, k, t)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dt = t[:, :, None] - t[:, None, :]
        time_mask = ((dt >= 0) & (dt <= self.max_dt)).unsqueeze(1)
        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=x.device, dtype=torch.bool)
        ).reshape(1, 1, sequence_length, sequence_length)  # reshape avoids contiguous-memory assumptions
        attention_mask = time_mask & causal_mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False,)
        out = out.transpose(1, 2).reshape(batch_size, sequence_length, hidden_size)
        return self.out(out)


class MoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = min(config.top_k, config.num_experts)
        if self.top_k < 1:  # validate MoE routes at least one expert
            raise ValueError("top_k must be at least 1 for MoE routing.")
        self.router_noise = config.moe_router_noise  # routing noise for MoE training
        self.load_balancing_weight = config.moe_load_balancing_weight  # load-balancing coefficient for MoE training
        self.load_balancing_loss: torch.Tensor | None = None  # expose MoE auxiliary loss to the trainer
        self.last_routing: dict[str, torch.Tensor | int] | None = None
        self.gate = nn.Linear(config.d_model, config.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model * config.moe_expansion_factor),
                    nn.GELU(),
                    nn.Dropout(config.moe_dropout),
                    nn.Linear(config.d_model * config.moe_expansion_factor, config.d_model),
                )
                for _ in range(config.num_experts)
            ]
        )

    def _load_balancing_loss(self, weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.load_balancing_weight <= 0.0:  # allow disabling MoE load-balancing from config
            return weights.new_zeros(())
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).to(weights.dtype)  # selected-expert counts for MoE balancing
        expert_fraction = expert_mask.mean(dim=(0, 1, 2))  # load-balancing usage estimate across selected experts
        router_probability = weights.mean(dim=(0, 1))  # load-balancing probability estimate from router softmax
        balance_loss = self.num_experts * torch.sum(expert_fraction * router_probability)  # load-balancing for MoE training
        return balance_loss * self.load_balancing_weight  # scale MoE load-balancing loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = x.shape
        gate_logits = self.gate(x)
        if self.training and self.router_noise > 0.0:  # routing noise for MoE training
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.router_noise
        weights = torch.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_weights.dtype).eps)  # renormalize selected MoE expert weights
        self.load_balancing_loss = self._load_balancing_loss(weights, topk_indices) if self.training else weights.new_zeros(())  # load-balancing for MoE training
        self.last_routing = {
            "topk_indices": topk_indices.detach(),
            "topk_weights": topk_weights.detach(),
            "router_probabilities": weights.detach(),
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }

        x_flat = x.reshape(-1, hidden_size)
        topk_indices_flat = topk_indices.reshape(-1, self.top_k)
        topk_weights_flat = topk_weights.reshape(-1, self.top_k)
        out_flat = torch.zeros_like(x_flat)

        for expert_index in range(self.num_experts):
            batch_indices, kth_choice = torch.where(topk_indices_flat == expert_index)
            if batch_indices.numel() == 0:
                continue

            selected_tokens = x_flat[batch_indices]
            expert_output = self.experts[expert_index](selected_tokens)
            expert_weights = topk_weights_flat[batch_indices, kth_choice].unsqueeze(-1)
            out_flat.index_add_(0, batch_indices, expert_output * expert_weights)

        return out_flat.reshape(batch_size, sequence_length, hidden_size)  # reshape avoids contiguous-memory assumptions


class DenseFNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.moe_expansion_factor),
            nn.GELU(),
            nn.Dropout(config.moe_dropout),
            nn.Linear(config.d_model * config.moe_expansion_factor, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawFeatureDualAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.feature_embed_dim % config.num_heads != 0:
            raise ValueError("feature_embed_dim must be divisible by num_heads.")

        self.config = config
        self.d_input = config.resolved_d_input()
        self.feature_embedding = FeaturePositionalEmbedding(
            d_input=self.d_input,
            embed_dim=config.feature_embed_dim,
            num_frequencies=config.feature_num_frequencies,
            sigma=config.feature_sigma,
        )
        self.flattened_feature_dim = self.d_input * config.feature_embed_dim
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=config.feature_embed_dim,
            num_heads=config.num_heads, #same nb of heads for spatial/timewise
            batch_first=True,
        )
        self.projection = nn.Linear(self.flattened_feature_dim, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.temporal_attention = ContinuousTimeAttention(config)  #same nb of heads for spatial/timewise
        self.norm2 = nn.LayerNorm(config.d_model)
        self.use_moe_tail = config.use_moe and config.num_layers == 1
        self.moe = MoE(config) if self.use_moe_tail else None
        self.dense_fnn = None if self.use_moe_tail else DenseFNN(config)

    def forward(self, x: torch.Tensor, relative_time: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:  # validate x has shape [batch, sequence, features]
            raise ValueError(f"x must have shape [batch, sequence, features], got {tuple(x.shape)}.")
        if relative_time.ndim != 2:  # validate t has shape [batch, sequence]
            raise ValueError(f"t must have shape [batch, sequence], got {tuple(relative_time.shape)}.")
        if x.shape[:2] != relative_time.shape:  # validate x and t share batch and sequence dimensions
            raise ValueError(
                f"x and t must share [batch, sequence], got {tuple(x.shape[:2])} and {tuple(relative_time.shape)}."
            )
        if x.shape[-1] != self.d_input:  # validate input feature dimension before embedding
            raise ValueError(f"x last dimension must be d_input={self.d_input}, got {x.shape[-1]}.")
        batch_size, sequence_length, _ = x.shape

        embedded = self.feature_embedding(x)
        embedded = embedded.reshape(batch_size * sequence_length, self.d_input, self.config.feature_embed_dim)  # reshape avoids contiguous-memory assumptions
        spatial_attended, _ = self.spatial_attention(embedded, embedded, embedded)
        embedded = (embedded + spatial_attended).reshape(batch_size, sequence_length, self.flattened_feature_dim)  # reshape avoids contiguous-memory assumptions

        projected = self.projection(embedded)
        projected = projected + self.temporal_attention(self.norm1(projected), relative_time)
        tail = self.moe if self.moe is not None else self.dense_fnn
        if tail is None:
            raise RuntimeError("RawFeatureDualAttentionBlock has no tail module.")
        projected = projected + tail(self.norm2(projected))
        return projected


class LatentDualAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig, *, use_moe_tail: bool) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.latent_spatial_embed_dim = config.resolved_latent_spatial_embed_dim()
        self.num_latent_features = config.d_model // self.latent_spatial_embed_dim
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.latent_spatial_embed_dim,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.temporal_attention = ContinuousTimeAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = MoE(config) if use_moe_tail else None
        self.dense_fnn = None if use_moe_tail else DenseFNN(config)

    def forward(self, h: torch.Tensor, relative_time: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must have shape [batch, sequence, d_model], got {tuple(h.shape)}.")
        if h.shape[-1] != self.d_model:
            raise ValueError(f"h last dimension must be d_model={self.d_model}, got {h.shape[-1]}.")
        if relative_time.ndim != 2:
            raise ValueError(f"t must have shape [batch, sequence], got {tuple(relative_time.shape)}.")
        if h.shape[:2] != relative_time.shape:
            raise ValueError(
                f"h and t must share [batch, sequence], got {tuple(h.shape[:2])} and {tuple(relative_time.shape)}."
            )
        batch_size, sequence_length, _ = h.shape
        chunks = h.reshape(
            batch_size * sequence_length,
            self.num_latent_features,
            self.latent_spatial_embed_dim,
        )
        spatial_attended, _ = self.spatial_attention(chunks, chunks, chunks)
        h = (chunks + spatial_attended).reshape(batch_size, sequence_length, self.d_model)
        h = h + self.temporal_attention(self.norm1(h), relative_time)
        tail = self.moe if self.moe is not None else self.dense_fnn
        if tail is None:
            raise RuntimeError("LatentDualAttentionBlock has no tail module.")
        return h + tail(self.norm2(h))


class DualAttentionEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = [RawFeatureDualAttentionBlock(config)]
        for layer_index in range(1, config.num_layers):
            layers.append(
                LatentDualAttentionBlock(
                    config,
                    use_moe_tail=config.use_moe and layer_index == config.num_layers - 1,
                )
            )
        self.layers = nn.ModuleList(layers)
        final_layer = self.layers[-1]
        self.moe = (
            final_layer.moe
            if isinstance(final_layer, (RawFeatureDualAttentionBlock, LatentDualAttentionBlock))
            else None
        )

    def forward(self, x: torch.Tensor, continuous_times: torch.Tensor) -> torch.Tensor:
        if continuous_times.ndim != 2:
            raise ValueError(f"t must have shape [batch, sequence], got {tuple(continuous_times.shape)}.")
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, sequence, features], got {tuple(x.shape)}.")
        if x.shape[:2] != continuous_times.shape:
            raise ValueError(
                f"x and t must share [batch, sequence], got {tuple(x.shape[:2])} and {tuple(continuous_times.shape)}."
            )
        relative_time = continuous_times - continuous_times[:, :1]
        h = x
        for layer in self.layers:
            h = layer(h, relative_time)
        return h


class TrendClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pooling_methods = tuple(config.classifier_pooling.methods)
        self.last_k = int(config.classifier_pooling.last_k)
        input_dim = len(self.pooling_methods) * config.d_model
        aux_cfg = config.auxiliary_heads
        hidden_dim = aux_cfg.hidden_dim if aux_cfg.hidden_dim is not None else config.d_model // 2
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
        )
        self.class_head = nn.Linear(hidden_dim, config.num_classes)
        self.use_auxiliary_heads = bool(aux_cfg.enabled)
        self.movement_head = nn.Linear(hidden_dim, 1) if self.use_auxiliary_heads and aux_cfg.movement else None
        self.direction_head = nn.Linear(hidden_dim, 2) if self.use_auxiliary_heads and aux_cfg.direction else None
        self.last_auxiliary_outputs: dict[str, torch.Tensor] = {}

    def _pool(self, transformer_output: torch.Tensor) -> torch.Tensor:
        if transformer_output.ndim != 3:
            raise ValueError(
                "transformer_output must have shape [batch, sequence, d_model], "
                f"got {tuple(transformer_output.shape)}."
            )
        sequence_length = transformer_output.shape[1]
        if sequence_length < 1:
            raise ValueError("transformer_output sequence length must be >= 1.")
        effective_k = min(self.last_k, sequence_length)
        tail = transformer_output[:, -effective_k:, :]

        pooled_outputs: list[torch.Tensor] = []
        for method in self.pooling_methods:
            if method == "last":
                pooled_outputs.append(transformer_output[:, -1, :])
            elif method == "mean":
                pooled_outputs.append(tail.mean(dim=1))
            elif method == "max":
                pooled_outputs.append(tail.max(dim=1).values)
            else:
                raise RuntimeError(f"Unsupported classifier pooling method: {method}")
        return pooled_outputs[0] if len(pooled_outputs) == 1 else torch.cat(pooled_outputs, dim=-1)

    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(transformer_output)
        hidden = self.trunk(pooled)
        class_logits = self.class_head(hidden)

        self.last_auxiliary_outputs = {}
        if self.movement_head is not None:
            self.last_auxiliary_outputs["movement_logit"] = self.movement_head(hidden).squeeze(-1)
        if self.direction_head is not None:
            self.last_auxiliary_outputs["direction_logits"] = self.direction_head(hidden)

        return class_logits

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Preserve compatibility with checkpoints saved before the classifier head was split.
        legacy_key_map = {
            "head.0.": "trunk.0.",
            "head.2.": "trunk.2.",
            "head.5.": "class_head.",
        }
        for legacy_prefix, current_prefix in legacy_key_map.items():
            for suffix in ("weight", "bias"):
                legacy_key = f"{prefix}{legacy_prefix}{suffix}"
                current_key = f"{prefix}{current_prefix}{suffix}"
                if legacy_key in state_dict:
                    if current_key not in state_dict:
                        state_dict[current_key] = state_dict[legacy_key]
                    state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LobTrendTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        resolved_config = replace(config, d_input=config.resolved_d_input(config.d_input))
        self.config = resolved_config
        self.encoder = DualAttentionEncoder(resolved_config)
        self.classifier = TrendClassifier(resolved_config)
        self.moe_load_balancing_loss: torch.Tensor | None = None  # expose MoE auxiliary loss to training
        self.moe_routing: dict[str, torch.Tensor | int] | None = None
        self.auxiliary_outputs: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x, t)
        logits = self.classifier(encoded)
        self.auxiliary_outputs = self.classifier.last_auxiliary_outputs
        if self.encoder.moe is None:
            self.moe_load_balancing_loss = None
            self.moe_routing = None
        else:
            self.moe_load_balancing_loss = self.encoder.moe.load_balancing_loss  # load-balancing for MoE training
            self.moe_routing = self.encoder.moe.last_routing
        return logits


def build_model(config: ModelConfig | None = None, d_input: int | None = None) -> LobTrendTransformer:
    model_config = config or load_config().model
    if d_input is not None:
        model_config = replace(model_config, d_input=d_input)
    elif model_config.d_input is None:
        raise ValueError("d_input must be provided either in the YAML file or as an argument.")
    return LobTrendTransformer(model_config)


FeaturePE = FeaturePositionalEmbedding
RoPE = DiscreteRotaryEmbedding
cRoPE = ContinuousRotaryEmbedding
Hybrid_cRoPE = HybridContinuousRotaryEmbedding
MultiheadAttention_cRoPE = ContinuousTimeAttention
DualAttention = DualAttentionEncoder
