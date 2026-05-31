#!/usr/bin/env python3
"""Reusable H1 energy-conditioned adapter scaffold.

This module is intentionally model-agnostic: it defines the project-local
adapter interface, origin-time conditioning guard, synthetic smoke helpers and a
tiny frozen-shell wrapper used before wiring the method into Chronos-2 or
TimesFM 2.5 runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


CONDITIONING_FEATURES: tuple[str, ...] = (
    "domain_family_load",
    "domain_family_pv",
    "domain_family_industrial",
    "domain_family_microgrid",
    "domain_family_aidc",
    "horizon_4h",
    "horizon_24h",
    "native_resolution_minutes",
    "context_length_steps",
    "horizon_steps",
    "context_mean",
    "context_std",
    "context_cv",
    "context_zero_fraction",
    "context_recent_ramp",
    "context_peak_concentration",
    "label_reliability",
    "known_future_calendar_phase",
)

ALLOWED_SOURCE_TAGS: tuple[str, ...] = (
    "static_metadata",
    "task_geometry",
    "train_visible_statistic",
    "context_visible_statistic",
    "known_future_calendar",
)

FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "future_target",
    "target_window",
    "validation_label",
    "test_label",
    "y_true_future",
    "future_measured",
    "future_weather_observed",
    "future_irradiance_observed",
    "residual",
    "forecast_error",
    "post_hoc",
    "wape",
    "rmse",
    "mae",
    "metric",
    "rank",
)


class InformationBoundaryError(ValueError):
    """Raised when a conditioning feature violates the origin-time boundary."""


@dataclass(frozen=True)
class ConditioningSpec:
    """Feature names and their visibility/source tags."""

    feature_names: tuple[str, ...] = CONDITIONING_FEATURES
    source_tags: Mapping[str, str] | None = None

    def validated(self) -> "ConditioningSpec":
        validate_conditioning_feature_names(self.feature_names)
        source_tags = self.source_tags or default_source_tags()
        missing = [name for name in self.feature_names if name not in source_tags]
        invalid = {
            name: source_tags[name]
            for name in self.feature_names
            if name in source_tags and source_tags[name] not in ALLOWED_SOURCE_TAGS
        }
        if missing or invalid:
            raise InformationBoundaryError(
                f"Invalid conditioning sources: missing={missing}, invalid={invalid}"
            )
        return ConditioningSpec(tuple(self.feature_names), dict(source_tags))


@dataclass
class EnergyConditioningBatch:
    """Tensor batch plus audited feature metadata."""

    values: torch.Tensor
    feature_names: tuple[str, ...]
    source_tags: Mapping[str, str]
    origin_time_visible: bool = True

    def validate(self) -> "EnergyConditioningBatch":
        spec = ConditioningSpec(self.feature_names, self.source_tags).validated()
        if self.values.ndim != 2:
            raise InformationBoundaryError(f"Conditioning tensor must be 2-D, got {tuple(self.values.shape)}")
        if self.values.shape[1] != len(spec.feature_names):
            raise InformationBoundaryError(
                f"Conditioning width {self.values.shape[1]} != feature count {len(spec.feature_names)}"
            )
        if not self.origin_time_visible:
            raise InformationBoundaryError("Conditioning batch is not origin-time visible.")
        return self

    def to(self, device: torch.device | str) -> "EnergyConditioningBatch":
        return EnergyConditioningBatch(
            values=self.values.to(device),
            feature_names=self.feature_names,
            source_tags=dict(self.source_tags),
            origin_time_visible=self.origin_time_visible,
        )


def default_source_tags() -> dict[str, str]:
    tags: dict[str, str] = {}
    for name in CONDITIONING_FEATURES:
        if name.startswith("domain_family_") or name == "label_reliability":
            tags[name] = "static_metadata"
        elif name in {"horizon_4h", "horizon_24h", "native_resolution_minutes", "context_length_steps", "horizon_steps"}:
            tags[name] = "task_geometry"
        elif name == "known_future_calendar_phase":
            tags[name] = "known_future_calendar"
        else:
            tags[name] = "context_visible_statistic"
    return tags


def validate_conditioning_feature_names(feature_names: Iterable[str]) -> None:
    names = tuple(feature_names)
    if len(names) != len(set(names)):
        raise InformationBoundaryError("Duplicate conditioning feature names are forbidden.")
    unknown = sorted(set(names) - set(CONDITIONING_FEATURES))
    if unknown:
        raise InformationBoundaryError(f"Unknown conditioning features: {unknown}")
    lowered = " ".join(names).lower()
    blocked = [token for token in FORBIDDEN_FEATURE_TOKENS if token in lowered]
    if blocked:
        raise InformationBoundaryError(f"Forbidden leakage-like feature tokens: {blocked}")


def records_to_conditioning_batch(
    records: list[Mapping[str, Any]],
    *,
    feature_names: tuple[str, ...] = CONDITIONING_FEATURES,
    source_tags: Mapping[str, str] | None = None,
    device: torch.device | str | None = None,
) -> EnergyConditioningBatch:
    """Convert origin-time-visible records into a conditioning tensor.

    Missing allowed features default to 0.0 so callers can share one fixed
    interface across domains; forbidden/unknown feature names are rejected.
    """

    spec = ConditioningSpec(feature_names, source_tags or default_source_tags()).validated()
    rows: list[list[float]] = []
    for record in records:
        rows.append([float(record.get(name, 0.0) or 0.0) for name in spec.feature_names])
    tensor = torch.tensor(rows, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return EnergyConditioningBatch(
        values=tensor,
        feature_names=spec.feature_names,
        source_tags=dict(spec.source_tags or {}),
        origin_time_visible=True,
    ).validate()


class EnergyConditionedAdapter(nn.Module):
    """Residual adapter with condition-dependent gate, scale and shift."""

    def __init__(self, d_model: int, cond_dim: int, bottleneck: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or cond_dim <= 0 or bottleneck <= 0:
            raise ValueError("d_model, cond_dim and bottleneck must be positive.")
        self.d_model = int(d_model)
        self.cond_dim = int(cond_dim)
        self.bottleneck = int(bottleneck)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.gate = nn.Linear(cond_dim, d_model)
        self.scale = nn.Linear(cond_dim, d_model)
        self.shift = nn.Linear(cond_dim, d_model)

    def forward(self, hidden: torch.Tensor, condition: EnergyConditioningBatch | torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be [batch, steps, d_model], got {tuple(hidden.shape)}")
        condition_values = condition.values if isinstance(condition, EnergyConditioningBatch) else condition
        if condition_values.ndim != 2:
            raise ValueError(f"condition must be [batch, cond_dim], got {tuple(condition_values.shape)}")
        if condition_values.shape[0] != hidden.shape[0] or condition_values.shape[1] != self.cond_dim:
            raise ValueError(
                f"condition shape {tuple(condition_values.shape)} incompatible with hidden {tuple(hidden.shape)}"
            )
        residual = self.up(self.dropout(F.gelu(self.down(self.norm(hidden)))))
        gate = torch.sigmoid(self.gate(condition_values)).unsqueeze(1)
        scale = torch.tanh(self.scale(condition_values)).unsqueeze(1)
        shift = self.shift(condition_values).unsqueeze(1)
        return hidden + gate * residual * (1.0 + scale) + shift


class EnergyAdapterForecastShell(nn.Module):
    """Tiny frozen-shell wrapper used for interface tests before backbone wiring."""

    def __init__(self, input_dim: int, d_model: int, cond_dim: int, bottleneck: int, horizon_steps: int) -> None:
        super().__init__()
        self.backbone = nn.Linear(input_dim, d_model)
        self.adapter = EnergyConditionedAdapter(d_model=d_model, cond_dim=cond_dim, bottleneck=bottleneck)
        self.head = nn.Linear(d_model, horizon_steps)
        freeze_module(self.backbone)
        freeze_module(self.head)

    def forward(self, x: torch.Tensor, condition: EnergyConditioningBatch | torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        adapted = self.adapter(hidden, condition)
        pooled = adapted[:, -1, :]
        return self.head(pooled)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().item())
    return total**0.5


def has_any_grad(module: nn.Module) -> bool:
    return any(param.grad is not None for param in module.parameters())


def synthetic_condition_records(batch_size: int) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for idx in range(batch_size):
        records.append(
            {
                "domain_family_load": 1.0 if idx % 2 == 0 else 0.0,
                "domain_family_pv": 1.0 if idx % 2 == 1 else 0.0,
                "horizon_4h": 1.0,
                "horizon_24h": 0.0,
                "native_resolution_minutes": 15.0,
                "context_length_steps": 12.0,
                "horizon_steps": 6.0,
                "context_mean": 0.1 * (idx + 1),
                "context_std": 0.02 * (idx + 1),
                "context_cv": 0.2,
                "context_zero_fraction": 0.0,
                "context_recent_ramp": -0.1 + 0.05 * idx,
                "context_peak_concentration": 0.3 + 0.01 * idx,
                "label_reliability": 0.85,
                "known_future_calendar_phase": idx / max(batch_size - 1, 1),
            }
        )
    return records


def run_synthetic_cuda_interface_check(seed: int = 20260516) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; H1 scaffold interface check is GPU-only.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    batch_size = 4
    context_steps = 12
    input_dim = 5
    d_model = 32
    bottleneck = 8
    horizon_steps = 6

    condition = records_to_conditioning_batch(
        synthetic_condition_records(batch_size),
        device=device,
    )
    alternate_records = synthetic_condition_records(batch_size)
    for record in alternate_records:
        record["domain_family_load"] = 0.0
        record["domain_family_pv"] = 1.0
        record["horizon_4h"] = 0.0
        record["horizon_24h"] = 1.0
        record["label_reliability"] = 0.35
    alternate_condition = records_to_conditioning_batch(alternate_records, device=device)

    model = EnergyAdapterForecastShell(
        input_dim=input_dim,
        d_model=d_model,
        cond_dim=len(CONDITIONING_FEATURES),
        bottleneck=bottleneck,
        horizon_steps=horizon_steps,
    ).to(device)
    model.train()
    x = torch.randn(batch_size, context_steps, input_dim, device=device)
    output = model(x, condition)
    alternate_output = model(x, alternate_condition)
    target = torch.randn_like(output)
    synthetic_loss = F.mse_loss(output, target)
    synthetic_loss.backward()

    adapter_grad = grad_norm(model.adapter)
    condition_delta = float((output.detach() - alternate_output.detach()).abs().mean().item())
    payload = {
        "status": "ok",
        "seed": seed,
        "cuda_required": True,
        "cuda_available": True,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "synthetic_only": True,
        "real_project_data_used": False,
        "train_val_test_data_used": False,
        "validation_or_test_data_used": False,
        "forecast_metrics_computed": False,
        "prediction_artifact_saved": False,
        "optimizer_steps": 0,
        "training_launched": False,
        "fine_tuning_launched": False,
        "inference_on_real_data_launched": False,
        "test_artifacts_created": False,
        "input_shape": list(x.shape),
        "condition_shape": list(condition.values.shape),
        "output_shape": list(output.shape),
        "alternate_output_shape": list(alternate_output.shape),
        "conditioning_feature_names": list(condition.feature_names),
        "conditioning_source_tags": dict(condition.source_tags),
        "allowed_source_tags": list(ALLOWED_SOURCE_TAGS),
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "adapter_interface": {
            "d_model": d_model,
            "cond_dim": len(CONDITIONING_FEATURES),
            "bottleneck": bottleneck,
            "horizon_steps": horizon_steps,
            "residual_adapter": True,
            "condition_gate": True,
            "condition_scale": True,
            "condition_shift": True,
        },
        "synthetic_loss_value_for_gradient_check": float(synthetic_loss.detach().item()),
        "adapter_grad_norm": adapter_grad,
        "adapter_grad_nonzero": adapter_grad > 0.0,
        "condition_delta_mean_abs": condition_delta,
        "condition_delta_nonzero": condition_delta > 0.0,
        "backbone_frozen": all(not param.requires_grad for param in model.backbone.parameters()),
        "head_frozen": all(not param.requires_grad for param in model.head.parameters()),
        "backbone_grad_absent": not has_any_grad(model.backbone),
        "head_grad_absent": not has_any_grad(model.head),
        "adapter_param_count": parameter_count(model.adapter),
        "trainable_param_count": trainable_parameter_count(model),
        "total_param_count": parameter_count(model),
        "only_adapter_trainable": trainable_parameter_count(model) == parameter_count(model.adapter),
    }
    del model, x, output, alternate_output, target, synthetic_loss
    torch.cuda.empty_cache()
    return payload


__all__ = [
    "ALLOWED_SOURCE_TAGS",
    "CONDITIONING_FEATURES",
    "EnergyAdapterForecastShell",
    "EnergyConditionedAdapter",
    "EnergyConditioningBatch",
    "FORBIDDEN_FEATURE_TOKENS",
    "InformationBoundaryError",
    "ConditioningSpec",
    "default_source_tags",
    "records_to_conditioning_batch",
    "run_synthetic_cuda_interface_check",
    "validate_conditioning_feature_names",
]
