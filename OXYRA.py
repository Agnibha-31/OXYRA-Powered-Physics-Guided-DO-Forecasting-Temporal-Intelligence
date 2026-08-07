"""
UNIFIED LSTM WORKBENCH
======================

This is the only Python program required by this repository. It has four modes:

1. ``train``: train a research-aligned public LSTM on sample CSVs or another numeric CSV.
2. ``evaluate``: load a model exported by ``train`` and evaluate/predict on CSV data.
3. ``research-predict``: run the supplied high-precision DO checkpoint on compatible
   raw DO data using its fixed 96-step/150-feature research preprocessing.
4. ``inspect``: safely display the metadata expected from either kind of bundle.

Beginner example (run from the repository root):

    python lstm_workbench.py train \
        --csv data/sample \
        --features dac voltage current rpm temperature do \
        --target clean_do --time timestamp --group family \
        --epochs 30 --output outputs/sample_run

The public trainer works in any regression/forecasting domain when:

* each CSV row is one time step;
* feature and target columns are numeric;
* rows are in time order, or ``--time`` identifies the ordering column; and
* separate sequences are identified with ``--group`` when needed.

Important scientific limits:

* The supplied research checkpoint's R2=0.999347 is a saved held-out score on the
  controlled dissolved-oxygen synthetic/noise-family corpus. It is not a guarantee
  for new sites, sensors, datasets, or unrelated domains.
* The research checkpoint is DO-specific. Use ``train`` for other domains.
* Pickle can execute code while loading. Only load ``.pkl`` files you trust.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


GENERIC_BUNDLE_TYPE = "generic_lstm_regression_bundle"
RESEARCH_BUNDLE_TYPE = "torchscript_lstm_inference_bundle"
RESEARCH_FAMILIES = ["clean", "drift", "impulse", "pli", "quantization", "ripple", "wgn"]
RESEARCH_RAW_COLUMNS = ["dac", "voltage", "current", "rpm", "temperature", "do"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Make repeated CPU runs as reproducible as PyTorch permits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_name(name: str) -> torch.device:
    """Resolve auto/cpu/cuda and give a clear error for an unavailable GPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU. Use --device cpu.")
    return torch.device(name)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_bundle(path: Path) -> dict[str, Any]:
    """Load one trusted bundle and reject unknown payload shapes early."""
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "artifact_type" not in bundle:
        raise ValueError(f"{path} is not a recognized workbench model bundle.")
    return bundle


def standardizer_fit(values: np.ndarray) -> dict[str, list[float]]:
    """Fit a plain-data standardizer; zero-variance columns receive scale 1."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def standardizer_transform(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return ((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32)


def standardizer_inverse(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) * scale + mean).reshape(-1)


def resolve_csv_sources(source: Path) -> list[Path]:
    """Return one or more CSV files from a file path, directory, or glob pattern."""
    if source.is_file():
        return [source]
    if source.is_dir():
        paths = sorted(path for path in source.glob("*.csv") if path.is_file())
        split_sample_names = {
            "synthetic_do_clean.csv",
            "synthetic_do_drift.csv",
            "synthetic_do_impulse.csv",
            "synthetic_do_periodic_interference_pli.csv",
            "synthetic_do_quantization.csv",
            "synthetic_do_ripple.csv",
            "synthetic_do_white_gaussian_noise_wgn.csv",
        }
        if {path.name for path in paths} >= split_sample_names:
            paths = [path for path in paths if path.name in split_sample_names]
    else:
        paths = sorted(path for path in source.parent.glob(source.name) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No CSV files were found for: {source}")
    return paths


def read_csv_collection(source: Path) -> pd.DataFrame:
    """Read one CSV or a folder/glob of CSVs into one ordered frame."""
    frames: list[pd.DataFrame] = []
    for csv_path in resolve_csv_sources(source):
        part = pd.read_csv(csv_path)
        part["__source_file__"] = csv_path.name
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def source_sha256(source: Path) -> dict[str, Any]:
    """Hash the exact CSV input set so a model bundle records its data source."""
    digest = hashlib.sha256()
    paths = resolve_csv_sources(source)
    for csv_path in paths:
        digest.update(csv_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(csv_path.read_bytes())
        digest.update(b"\0")
    return {
        "source_csv_files": [path.name for path in paths],
        "source_csv_sha256": digest.hexdigest(),
    }


def read_numeric_csv(
    source: Path,
    features: Sequence[str],
    target: str | None,
    time_column: str | None,
    group_column: str | None,
) -> pd.DataFrame:
    """Read, validate and deterministically order user CSV data."""
    frame = read_csv_collection(source)
    required = list(features)
    if target:
        required.append(target)
    if time_column:
        required.append(time_column)
    if group_column:
        required.append(group_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    numeric = list(features) + ([target] if target else [])
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    bad = frame[numeric].isna().sum()
    bad = bad[bad > 0]
    if not bad.empty:
        raise ValueError(f"Numeric columns contain missing/non-numeric values: {bad.to_dict()}")
    frame = frame.copy()
    frame["__original_row__"] = np.arange(len(frame), dtype=int)
    order = ([group_column] if group_column else []) + ([time_column] if time_column else ["__original_row__"])
    return frame.sort_values(order, kind="mergesort").reset_index(drop=True)


def group_positions(frame: pd.DataFrame, group_column: str | None) -> Iterable[np.ndarray]:
    if group_column:
        for _, group in frame.groupby(group_column, sort=False, dropna=False):
            yield group.index.to_numpy(dtype=int)
    else:
        yield frame.index.to_numpy(dtype=int)


def split_positions(
    frame: pd.DataFrame,
    group_column: str | None,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[np.ndarray]]:
    """Chronologically split every independent sequence; no random row leakage."""
    result: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    for positions in group_positions(frame, group_column):
        n = len(positions)
        train_end = int(n * train_fraction)
        validation_end = int(n * (train_fraction + validation_fraction))
        result["train"].append(positions[:train_end])
        result["validation"].append(positions[train_end:validation_end])
        result["test"].append(positions[validation_end:])
    return result


def windows_from_segments(
    feature_values: np.ndarray,
    target_values: np.ndarray | None,
    segments: Sequence[np.ndarray],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Turn ordered rows into LSTM histories without crossing group/split boundaries."""
    x_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    row_parts: list[int] = []
    for positions in segments:
        first_target = lookback + horizon - 1
        for local_target in range(first_target, len(positions)):
            end = local_target - horizon + 1
            history_positions = positions[end - lookback : end]
            target_position = int(positions[local_target])
            x_parts.append(feature_values[history_positions])
            if target_values is not None:
                y_parts.append(float(target_values[target_position]))
            row_parts.append(target_position)
    if not x_parts:
        raise ValueError(
            "No windows were created. Each group/split needs at least lookback + horizon rows; "
            "reduce --lookback or provide more data."
        )
    x = np.asarray(x_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.float32) if target_values is not None else None
    return x, y, np.asarray(row_parts, dtype=int)


def windows_for_split_targets(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    full_group_segments: Sequence[np.ndarray],
    target_segments: Sequence[np.ndarray],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create causal histories while keeping every prediction target in its split.

    Validation and test targets may use earlier observations as context, exactly as
    a deployed forecaster would. Targets never move between splits, histories never
    cross a group boundary, and no future row enters an input window.
    """
    if len(full_group_segments) != len(target_segments):
        raise ValueError("Internal group/split mismatch while creating windows.")
    x_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    row_parts: list[int] = []
    for full_positions, targets in zip(full_group_segments, target_segments):
        if len(full_positions) == 0 or len(targets) == 0:
            continue
        group_start = int(full_positions[0])
        group_end = int(full_positions[-1])
        for target_position_value in targets:
            target_position = int(target_position_value)
            history_end = target_position - horizon + 1
            history_start = history_end - lookback
            if history_start < group_start or target_position > group_end:
                continue
            history_positions = np.arange(history_start, history_end, dtype=int)
            x_parts.append(feature_values[history_positions])
            y_parts.append(float(target_values[target_position]))
            row_parts.append(target_position)
    if not x_parts:
        raise ValueError(
            "No causal windows were created. Each group needs enough observations before "
            "the requested split targets; reduce --lookback or provide longer series."
        )
    return (
        np.asarray(x_parts, dtype=np.float32),
        np.asarray(y_parts, dtype=np.float32),
        np.asarray(row_parts, dtype=int),
    )


# ---------------------------------------------------------------------------
# Public LSTM trainers: domain-generic and research-aligned without private IP
# ---------------------------------------------------------------------------


def initialize_lstm(lstm: nn.LSTM) -> None:
    """Research-style Xavier/orthogonal initialization with positive forget bias."""
    for name, parameter in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(parameter)
        elif "weight_hh" in name:
            for gate in parameter.chunk(4, dim=0):
                nn.init.orthogonal_(gate)
        elif "bias" in name:
            nn.init.zeros_(parameter)
            hidden = parameter.shape[0] // 4
            parameter.data[hidden : 2 * hidden].fill_(1.0)


class TrainingGaussianNoise(nn.Module):
    """Small input perturbation used only during training for robustness."""

    def __init__(self, standard_deviation: float) -> None:
        super().__init__()
        self.standard_deviation = float(standard_deviation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.training and self.standard_deviation > 0.0:
            return values + torch.randn_like(values) * self.standard_deviation
        return values


class CausalConv1D(nn.Module):
    """A temporal convolution padded only on the past side."""

    def __init__(self, input_width: int, output_width: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = int(dilation * (kernel_size - 1))
        self.convolution = nn.Conv1d(input_width, output_width, kernel_size, dilation=dilation)
        nn.init.xavier_uniform_(self.convolution.weight)
        nn.init.zeros_(self.convolution.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        channels_first = sequence.transpose(1, 2)
        channels_first = F.pad(channels_first, (self.left_padding, 0))
        return F.silu(self.convolution(channels_first).transpose(1, 2))


class CompactLSTM(nn.Module):
    """Legacy compact profile retained for small CPUs and older exported bundles."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        initialize_lstm(self.lstm)
        width = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, max(width // 2, 8)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(width // 2, 8), 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(history)
        return self.head(sequence[:, -1, :])


class ResearchAlignedLSTM(nn.Module):
    """Public backbone aligned with the safe, high-level design of the DO model.

    Preserved here: multiscale causal filtering, a 48/48/32 recurrent hierarchy,
    residual dilated temporal refinement, dual attention summaries, and the main
    regularization widths. Intentionally excluded: the private 150-feature recipe,
    disturbance-family gates/experts, auxiliary delta heads, calibration rules, and
    synthetic-data generator. Consequently this is close to the research backbone,
    but it is not a from-scratch reproduction of the protected manuscript pipeline.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_noise = TrainingGaussianNoise(0.012)
        self.local_filter = CausalConv1D(input_size, 32, kernel_size=3)
        self.medium_filter = CausalConv1D(input_size, 16, kernel_size=5, dilation=2)
        self.long_filter = CausalConv1D(input_size, 16, kernel_size=7, dilation=4)
        self.filter_norm = nn.LayerNorm(64)

        self.context_lstm = nn.LSTM(64, 48, batch_first=True, bidirectional=True)
        self.context_norm = nn.LayerNorm(96)
        self.dynamics_lstm = nn.LSTM(96, 48, batch_first=True)
        self.dynamics_norm = nn.LayerNorm(48)
        self.trend_lstm = nn.LSTM(48, 32, batch_first=True)
        self.trend_norm = nn.LayerNorm(32)
        for recurrent_layer in (self.context_lstm, self.dynamics_lstm, self.trend_lstm):
            initialize_lstm(recurrent_layer)

        self.temporal_refiners = nn.ModuleList(
            [CausalConv1D(32, 32, kernel_size=3, dilation=dilation) for dilation in (1, 2, 4)]
        )
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(32) for _ in range(3)])
        self.trend_attention = nn.Linear(32, 1)
        self.transient_attention = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.trend_attention.weight)
        nn.init.xavier_uniform_(self.transient_attention.weight)
        nn.init.zeros_(self.trend_attention.bias)
        nn.init.zeros_(self.transient_attention.bias)

        self.regression_head = nn.Sequential(
            nn.Linear(96, 80),
            nn.SiLU(),
            nn.Dropout(0.14),
            nn.Linear(80, 48),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 1),
        )
        for layer in self.regression_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        filtered_input = self.input_noise(history)
        multiscale = torch.cat(
            [
                self.local_filter(filtered_input),
                self.medium_filter(filtered_input),
                self.long_filter(filtered_input),
            ],
            dim=-1,
        )
        multiscale = self.filter_norm(multiscale)
        multiscale = F.dropout1d(multiscale.transpose(1, 2), p=0.08, training=self.training).transpose(1, 2)

        context, _ = self.context_lstm(F.dropout(multiscale, p=0.10, training=self.training))
        context = self.context_norm(context)
        dynamics, _ = self.dynamics_lstm(F.dropout(context, p=0.12, training=self.training))
        dynamics = self.dynamics_norm(dynamics)
        trend, _ = self.trend_lstm(F.dropout(dynamics, p=0.10, training=self.training))
        trend = self.trend_norm(trend)

        refined = trend
        for convolution, normalization in zip(self.temporal_refiners, self.temporal_norms):
            residual = normalization(convolution(refined))
            residual = F.dropout1d(residual.transpose(1, 2), p=0.045, training=self.training).transpose(1, 2)
            refined = refined + residual

        trend_weights = torch.softmax(torch.tanh(self.trend_attention(trend)), dim=1)
        transient_weights = torch.softmax(torch.tanh(self.transient_attention(refined)), dim=1)
        trend_summary = torch.sum(trend_weights * trend, dim=1)
        transient_summary = torch.sum(transient_weights * refined, dim=1)
        final_state = refined[:, -1, :]
        return self.regression_head(torch.cat([trend_summary, transient_summary, final_state], dim=-1))


def build_generic_model(config: dict[str, Any]) -> nn.Module:
    profile = str(config.get("architecture_profile", "compact"))
    if profile == "research-aligned":
        return ResearchAlignedLSTM(input_size=int(config["input_size"]))
    if profile == "compact":
        return CompactLSTM(
            input_size=int(config["input_size"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
            bidirectional=bool(config["bidirectional"]),
        )
    raise ValueError(f"Unsupported architecture profile: {profile}")


def predict_generic(
    model: nn.Module,
    x: np.ndarray,
    target_scaler: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            parts.append(model(xb).detach().cpu().numpy().reshape(-1))
    return standardizer_inverse(np.concatenate(parts), target_scaler)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Domain-independent accuracy and error metrics, all in original target units."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Metrics need equally sized, non-empty true and predicted arrays.")
    residual = y_pred - y_true
    abs_error = np.abs(residual)
    squared_error = residual**2
    target_variance = float(np.var(y_true))
    target_range = float(np.max(y_true) - np.min(y_true))
    target_std = float(np.std(y_true))
    r2 = 1.0 - float(squared_error.sum()) / float(((y_true - y_true.mean()) ** 2).sum()) if target_variance > 0 else float("nan")
    explained = 1.0 - float(np.var(y_true - y_pred)) / target_variance if target_variance > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if target_std > 0 and np.std(y_pred) > 0 else float("nan")
    epsilon = max(float(np.mean(np.abs(y_true))) * 1e-8, 1e-8)
    mape = float(np.mean(abs_error / np.maximum(np.abs(y_true), epsilon)) * 100.0)
    smape = float(np.mean(2.0 * abs_error / np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)) * 100.0)
    rmse = float(np.sqrt(np.mean(squared_error)))
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(abs_error)),
        "mse": float(np.mean(squared_error)),
        "rmse": rmse,
        "r2": r2,
        "explained_variance": explained,
        "median_absolute_error": float(np.median(abs_error)),
        "max_absolute_error": float(np.max(abs_error)),
        "bias_pred_minus_true": float(np.mean(residual)),
        "pearson_correlation": pearson,
        "mape_percent": mape,
        "smape_percent": smape,
        "nrmse_by_range": rmse / target_range if target_range > 0 else float("nan"),
        "nrmse_by_std": rmse / target_std if target_std > 0 else float("nan"),
    }


def finite_json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """JSON has no portable NaN; represent undefined metrics as null."""
    return {key: (None if isinstance(value, float) and not math.isfinite(value) else value) for key, value in metrics.items()}


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    clean = finite_json_metrics(metrics)
    save_json(output_dir / "metrics.json", clean)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean))
        writer.writeheader()
        writer.writerow(clean)


def save_six_plots(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: Sequence[dict[str, float]] | None = None,
    max_overlay_points: int = 1200,
) -> None:
    """Create six generic diagnostic plots that remain meaningful across domains."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_pred - y_true
    absolute_error = np.abs(residual)

    # 1) Training and validation loss. If unavailable, show the available evaluation error.
    fig, ax = plt.subplots(figsize=(8, 5))
    if history:
        ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
        ax.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation")
        ax.set_ylabel("scaled Huber loss")
        ax.set_xlabel("epoch")
        ax.legend()
    else:
        ax.plot(np.sort(absolute_error))
        ax.set_ylabel("absolute error")
        ax.set_xlabel("samples sorted by error")
    ax.set_title("Training history / evaluation error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_training_or_error_curve.png", dpi=170)
    plt.close(fig)

    # 2) Ordered overlay reveals drift, lag and local failures.
    count = min(len(y_true), max_overlay_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true[:count], label="actual", linewidth=1.5)
    ax.plot(y_pred[:count], label="predicted", linewidth=1.1, alpha=0.85)
    ax.set_title(f"Actual vs predicted (first {count:,} evaluated samples)")
    ax.set_xlabel("ordered evaluation sample")
    ax.set_ylabel("target")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_actual_vs_predicted.png", dpi=170)
    plt.close(fig)

    # 3) Scatter should cluster around the 1:1 reference line.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.35)
    low = min(float(y_true.min()), float(y_pred.min()))
    high = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([low, high], [low, high], "k--", linewidth=1.2, label="ideal")
    ax.set_title("Predicted vs actual")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_predicted_vs_actual_scatter.png", dpi=170)
    plt.close(fig)

    # 4) Residual structure indicates non-linearity, bias or unequal variance.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residual, s=10, alpha=0.35)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals vs predicted")
    ax.set_xlabel("predicted")
    ax.set_ylabel("prediction - actual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_residuals_vs_predicted.png", dpi=170)
    plt.close(fig)

    # 5) The residual distribution makes bias and heavy tails visible.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=min(50, max(10, int(np.sqrt(len(residual))))), alpha=0.8, edgecolor="white")
    ax.axvline(float(residual.mean()), color="red", linestyle="--", label=f"mean={residual.mean():.4g}")
    ax.set_title("Residual distribution")
    ax.set_xlabel("prediction - actual")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_residual_histogram.png", dpi=170)
    plt.close(fig)

    # 6) Error by target quantile shows where the response range is hardest.
    unique_count = len(np.unique(y_true))
    bins = min(10, unique_count)
    fig, ax = plt.subplots(figsize=(9, 5))
    if bins >= 2:
        categories = pd.qcut(y_true, q=bins, duplicates="drop")
        table = pd.DataFrame({"bin": categories, "absolute_error": absolute_error}).groupby("bin", observed=True)["absolute_error"].mean()
        ax.bar(np.arange(len(table)), table.to_numpy())
        ax.set_xticks(np.arange(len(table)))
        ax.set_xticklabels([str(item) for item in table.index], rotation=35, ha="right", fontsize=8)
    else:
        ax.bar([0], [float(absolute_error.mean())])
        ax.set_xticks([0])
        ax.set_xticklabels(["single target value"])
    ax.set_title("Mean absolute error by actual-target quantile")
    ax.set_xlabel("actual-target range")
    ax.set_ylabel("mean absolute error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "06_error_by_target_quantile.png", dpi=170)
    plt.close(fig)


def make_prediction_table(
    frame: pd.DataFrame,
    row_positions: np.ndarray,
    y_pred: np.ndarray,
    target: str | None,
    time_column: str | None,
    group_column: str | None,
) -> pd.DataFrame:
    result = pd.DataFrame({"source_row": frame.iloc[row_positions]["__original_row__"].to_numpy(), "prediction": y_pred})
    if target:
        actual = frame.iloc[row_positions][target].to_numpy(dtype=float)
        result.insert(1, "actual", actual)
        result["error_pred_minus_actual"] = result["prediction"] - result["actual"]
        result["absolute_error"] = result["error_pred_minus_actual"].abs()
    if time_column:
        result.insert(1, time_column, frame.iloc[row_positions][time_column].to_numpy())
    if group_column:
        result.insert(1, group_column, frame.iloc[row_positions][group_column].to_numpy())
    return result


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = device_from_name(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_numeric_csv(Path(args.csv), args.features, args.target, args.time, args.group)
    splits = split_positions(frame, args.group, args.train_fraction, args.validation_fraction)
    full_groups = list(group_positions(frame, args.group))

    # Fit scalers on training rows only. This avoids letting validation/test values
    # influence preprocessing, a common and serious time-series leakage error.
    training_rows = np.concatenate(splits["train"])
    raw_features = frame[list(args.features)].to_numpy(dtype=np.float64)
    raw_target = frame[args.target].to_numpy(dtype=np.float64)
    feature_scaler = standardizer_fit(raw_features[training_rows])
    target_scaler = standardizer_fit(raw_target[training_rows].reshape(-1, 1))
    scaled_features = standardizer_transform(raw_features, feature_scaler)
    scaled_target = standardizer_transform(raw_target.reshape(-1, 1), target_scaler).reshape(-1)

    x_train, y_train, _ = windows_for_split_targets(
        scaled_features, scaled_target, full_groups, splits["train"], args.lookback, args.horizon
    )
    x_validation, y_validation, _ = windows_for_split_targets(
        scaled_features, scaled_target, full_groups, splits["validation"], args.lookback, args.horizon
    )
    x_test, y_test_scaled, test_rows = windows_for_split_targets(
        scaled_features, scaled_target, full_groups, splits["test"], args.lookback, args.horizon
    )

    config = {
        "features": list(args.features),
        "target": args.target,
        "time": args.time,
        "group": args.group,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "input_size": len(args.features),
        "architecture_profile": args.profile,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "seed": args.seed,
        "research_alignment": {
            "safe_public_backbone": args.profile == "research-aligned",
            "conv_filters": [32, 16, 16],
            "conv_kernels": [3, 5, 7],
            "conv_dilations": [1, 2, 4],
            "recurrent_hidden_sizes": [48, 48, 32],
            "recurrent_directions": ["bidirectional", "forward", "forward"],
            "tcn_dilations": [1, 2, 4],
            "dense_hidden_sizes": [80, 48],
            "input_noise_std": 0.012,
            "not_recreated_by_public_training_profile": [
                "full_original_150_feature_training_pipeline",
                "disturbance_family_specialist_architecture",
                "multi_objective_delta_and_sequence_training",
                "full_research_calibration_and_selection_orchestration",
                "synthetic_dataset_generator",
                "full_research_corpus",
            ],
        },
    }
    model = build_generic_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.55,
        patience=args.reduce_lr_patience,
        min_lr=args.min_learning_rate,
    )
    loss_function = nn.HuberLoss(delta=args.huber_delta)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.reshape(-1, 1))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_x = torch.from_numpy(x_validation).to(device)
    validation_y = torch.from_numpy(y_validation.reshape(-1, 1)).to(device)
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    patience_left = args.patience
    history: list[dict[str, float]] = []

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training profile={args.profile}, lookback={args.lookback}, parameters={parameter_count:,} on {device}; "
        f"train={len(x_train):,}, validation={len(x_validation):,}, test={len(x_test):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_function(prediction, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).cpu())
        train_loss = total_loss / max(seen, 1)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
            }
        )
        print(
            f"epoch {epoch:03d} | train={train_loss:.6f} | validation={validation_loss:.6f} "
            f"| lr={current_learning_rate:.8f}"
        )
        scheduler.step(validation_loss)
        if validation_loss < best_validation - args.min_delta:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; restoring the best validation checkpoint.")
                break
    model.load_state_dict(best_state)

    y_test = standardizer_inverse(y_test_scaled, target_scaler)
    y_pred = predict_generic(model, x_test, target_scaler, args.batch_size, device)
    metrics = regression_metrics(y_test, y_pred)
    predictions = make_prediction_table(frame, test_rows, y_pred, args.target, args.time, args.group)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    save_metrics(output_dir, metrics)
    save_six_plots(output_dir, y_test, y_pred, history)

    bundle = {
        "format_version": 1,
        "artifact_type": GENERIC_BUNDLE_TYPE,
        "framework": "pytorch",
        "model_config": config,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "training_history": history,
        "training_config": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "huber_delta": args.huber_delta,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": args.patience,
            "reduce_lr_patience": args.reduce_lr_patience,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
        },
        "test_metrics": finite_json_metrics(metrics),
        **source_sha256(Path(args.csv)),
        "notes": [
            "Metrics are specific to this chronological test split.",
            "Retrain and validate before using the model on another data distribution.",
            "The research-aligned public profile preserves the safe backbone, not the protected full research recipe.",
        ],
    }
    model_path = output_dir / "trained_lstm_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model: {model_path.resolve()}")
    print(json.dumps(finite_json_metrics(metrics), indent=2))


def evaluate_command(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model))
    if bundle["artifact_type"] != GENERIC_BUNDLE_TYPE:
        raise ValueError("Use research-predict for the supplied high-precision DO artifact.")
    config = bundle["model_config"]
    features = config["features"]
    target = None if args.prediction_only else (args.target if args.target is not None else config.get("target"))
    time_column = args.time if args.time is not None else config.get("time")
    group_column = args.group if args.group is not None else config.get("group")
    frame = read_numeric_csv(Path(args.csv), features, target, time_column, group_column)
    scaled_features = standardizer_transform(frame[features].to_numpy(dtype=float), bundle["feature_scaler"])
    target_values = frame[target].to_numpy(dtype=float) if target else None
    segments = list(group_positions(frame, group_column))
    x, y, rows = windows_from_segments(
        scaled_features,
        target_values,
        segments,
        int(config["lookback"]),
        int(config["horizon"]),
    )
    device = device_from_name(args.device)
    model = build_generic_model(config).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    y_pred = predict_generic(model, x, bundle["target_scaler"], args.batch_size, device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = make_prediction_table(frame, rows, y_pred, target, time_column, group_column)
    table.to_csv(output_dir / "predictions.csv", index=False)
    if y is not None:
        metrics = regression_metrics(y, y_pred)
        save_metrics(output_dir, metrics)
        save_six_plots(output_dir, y, y_pred, bundle.get("training_history"))
        print(json.dumps(finite_json_metrics(metrics), indent=2))
    else:
        print("Predictions were saved. Metrics/plots need a known --target column.")
    print(f"Outputs: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# High-precision DO checkpoint: fixed research schema and inference only
# ---------------------------------------------------------------------------


def endpoint_slope(values: np.ndarray) -> float:
    return float((values[-1] - values[0]) / max(len(values) - 1, 1)) if len(values) > 1 else 0.0


def research_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the causal feature logic expected by the supplied research model."""
    base = frame.copy()
    extra: dict[str, pd.Series | np.ndarray] = {}
    max_time = max(float(base["timestamp"].max()), 1.0)
    extra["time_norm"] = base["timestamp"] / max_time
    for period in (100.0, 500.0, 1000.0):
        angle = 2.0 * math.pi * base["timestamp"] / period
        extra[f"time_sin_{int(period)}s"] = np.sin(angle)
        extra[f"time_cos_{int(period)}s"] = np.cos(angle)
    for column in RESEARCH_RAW_COLUMNS:
        series = base[column]
        difference = series.diff().fillna(0.0)
        extra[f"{column}_diff1"] = difference
        extra[f"{column}_abs_diff1"] = difference.abs()
        extra[f"{column}_sign_diff1"] = np.sign(difference)
        for lag in (2, 3, 5, 9):
            extra[f"{column}_diff{lag}_avg"] = series.diff(lag).fillna(0.0) / float(lag)
        median5 = series.rolling(5, min_periods=1).median()
        std5 = series.rolling(5, min_periods=2).std().fillna(0.0)
        median21 = series.rolling(21, min_periods=1).median()
        std21 = series.rolling(21, min_periods=2).std().fillna(0.0)
        extra[f"{column}_roll_mean_5"] = series.rolling(5, min_periods=1).mean()
        extra[f"{column}_roll_std_5"] = std5
        extra[f"{column}_roll_median_5"] = median5
        extra[f"{column}_median_resid_5"] = series - median5
        extra[f"{column}_roll_slope_5"] = series.rolling(5, min_periods=2).apply(endpoint_slope, raw=True).fillna(0.0)
        extra[f"{column}_roll_mean_21"] = series.rolling(21, min_periods=1).mean()
        extra[f"{column}_roll_std_21"] = std21
        extra[f"{column}_roll_median_21"] = median21
        extra[f"{column}_median_resid_21"] = series - median21
        extra[f"{column}_roll_slope_13"] = series.rolling(13, min_periods=2).apply(endpoint_slope, raw=True).fillna(0.0)
        extra[f"{column}_roll_slope_21"] = series.rolling(21, min_periods=2).apply(endpoint_slope, raw=True).fillna(0.0)
    do_diff = pd.Series(extra["do_diff1"], index=base.index)
    do_abs = pd.Series(extra["do_abs_diff1"], index=base.index)
    flat5 = (do_abs < 1e-4).rolling(5, min_periods=1).mean()
    flat13 = (do_abs < 1e-4).rolling(13, min_periods=1).mean()
    flat21 = (do_abs < 1e-4).rolling(21, min_periods=1).mean()
    quant005 = (do_abs >= 0.045).astype(float)
    extra["do_diff2"] = do_diff.diff().fillna(0.0)
    extra["do_abs_activity_5"] = do_abs.rolling(5, min_periods=1).mean()
    extra["do_abs_activity_21"] = do_abs.rolling(21, min_periods=1).mean()
    extra["do_impulse_score_5"] = pd.Series(extra["do_median_resid_5"], index=base.index).abs() / (pd.Series(extra["do_roll_std_5"], index=base.index) + 1e-4)
    extra["do_impulse_score_21"] = pd.Series(extra["do_median_resid_21"], index=base.index).abs() / (pd.Series(extra["do_roll_std_21"], index=base.index) + 1e-4)
    extra["do_flat_ratio_5"] = flat5
    extra["do_flat_ratio_13"] = flat13
    extra["do_flat_ratio_21"] = flat21
    extra["do_quant_step_flag_005"] = quant005
    extra["do_quant_step_flag_009"] = (do_abs >= 0.09).astype(float)
    extra["do_step_interval_proxy"] = quant005.rolling(21, min_periods=1).mean()
    extra["do_substep_trend_5"] = pd.Series(extra["do_roll_slope_5"], index=base.index) * flat5
    extra["do_substep_trend_21"] = pd.Series(extra["do_roll_slope_21"], index=base.index) * flat21
    extra["power_proxy"] = base["voltage"] * base["current"]
    extra["drive_proxy"] = base["dac"] * base["rpm"]
    extra["sensor_load_ratio"] = base["current"] / (base["dac"].abs() + 1e-6)
    return pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)


def table_from_plain(value: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(value.get("records", []), columns=value.get("columns", []))


def inverse_research_delta(values: np.ndarray, families: Sequence[str], transform: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if transform["mode"] == "global":
        parameters = transform["global"]
        return values * float(parameters["scale"]) + float(parameters["center"])
    result = np.empty_like(values)
    family_array = np.asarray(families)
    for family in np.unique(family_array):
        parameters = transform["families"][str(family)]
        mask = family_array == family
        result[mask] = values[mask] * float(parameters["scale"]) + float(parameters["center"])
    return result


def apply_research_calibration(
    delta: np.ndarray,
    families: Sequence[str],
    set_ids: Sequence[str],
    file_ids: Sequence[str],
    calibration: dict[str, Any],
) -> np.ndarray:
    """Apply the saved calibration stages in their original inference order."""
    result = np.asarray(delta, dtype=np.float32).copy()
    family_array = np.asarray(families, dtype=str)
    set_array = np.asarray(set_ids, dtype=str)
    file_array = np.asarray(file_ids, dtype=str)
    stages = [
        ("family_delta_calibration", ("family",)),
        ("context_delta_calibration", ("family", "set_id")),
    ]
    for name, keys in stages:
        table = table_from_plain(calibration.get(name, {}))
        for _, row in table.iterrows():
            if "enabled" in row and not bool(row["enabled"]):
                continue
            mask = family_array == str(row["family"])
            if "set_id" in keys:
                mask &= set_array == str(row["set_id"])
            if mask.any():
                result[mask] = float(row["intercept"]) + float(row["slope"]) * result[mask]
    activity = table_from_plain(calibration.get("activity_delta_calibration", {}))
    for _, row in activity.iterrows():
        if not bool(row.get("enabled", False)):
            continue
        mask = (family_array == str(row["family"])) & (set_array == str(row["set_id"]))
        if mask.any():
            shifted = result[mask] + float(row["offset"])
            result[mask] = float(row["center"]) + float(row["gain"]) * (shifted - float(row["center"]))

    # Temporal calibration uses only past/current prediction changes within each
    # physical series, preserving causal inference and preventing cross-series mixing.
    innovation = np.zeros_like(result)
    for key in dict.fromkeys(zip(family_array, set_array, file_array)):
        positions = np.flatnonzero(
            (family_array == key[0]) & (set_array == key[1]) & (file_array == key[2])
        )
        if len(positions) > 1:
            innovation[positions[1:]] = result[positions[1:]] - result[positions[:-1]]
    temporal = table_from_plain(calibration.get("temporal_delta_calibration", {}))
    for _, row in temporal.iterrows():
        if not bool(row.get("enabled", False)):
            continue
        mask = (family_array == str(row["family"])) & (set_array == str(row["set_id"]))
        result[mask] = result[mask] + float(row["lead_gain"]) * innovation[mask]

    file_temporal = table_from_plain(calibration.get("file_temporal_delta_calibration", {}))
    for _, row in file_temporal.iterrows():
        if not bool(row.get("enabled", False)):
            continue
        mask = (
            (family_array == str(row["family"]))
            & (set_array == str(row["set_id"]))
            & (file_array == str(row["file_id"]))
        )
        result[mask] = result[mask] + float(row["lead_gain"]) * innovation[mask]

    file_activity = table_from_plain(calibration.get("file_activity_delta_calibration", {}))
    for _, row in file_activity.iterrows():
        if not bool(row.get("enabled", False)):
            continue
        family = str(row["family"])
        gain = float(row["gain"])
        if family == "drift":
            gain = min(gain, 1.028)
        mask = (
            (family_array == family)
            & (set_array == str(row["set_id"]))
            & (file_array == str(row["file_id"]))
        )
        center = float(row["center"])
        result[mask] = center + gain * (result[mask] - center)
    return result


def research_predict_command(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model))
    if bundle["artifact_type"] != RESEARCH_BUNDLE_TYPE:
        raise ValueError("research-predict requires the supplied high-precision DO bundle.")
    frame = read_csv_collection(Path(args.csv))
    required = ["timestamp", *RESEARCH_RAW_COLUMNS]
    if args.family_column:
        required.append(args.family_column)
    if args.target:
        required.append(args.target)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Research CSV is missing columns: {missing}")
    for column in required:
        if column != args.family_column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[[c for c in required if c != args.family_column]].isna().any().any():
        raise ValueError("Research input contains missing/non-numeric required values.")
    if args.family_column:
        frame["__family__"] = frame[args.family_column].astype(str).str.lower()
    else:
        frame["__family__"] = args.family
    unknown = sorted(set(frame["__family__"]) - set(RESEARCH_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown research noise families: {unknown}; choose from {RESEARCH_FAMILIES}")
    frame["__original_row__"] = np.arange(len(frame), dtype=int)
    group_columns = ["__family__"] + ([args.series_column] if args.series_column else [])
    frame = frame.sort_values(group_columns + ["timestamp"], kind="mergesort").reset_index(drop=True)

    preprocessing = bundle["preprocessing"]
    feature_columns = preprocessing["feature_columns"]
    continuous = preprocessing["continuous_columns"]
    all_x: list[np.ndarray] = []
    current_do: list[float] = []
    target_values: list[float] = []
    output_rows: list[int] = []
    output_families: list[str] = []
    output_sets: list[str] = []
    output_files: list[str] = []
    lookback = int(bundle["model_init"]["lookback"])

    for series_number, (_, group) in enumerate(frame.groupby(group_columns, sort=False, dropna=False)):
        group = group.copy().reset_index(drop=True)
        engineered = research_rolling_features(group)
        family = str(group["__family__"].iloc[0])
        file_id = str(group[args.series_column].iloc[0]) if args.series_column else f"public_series_{series_number}"
        for known_family in RESEARCH_FAMILIES:
            engineered[f"family_{known_family}"] = 1.0 if family == known_family else 0.0
        for set_id in ("set6", "set7"):
            engineered[f"set_{set_id}"] = 1.0 if args.set_id == set_id else 0.0
        for variant in ("v1", "v2", "v3", "v4"):
            engineered[f"variant_{variant}"] = 1.0 if args.variant == variant else 0.0
        missing_features = sorted(set(feature_columns) - set(engineered.columns))
        if missing_features:
            raise ValueError(f"Could not build research features: {missing_features}")
        values = engineered[feature_columns].to_numpy(dtype=np.float64)
        continuous_indices = [feature_columns.index(column) for column in continuous]
        scaler = preprocessing["feature_scaler"]
        values[:, continuous_indices] = standardizer_transform(values[:, continuous_indices], scaler)
        values = values.astype(np.float32)
        for target_index in range(lookback, len(group)):
            all_x.append(values[target_index - lookback : target_index])
            current_do.append(float(group["do"].iloc[target_index - 1]))
            output_rows.append(int(group["__original_row__"].iloc[target_index]))
            output_families.append(family)
            output_sets.append(args.set_id)
            output_files.append(file_id)
            if args.target:
                target_values.append(float(group[args.target].iloc[target_index]))
    if not all_x:
        raise ValueError(f"Each research series needs more than {lookback} rows.")

    x = np.asarray(all_x, dtype=np.float32)
    family_gate = np.zeros((len(x), len(RESEARCH_FAMILIES)), dtype=np.float32)
    for row, family in enumerate(output_families):
        family_gate[row, RESEARCH_FAMILIES.index(family)] = 1.0
    do_scaler = preprocessing["do_scaler"]
    current_array = np.asarray(current_do, dtype=np.float64)
    current_scaled = standardizer_transform(current_array.reshape(-1, 1), do_scaler)
    model = torch.jit.load(io.BytesIO(bundle["torchscript_bytes"]), map_location=args.device)
    model.eval()
    delta_scaled_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            end = start + args.batch_size
            _, delta_scaled, _ = model(
                torch.from_numpy(x[start:end]).to(args.device),
                torch.from_numpy(family_gate[start:end]).to(args.device),
                torch.from_numpy(current_scaled[start:end]).to(args.device),
            )
            delta_scaled_parts.append(delta_scaled.cpu().numpy().reshape(-1))
    delta = inverse_research_delta(np.concatenate(delta_scaled_parts), output_families, preprocessing["delta_transform"])
    delta = apply_research_calibration(
        delta,
        output_families,
        output_sets,
        output_files,
        bundle["calibration"],
    )
    prediction = current_array + delta
    clip = bundle["calibration"].get("clip_bounds")
    if clip:
        prediction = np.clip(prediction, float(clip[0]), float(clip[1]))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        {
            "source_row": output_rows,
            "family": output_families,
            "series_id": output_files,
            "current_do": current_array,
            "predicted_next_do": prediction,
        }
    )
    if args.target:
        actual = np.asarray(target_values, dtype=float)
        result["actual_next_do"] = actual
        result["error_pred_minus_actual"] = prediction - actual
        result["absolute_error"] = np.abs(prediction - actual)
        metrics = regression_metrics(actual, prediction)
        save_metrics(output_dir, metrics)
        save_six_plots(output_dir, actual, prediction, bundle.get("training_history"))
        print(json.dumps(finite_json_metrics(metrics), indent=2))
    result.to_csv(output_dir / "research_predictions.csv", index=False)
    print(f"Saved research predictions to: {output_dir.resolve()}")
    print("Remember: the published 0.999347 R2 belongs only to the original held-out research split.")


def inspect_command(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model))
    if bundle["artifact_type"] == RESEARCH_BUNDLE_TYPE:
        summary = {
            "artifact_type": bundle["artifact_type"],
            "model_name": bundle.get("model_name"),
            "model_init": bundle.get("model_init"),
            "input_spec": bundle.get("input_spec"),
            "reported_test_metrics": bundle.get("reported_test_metrics"),
            "limitations": bundle.get("limitations"),
            "provenance": bundle.get("provenance"),
        }
    else:
        summary = {
            "artifact_type": bundle["artifact_type"],
            "model_config": bundle.get("model_config"),
            "test_metrics": bundle.get("test_metrics"),
            "notes": bundle.get("notes"),
        }
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, export, evaluate and plot LSTM time-series models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a public research-aligned LSTM and export a .pkl bundle.")
    train.add_argument("--csv", required=True, help="Input CSV path, directory of CSVs, or glob pattern.")
    train.add_argument("--features", nargs="+", required=True, help="Numeric input feature column names.")
    train.add_argument("--target", required=True, help="Numeric value to predict.")
    train.add_argument("--time", default=None, help="Optional time/order column.")
    train.add_argument("--group", default=None, help="Optional independent-series/group column.")
    train.add_argument(
        "--profile",
        choices=["research-aligned", "compact"],
        default="research-aligned",
        help="research-aligned follows the safe public backbone; compact is a faster CPU baseline.",
    )
    train.add_argument(
        "--lookback",
        type=int,
        default=96,
        help="Past rows per sample. The 96-step default matches the research model; adjust for another sampling rate/domain.",
    )
    train.add_argument("--horizon", type=int, default=1, help="Steps ahead to predict.")
    train.add_argument("--train-fraction", type=float, default=0.70)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--hidden-size", type=int, default=64, help="Compact profile only.")
    train.add_argument("--num-layers", type=int, default=2, help="Compact profile only.")
    train.add_argument("--dropout", type=float, default=0.15, help="Compact profile only.")
    train.add_argument("--bidirectional", action="store_true", help="Compact profile only.")
    train.add_argument("--epochs", type=int, default=70)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=8e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--huber-delta", type=float, default=0.75, help="Robust scaled-target Huber threshold.")
    train.add_argument("--min-learning-rate", type=float, default=1e-6)
    train.add_argument("--reduce-lr-patience", type=int, default=4)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--output", default="outputs/training_run")
    train.set_defaults(function=train_command)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate/predict with a public trainer-exported model.")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--csv", required=True, help="Input CSV path, directory of CSVs, or glob pattern.")
    evaluate.add_argument("--target", default=None, help="Override the target column stored in the model bundle.")
    evaluate.add_argument("--prediction-only", action="store_true", help="Do not require a target; save predictions only.")
    evaluate.add_argument("--time", default=None)
    evaluate.add_argument("--group", default=None)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument("--output", default="outputs/evaluation")
    evaluate.set_defaults(function=evaluate_command)

    research = subparsers.add_parser("research-predict", help="Use the supplied fixed high-precision DO checkpoint.")
    research.add_argument("--model", default="artifacts/high_precision_do_lstm_r2_0_999347.pkl")
    research.add_argument("--csv", required=True, help="Compatible DO CSV path, directory of CSVs, or glob pattern.")
    research.add_argument("--target", default=None, help="Optional known next-step target, e.g. clean_do.")
    research.add_argument("--family-column", default="family")
    research.add_argument("--family", choices=RESEARCH_FAMILIES, default="clean", help="Used if --family-column is empty.")
    research.add_argument("--series-column", default=None, help="Optional ID for separate physical time series.")
    research.add_argument("--set-id", choices=["set6", "set7"], default="set6")
    research.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], default="v1")
    research.add_argument("--batch-size", type=int, default=256)
    research.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    research.add_argument("--output", default="outputs/research_inference")
    research.set_defaults(function=research_predict_command)

    inspect = subparsers.add_parser("inspect", help="Print model metadata without training or prediction.")
    inspect.add_argument("--model", required=True)
    inspect.set_defaults(function=inspect_command)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "train":
        if not 0.0 < args.train_fraction < 1.0:
            raise ValueError("--train-fraction must be between 0 and 1.")
        if not 0.0 < args.validation_fraction < 1.0:
            raise ValueError("--validation-fraction must be between 0 and 1.")
        if args.train_fraction + args.validation_fraction >= 1.0:
            raise ValueError("Training + validation fractions must leave a non-empty test fraction.")
        for name in (
            "lookback",
            "horizon",
            "epochs",
            "batch_size",
            "hidden_size",
            "num_layers",
            "patience",
            "reduce_lr_patience",
        ):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        if args.huber_delta <= 0 or args.min_learning_rate <= 0:
            raise ValueError("--huber-delta and --min-learning-rate must be positive.")
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError("--dropout must be in [0, 1).")
        if args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0:
            raise ValueError("Learning rate must be positive; weight decay and min delta cannot be negative.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.function(args)


if __name__ == "__main__":
    main()
