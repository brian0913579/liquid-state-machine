from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import simple_lsm as sl


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "presentation_assets"
ASSET_DIR.mkdir(exist_ok=True)


def fig_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return Image.fromarray(arr.astype(np.uint8))


def parse_metrics_from_log(log_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    sections = {"baseline": {}, "lsm": {}}
    current = None
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Baseline (raw sequence + ridge heads)"):
            current = "baseline"
            continue
        if stripped.startswith("LSM features + ridge heads"):
            current = "lsm"
            continue
        if current is None:
            continue
        parts = stripped.split()
        if len(parts) == 6 and parts[0] in {"val", "iid_test", "interp_test", "ood_test"}:
            sections[current][parts[0]] = {
                "class_acc": float(parts[1]),
                "param_mae": float(parts[2]),
                "active_mae": float(parts[3]),
                "radial_rmse": float(parts[4]),
                "geo_rmse": float(parts[5]),
            }
    return sections


def make_accuracy_chart(metrics: Dict[str, Dict[str, Dict[str, float]]], output: Path) -> None:
    splits = ["iid_test", "ood_test"]
    baseline = [metrics["baseline"][s]["class_acc"] for s in splits]
    lsm = [metrics["lsm"][s]["class_acc"] for s in splits]
    x = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    width = 0.32
    ax.bar(x - width / 2, baseline, width, label="Baseline", color="#f39c12")
    ax.bar(x + width / 2, lsm, width, label="LSM", color="#2980b9")

    ax.set_xticks(x)
    ax.set_xticklabels(["IID test", "OOD test"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Classification accuracy")
    ax.set_title("How much better the LSM performs")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    for i, v in enumerate(baseline):
        ax.text(i - width / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    for i, v in enumerate(lsm):
        ax.text(i + width / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def make_family_gallery_gif(theta: np.ndarray, output: Path) -> None:
    rng = np.random.default_rng(7)
    frames: List[Image.Image] = []

    for family in sl.FAMILY_NAMES:
        r, _, _ = sl.generate_curve(rng, family, theta, "val")
        x, y, _ = sl.normalize_curve(theta, r)
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=120)
        ax.plot(x, y, color="#1f77b4", linewidth=2.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.6, alpha=0.5)
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_title(f"Family: {family}", fontsize=14, pad=10)
        ax.text(
            0.02,
            0.02,
            "Generated from polar equation",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1100, loop=0)


def train_for_demo(seed: int = 42):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False, dtype=np.float32)
    splits = {
        "train": sl.build_split(rng, 600, theta, "train", 0.01),
        "iid_test": sl.build_split(rng, 200, theta, "iid_test", 0.01),
        "ood_test": sl.build_split(rng, 200, theta, "ood_test", 0.01),
    }

    raw_features = {k: v.sequences.reshape(v.sequences.shape[0], -1).astype(np.float32) for k, v in splits.items()}
    raw_scaler = sl.Standardizer().fit(raw_features["train"])
    raw_features = {k: raw_scaler.transform(v) for k, v in raw_features.items()}
    raw_heads = sl.train_heads(
        raw_features["train"], splits["train"].labels, splits["train"].params, splits["train"].radial_targets, 5e-2
    )

    lsm = sl.LiquidStateMachine(
        input_dim=2,
        num_excitatory=32,
        num_inhibitory=16,
        connectivity=0.05,
        spectral_radius=0.95,
        decay=0.9,
        threshold=1.0,
        reset_value=0.0,
        input_scale=0.8,
        seed=seed + 101,
    )
    lsm_features = {k: sl.extract_lsm_feature_matrix(lsm, v.sequences, bins=8) for k, v in splits.items()}
    lsm_scaler = sl.Standardizer().fit(lsm_features["train"])
    lsm_features = {k: lsm_scaler.transform(v) for k, v in lsm_features.items()}
    lsm_heads = sl.train_heads(
        lsm_features["train"], splits["train"].labels, splits["train"].params, splits["train"].radial_targets, 5e-2
    )

    return theta, splits, raw_features, lsm_features, raw_heads, lsm_heads


def pick_family_indices(labels: np.ndarray) -> List[int]:
    indices = []
    for family_id in range(len(sl.FAMILY_NAMES)):
        idx = int(np.where(labels == family_id)[0][0])
        indices.append(idx)
    return indices


def make_prediction_gif(
    split_name: str,
    theta: np.ndarray,
    split: sl.SplitData,
    raw_features: np.ndarray,
    lsm_features: np.ndarray,
    raw_heads: Dict[str, sl.RidgeReadout],
    lsm_heads: Dict[str, sl.RidgeReadout],
    output: Path,
) -> None:
    raw_pred_labels = np.argmax(raw_heads["class"].predict(raw_features), axis=1)
    lsm_pred_labels = np.argmax(lsm_heads["class"].predict(lsm_features), axis=1)
    raw_pred_r = raw_heads["recon"].predict(raw_features)
    lsm_pred_r = lsm_heads["recon"].predict(lsm_features)
    family_indices = pick_family_indices(split.labels)

    frames: List[Image.Image] = []
    for idx in family_indices:
        true_family = sl.FAMILY_NAMES[int(split.labels[idx])]
        pred_base = sl.FAMILY_NAMES[int(raw_pred_labels[idx])]
        pred_lsm = sl.FAMILY_NAMES[int(lsm_pred_labels[idx])]

        x_true = split.sequences[idx, :, 0]
        y_true = split.sequences[idx, :, 1]
        x_base = raw_pred_r[idx] * np.cos(theta)
        y_base = raw_pred_r[idx] * np.sin(theta)
        x_lsm = lsm_pred_r[idx] * np.cos(theta)
        y_lsm = lsm_pred_r[idx] * np.sin(theta)

        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=120)
        ax.plot(x_true, y_true, color="black", linewidth=2.7, label="Ground truth")
        ax.plot(x_base, y_base, color="#e67e22", linestyle="--", linewidth=2.2, label=f"Baseline ({pred_base})")
        ax.plot(x_lsm, y_lsm, color="#2980b9", linewidth=2.2, label=f"LSM ({pred_lsm})")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_title(f"{split_name.upper()} demo: {true_family}", fontsize=14)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1300, loop=0)


def make_reservoir_gif(theta: np.ndarray, output: Path) -> None:
    rng = np.random.default_rng(123)
    r, _, _ = sl.generate_curve(rng, "rose", theta, "val")
    x, y, _ = sl.normalize_curve(theta, r)
    sequence = np.stack([x, y], axis=1).astype(np.float32)

    lsm = sl.LiquidStateMachine(
        input_dim=2,
        num_excitatory=32,
        num_inhibitory=16,
        connectivity=0.05,
        spectral_radius=0.95,
        decay=0.9,
        threshold=1.0,
        reset_value=0.0,
        input_scale=0.8,
        seed=88,
    )

    spike_trace = np.zeros((sequence.shape[0], lsm.num_total), dtype=np.float32)
    for t in range(sequence.shape[0]):
        spike_trace[t] = lsm.step(sequence[t])

    frames: List[Image.Image] = []
    for t in range(8, sequence.shape[0] + 1, 4):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=120)

        axes[0].plot(x, y, color="lightgray", linewidth=1.5)
        axes[0].plot(x[:t], y[:t], color="#1f77b4", linewidth=2.7)
        axes[0].scatter([x[t - 1]], [y[t - 1]], color="#d62728", s=35)
        axes[0].set_title("Input curve scanned over time")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(-1.2, 1.2)
        axes[0].set_ylim(-1.2, 1.2)
        axes[0].grid(alpha=0.2, linestyle="--")

        im = axes[1].imshow(
            spike_trace[:t].T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
        )
        axes[1].set_title("Reservoir spikes (neurons × time)")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Neuron index")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=180, loop=0)


def main() -> None:
    best_log = ROOT / "run-phase2-conn0.05.log"
    metrics = parse_metrics_from_log(best_log)
    make_accuracy_chart(metrics, ASSET_DIR / "accuracy_comparison.png")

    theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False, dtype=np.float32)
    make_family_gallery_gif(theta, ASSET_DIR / "curve_families.gif")
    make_reservoir_gif(theta, ASSET_DIR / "reservoir_spikes.gif")

    demo_theta, splits, raw_feat, lsm_feat, raw_heads, lsm_heads = train_for_demo(seed=42)
    make_prediction_gif(
        "iid",
        demo_theta,
        splits["iid_test"],
        raw_feat["iid_test"],
        lsm_feat["iid_test"],
        raw_heads,
        lsm_heads,
        ASSET_DIR / "prediction_iid.gif",
    )
    make_prediction_gif(
        "ood",
        demo_theta,
        splits["ood_test"],
        raw_feat["ood_test"],
        lsm_feat["ood_test"],
        raw_heads,
        lsm_heads,
        ASSET_DIR / "prediction_ood.gif",
    )
    print(f"Presentation assets saved to: {ASSET_DIR}")


if __name__ == "__main__":
    main()
