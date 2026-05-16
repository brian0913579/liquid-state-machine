import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

FAMILY_NAMES = ("limacon", "cardioid", "rose", "circle", "lemniscate", "conic")
PARAM_DIM = 6  # [a, b, n, e, l, form_code]


@dataclass
class SplitData:
    sequences: np.ndarray  # (N, T, 2) -> normalized (x, y)
    radial_targets: np.ndarray  # (N, T) -> normalized r(theta)
    labels: np.ndarray  # (N,)
    params: np.ndarray  # (N, PARAM_DIM)
    active_mask: np.ndarray  # (N, PARAM_DIM)


class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer must be fit before transform().")
        return (x - self.mean) / self.std


class RidgeReadout:
    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        x_bias = np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
        reg = self.alpha * np.eye(x_bias.shape[1], dtype=x.dtype)
        self.weights = np.linalg.solve(x_bias.T @ x_bias + reg, x_bias.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("RidgeReadout must be fit before predict().")
        x_bias = np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
        return x_bias @ self.weights


class LiquidStateMachine:
    def __init__(
        self,
        input_dim: int,
        num_excitatory: int = 56,
        num_inhibitory: int = 24,
        connectivity: float = 0.1,
        spectral_radius: float = 0.95,
        decay: float = 0.9,
        threshold: float = 1.0,
        reset_value: float = 0.0,
        input_scale: float = 0.8,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.num_exc = num_excitatory
        self.num_inh = num_inhibitory
        self.num_total = self.num_exc + self.num_inh
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.decay = decay
        self.threshold = threshold
        self.reset_value = reset_value

        self.rng = np.random.default_rng(seed)
        self.recurrent = self._init_recurrent()
        self.w_in = self.rng.normal(0.0, input_scale, size=(self.num_total, self.input_dim)).astype(np.float32)
        self.voltage = np.zeros(self.num_total, dtype=np.float32)
        self.spikes = np.zeros(self.num_total, dtype=np.float32)

    def _init_recurrent(self) -> np.ndarray:
        weights = self.rng.normal(
            0.0, 1.0 / np.sqrt(self.num_total), size=(self.num_total, self.num_total)
        ).astype(np.float32)
        mask = (self.rng.random((self.num_total, self.num_total)) < self.connectivity).astype(np.float32)
        weights *= mask
        np.fill_diagonal(weights, 0.0)

        # Dale's principle by presynaptic neuron type (column sign).
        weights[:, : self.num_exc] = np.abs(weights[:, : self.num_exc])
        weights[:, self.num_exc :] = -np.abs(weights[:, self.num_exc :])

        eigvals = np.linalg.eigvals(weights)
        radius = np.max(np.abs(eigvals))
        if radius > 0:
            weights *= self.spectral_radius / float(radius)
        return weights.astype(np.float32)

    def reset_state(self) -> None:
        self.voltage.fill(0.0)
        self.spikes.fill(0.0)

    def step(self, input_vec: np.ndarray) -> np.ndarray:
        recurrent_current = self.recurrent @ self.spikes
        input_current = self.w_in @ input_vec
        self.voltage = self.decay * self.voltage + recurrent_current + input_current
        self.spikes = (self.voltage >= self.threshold).astype(np.float32)
        self.voltage[self.spikes > 0] = self.reset_value
        return self.spikes

    def extract_features(self, sequence: np.ndarray, bins: int = 8) -> np.ndarray:
        self.reset_state()
        spike_trace = np.zeros((sequence.shape[0], self.num_total), dtype=np.float32)
        for t, input_vec in enumerate(sequence):
            spike_trace[t] = self.step(input_vec)

        pooled = [chunk.mean(axis=0) for chunk in np.array_split(spike_trace, bins)]
        return np.concatenate(pooled + [self.voltage.copy()], axis=0)


def sample_from_bands(rng: np.random.Generator, bands: Tuple[Tuple[float, float], ...]) -> float:
    idx = rng.integers(0, len(bands))
    low, high = bands[idx]
    return float(rng.uniform(low, high))


def sample_scale(rng: np.random.Generator, mode: str) -> float:
    if mode == "interp_test":
        return float(rng.uniform(1.0, 1.2))
    if mode == "ood_test":
        return float(rng.uniform(1.8, 2.5))
    return sample_from_bands(rng, ((0.8, 1.0), (1.2, 1.6)))


def sample_eccentricity(rng: np.random.Generator, mode: str) -> float:
    if mode == "interp_test":
        return float(rng.uniform(0.45, 0.65))
    if mode == "ood_test":
        return float(rng.uniform(1.05, 1.35))
    return sample_from_bands(rng, ((0.2, 0.45), (0.65, 0.9)))


def generate_curve(
    rng: np.random.Generator,
    family: str,
    theta: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.zeros(PARAM_DIM, dtype=np.float32)
    active = np.zeros(PARAM_DIM, dtype=np.float32)

    if family == "limacon":
        b = sample_scale(rng, mode)
        if mode == "interp_test":
            ratio = float(rng.uniform(0.9, 1.1))
        elif mode == "ood_test":
            ratio = sample_from_bands(rng, ((0.1, 0.25), (2.4, 3.2)))
        else:
            ratio = sample_from_bands(rng, ((0.35, 0.85), (1.15, 2.2)))
        a = ratio * b
        use_cos = bool(rng.integers(0, 2))
        trig = np.cos(theta) if use_cos else np.sin(theta)
        r = a + b * trig
        params[[0, 1, 5]] = (a, b, 1.0 if use_cos else 2.0)
        active[[0, 1, 5]] = 1.0
        return r, params, active

    if family == "cardioid":
        a = sample_scale(rng, mode)
        use_cos = bool(rng.integers(0, 2))
        trig = np.cos(theta) if use_cos else np.sin(theta)
        r = a * (1.0 + trig)
        params[[0, 1, 5]] = (a, a, 1.0 if use_cos else 2.0)
        active[[0, 1, 5]] = 1.0
        return r, params, active

    if family == "rose":
        a = sample_scale(rng, mode)
        n = int(rng.choice([6, 7, 8] if mode == "ood_test" else [2, 3, 4, 5]))
        use_cos = bool(rng.integers(0, 2))
        trig = np.cos(n * theta) if use_cos else np.sin(n * theta)
        r = a * trig
        params[[0, 2, 5]] = (a, float(n), 1.0 if use_cos else 2.0)
        active[[0, 2, 5]] = 1.0
        return r, params, active

    if family == "circle":
        a = sample_scale(rng, mode)
        form = int(rng.choice([0, 1, 2]))  # 0: r=a, 1: a cos(theta), 2: a sin(theta)
        if form == 0:
            r = np.full_like(theta, a)
        elif form == 1:
            r = a * np.cos(theta)
        else:
            r = a * np.sin(theta)
        params[[0, 5]] = (a, float(form))
        active[[0, 5]] = 1.0
        return r, params, active

    if family == "lemniscate":
        a = sample_scale(rng, mode)
        use_cos = bool(rng.integers(0, 2))
        core = np.cos(2.0 * theta) if use_cos else np.sin(2.0 * theta)
        r = np.sign(core) * a * np.sqrt(np.abs(core))
        params[[0, 2, 5]] = (a, 2.0, 1.0 if use_cos else 2.0)
        active[[0, 2, 5]] = 1.0
        return r, params, active

    if family == "conic":
        l = sample_scale(rng, mode)
        e = sample_eccentricity(rng, mode)
        use_cos = bool(rng.integers(0, 2))
        sign = int(rng.choice([-1, 1]))
        trig = np.cos(theta) if use_cos else np.sin(theta)
        denominator = 1.0 + sign * e * trig
        near_zero = np.abs(denominator) < 0.15
        if np.any(near_zero):
            denominator[near_zero] = 0.15 * np.sign(denominator[near_zero] + 1e-6)
        r = np.clip(l / denominator, -4.0 * l, 4.0 * l)
        form_code = 11 if (use_cos and sign > 0) else 12 if (use_cos and sign < 0) else 13 if sign > 0 else 14
        params[[3, 4, 5]] = (e, l, float(form_code))
        active[[3, 4, 5]] = 1.0
        return r, params, active

    raise ValueError(f"Unknown family: {family}")


def normalize_curve(theta: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    scale = float(np.max(np.sqrt(x**2 + y**2)))
    if scale < 1e-8:
        scale = 1.0
    return (x / scale).astype(np.float32), (y / scale).astype(np.float32), (r / scale).astype(np.float32)


def build_split(
    rng: np.random.Generator,
    n_samples: int,
    theta: np.ndarray,
    mode: str,
    train_noise_std: float,
) -> SplitData:
    sequences = np.zeros((n_samples, theta.shape[0], 2), dtype=np.float32)
    radial_targets = np.zeros((n_samples, theta.shape[0]), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)
    params = np.zeros((n_samples, PARAM_DIM), dtype=np.float32)
    active_mask = np.zeros((n_samples, PARAM_DIM), dtype=np.float32)

    for i in range(n_samples):
        label = int(rng.integers(0, len(FAMILY_NAMES)))
        family = FAMILY_NAMES[label]
        r, p, active = generate_curve(rng, family, theta, mode)
        x, y, r_norm = normalize_curve(theta, r)

        if mode == "train" and train_noise_std > 0:
            x = x + rng.normal(0.0, train_noise_std, size=x.shape).astype(np.float32)
            y = y + rng.normal(0.0, train_noise_std, size=y.shape).astype(np.float32)

        sequences[i, :, 0] = x
        sequences[i, :, 1] = y
        radial_targets[i] = r_norm
        labels[i] = label
        params[i] = p
        active_mask[i] = active

    return SplitData(
        sequences=sequences,
        radial_targets=radial_targets,
        labels=labels,
        params=params,
        active_mask=active_mask,
    )


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train_heads(
    x_train: np.ndarray,
    labels: np.ndarray,
    params: np.ndarray,
    radial_targets: np.ndarray,
    alpha: float,
) -> Dict[str, RidgeReadout]:
    heads = {
        "class": RidgeReadout(alpha).fit(x_train, one_hot(labels, len(FAMILY_NAMES))),
        "param": RidgeReadout(alpha).fit(x_train, params),
        "recon": RidgeReadout(alpha).fit(x_train, radial_targets),
    }
    return heads


def evaluate(
    heads: Dict[str, RidgeReadout],
    features: np.ndarray,
    split: SplitData,
    theta: np.ndarray,
) -> Dict[str, float]:
    class_logits = heads["class"].predict(features)
    pred_labels = np.argmax(class_logits, axis=1)
    class_acc = float((pred_labels == split.labels).mean())

    pred_params = heads["param"].predict(features)
    mae_all = float(np.mean(np.abs(pred_params - split.params)))
    active_denom = float(np.maximum(split.active_mask.sum(), 1.0))
    active_mae = float(np.sum(np.abs(pred_params - split.params) * split.active_mask) / active_denom)

    pred_r = heads["recon"].predict(features)
    radial_rmse = float(np.sqrt(np.mean((pred_r - split.radial_targets) ** 2)))

    x_pred = pred_r * np.cos(theta)[None, :]
    y_pred = pred_r * np.sin(theta)[None, :]
    geo_rmse = float(
        np.sqrt(np.mean((x_pred - split.sequences[:, :, 0]) ** 2 + (y_pred - split.sequences[:, :, 1]) ** 2))
    )
    return {
        "class_acc": class_acc,
        "param_mae_all": mae_all,
        "param_mae_active": active_mae,
        "radial_rmse": radial_rmse,
        "geo_rmse": geo_rmse,
    }


def print_metrics_table(title: str, all_metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print(
        "split".ljust(12)
        + "class_acc".rjust(12)
        + "param_mae".rjust(12)
        + "active_mae".rjust(12)
        + "radial_rmse".rjust(14)
        + "geo_rmse".rjust(12)
    )
    for split_name in ("val", "iid_test", "interp_test", "ood_test"):
        m = all_metrics[split_name]
        print(
            split_name.ljust(12)
            + f"{m['class_acc']:.4f}".rjust(12)
            + f"{m['param_mae_all']:.4f}".rjust(12)
            + f"{m['param_mae_active']:.4f}".rjust(12)
            + f"{m['radial_rmse']:.4f}".rjust(14)
            + f"{m['geo_rmse']:.4f}".rjust(12)
        )


def extract_lsm_feature_matrix(lsm: LiquidStateMachine, sequences: np.ndarray, bins: int) -> np.ndarray:
    first = lsm.extract_features(sequences[0], bins=bins)
    feature_matrix = np.zeros((sequences.shape[0], first.shape[0]), dtype=np.float32)
    feature_matrix[0] = first
    for i in range(1, sequences.shape[0]):
        feature_matrix[i] = lsm.extract_features(sequences[i], bins=bins)
    return feature_matrix


def run_experiment(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    theta = np.linspace(0.0, 2.0 * np.pi, args.theta_steps, endpoint=False, dtype=np.float32)

    splits = {
        "train": build_split(rng, args.train_samples, theta, "train", args.train_noise_std),
        "val": build_split(rng, args.val_samples, theta, "val", args.train_noise_std),
        "iid_test": build_split(rng, args.test_samples, theta, "iid_test", args.train_noise_std),
        "interp_test": build_split(rng, args.test_samples, theta, "interp_test", args.train_noise_std),
        "ood_test": build_split(rng, args.test_samples, theta, "ood_test", args.train_noise_std),
    }

    if args.save_dataset:
        np.savez_compressed(
            args.save_dataset,
            theta=theta,
            train_sequences=splits["train"].sequences,
            train_radial=splits["train"].radial_targets,
            train_labels=splits["train"].labels,
            train_params=splits["train"].params,
            val_sequences=splits["val"].sequences,
            val_radial=splits["val"].radial_targets,
            val_labels=splits["val"].labels,
            val_params=splits["val"].params,
            iid_sequences=splits["iid_test"].sequences,
            iid_radial=splits["iid_test"].radial_targets,
            iid_labels=splits["iid_test"].labels,
            iid_params=splits["iid_test"].params,
            interp_sequences=splits["interp_test"].sequences,
            interp_radial=splits["interp_test"].radial_targets,
            interp_labels=splits["interp_test"].labels,
            interp_params=splits["interp_test"].params,
            ood_sequences=splits["ood_test"].sequences,
            ood_radial=splits["ood_test"].radial_targets,
            ood_labels=splits["ood_test"].labels,
            ood_params=splits["ood_test"].params,
        )
        print(f"Saved generated dataset to {args.save_dataset}")

    if not args.skip_baseline:
        raw_features = {
            name: split.sequences.reshape(split.sequences.shape[0], -1).astype(np.float32) for name, split in splits.items()
        }
        raw_scaler = Standardizer().fit(raw_features["train"])
        raw_features = {name: raw_scaler.transform(feat) for name, feat in raw_features.items()}

        baseline_heads = train_heads(
            raw_features["train"],
            splits["train"].labels,
            splits["train"].params,
            splits["train"].radial_targets,
            args.ridge_alpha,
        )
        baseline_metrics = {
            name: evaluate(baseline_heads, raw_features[name], split, theta)
            for name, split in splits.items()
            if name != "train"
        }
        print_metrics_table("Baseline (raw sequence + ridge heads)", baseline_metrics)

    if not args.skip_lsm:
        lsm = LiquidStateMachine(
            input_dim=2,
            num_excitatory=args.lsm_exc,
            num_inhibitory=args.lsm_inh,
            connectivity=args.lsm_connectivity,
            spectral_radius=args.lsm_spectral_radius,
            decay=args.lsm_decay,
            threshold=args.lsm_threshold,
            reset_value=args.lsm_reset,
            input_scale=args.lsm_input_scale,
            seed=args.seed + 101,
        )

        lsm_features = {name: extract_lsm_feature_matrix(lsm, split.sequences, bins=args.lsm_bins) for name, split in splits.items()}
        lsm_scaler = Standardizer().fit(lsm_features["train"])
        lsm_features = {name: lsm_scaler.transform(feat) for name, feat in lsm_features.items()}

        lsm_heads = train_heads(
            lsm_features["train"],
            splits["train"].labels,
            splits["train"].params,
            splits["train"].radial_targets,
            args.ridge_alpha,
        )
        lsm_metrics = {
            name: evaluate(lsm_heads, lsm_features[name], split, theta)
            for name, split in splits.items()
            if name != "train"
        }
        print_metrics_table("LSM features + ridge heads", lsm_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSM pipeline for special polar graph learning.")
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--theta-steps", type=int, default=96)
    parser.add_argument("--train-noise-std", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dataset", type=str, default="")

    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-lsm", action="store_true")

    parser.add_argument("--lsm-exc", type=int, default=56)
    parser.add_argument("--lsm-inh", type=int, default=24)
    parser.add_argument("--lsm-connectivity", type=float, default=0.1)
    parser.add_argument("--lsm-spectral-radius", type=float, default=0.95)
    parser.add_argument("--lsm-decay", type=float, default=0.9)
    parser.add_argument("--lsm-threshold", type=float, default=1.0)
    parser.add_argument("--lsm-reset", type=float, default=0.0)
    parser.add_argument("--lsm-input-scale", type=float, default=0.8)
    parser.add_argument("--lsm-bins", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
