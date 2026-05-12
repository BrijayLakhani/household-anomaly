"""
lstm_ae.py — LSTM Autoencoder for sequence-level anomaly detection.

Architecture rationale
----------------------
The encoder–bottleneck–decoder structure forces the model to learn a compact
representation of *normal* power-consumption patterns.  At inference time,
sequences the model has never seen (anomalous patterns) are poorly
reconstructed → high MSE → anomaly flag.

Layer-by-layer choices
  LSTM(32, return_sequences=True)  : captures local temporal interactions
                                     across all 60 timesteps.
  Dropout(0.1)                     : light regularisation; prevents
                                     memorising training noise.
  LSTM(16, return_sequences=False) : compresses to a 16-dim summary vector
                                     (information bottleneck — model must
                                     generalise, not memorise).
  RepeatVector(60)                 : replicates the summary 60× to seed the
                                     decoder's recurrent state.
  LSTM(16) → LSTM(32)              : mirrored decoder; gradually expands the
                                     representation back to sequence length.
  TimeDistributed(Dense(7))        : independently maps each decoded timestep
                                     to the 7 original feature values.

Loss: MSE — smooth gradient, directly interpretable as reconstruction error.
Optimizer: Adam(lr=1e-3) — adaptive learning rate, robust default for LSTMs.

Training notes
  EarlyStopping(patience=2) stops training if validation loss stagnates,
  preventing over-fitting while avoiding wasteful computation.
  ReduceLROnPlateau(patience=1) halves the LR after one stagnant epoch,
  helping escape local minima without manual LR scheduling.
  TimeoutCallback aborts after 20 min on slow hardware (CPU with stride=1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _require_tf() -> None:
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed.\n"
            "Python 3.14 is not yet supported by stable TF releases.\n"
            "Options:\n"
            "  pip install tf-nightly          (may have Py3.14 wheels)\n"
            "  Use Python 3.12 + tensorflow==2.19.0\n"
        )


# ---------------------------------------------------------------------------
# Timeout callback
# ---------------------------------------------------------------------------

if TF_AVAILABLE:

    class _TimeoutCallback(keras.callbacks.Callback):
        """Abort training if wall-clock time exceeds max_seconds."""

        def __init__(self, max_seconds: int = 1200) -> None:
            super().__init__()
            self.max_seconds = max_seconds
            self._start: float = 0.0

        def on_train_begin(self, logs: dict | None = None) -> None:
            self._start = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            elapsed = time.perf_counter() - self._start
            if elapsed > self.max_seconds:
                print(
                    f"\n[TimeoutCallback] {elapsed:.0f}s > {self.max_seconds}s limit. "
                    "Stopping training. Re-run with fewer epochs or --stride > 1."
                )
                self.model.stop_training = True


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def build_lstm_ae(window: int = 60, n_features: int = 7) -> "keras.Model":
    """
    Construct the LSTM Autoencoder graph (untrained).

    Parameters
    ----------
    window:
        Number of timesteps per sequence (must match sequences.npy shape[1]).
    n_features:
        Number of sensor channels (7 raw features).

    Returns
    -------
    Compiled keras.Model ready for training.
    """
    _require_tf()
    tf.random.set_seed(42)

    inputs = keras.Input(shape=(window, n_features), name="input")

    # Encoder
    x = keras.layers.LSTM(32, return_sequences=True, name="enc_lstm1")(inputs)
    x = keras.layers.Dropout(0.1, name="enc_drop")(x)
    x = keras.layers.LSTM(16, return_sequences=False, name="enc_lstm2")(x)

    # Bottleneck
    x = keras.layers.RepeatVector(window, name="bottleneck")(x)

    # Decoder
    x = keras.layers.LSTM(16, return_sequences=True, name="dec_lstm1")(x)
    x = keras.layers.LSTM(32, return_sequences=True, name="dec_lstm2")(x)
    x = keras.layers.Dropout(0.1, name="dec_drop")(x)
    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(n_features), name="output"
    )(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_lstm_ae(
    X_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 512,
    val_split: float = 0.1,
) -> "tuple[keras.Model, dict]":
    """
    Train the LSTM-AE on normal (training) sequences.

    Parameters
    ----------
    X_train:
        float32 array (N, 60, 7) of scaled training sequences (no NaN).
    epochs:
        Maximum training epochs. EarlyStopping may halt earlier.
    batch_size:
        Mini-batch size. 512 balances GPU/CPU utilisation with memory use.
    val_split:
        Fraction of X_train held out for validation loss monitoring.

    Returns
    -------
    model:
        Trained keras.Model with best weights restored.
    history_dict:
        {'loss': [...], 'val_loss': [...], 'epochs_run': int,
         'training_seconds': float}
    """
    _require_tf()
    tf.random.set_seed(42)

    model = build_lstm_ae(window=X_train.shape[1], n_features=X_train.shape[2])
    model.summary(print_fn=lambda s: print(" ", s))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1,
        ),
        _TimeoutCallback(max_seconds=1200),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        X_train, X_train,          # autoencoder: target = input
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nLSTM-AE training complete in {elapsed:.1f}s")

    history_dict = {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "epochs_run": len(history.history["loss"]),
        "training_seconds": round(elapsed, 1),
    }
    return model, history_dict


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_lstm_ae(
    model: "keras.Model",
    X: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """
    Compute per-sequence reconstruction error and binary flag.

    Reconstruction error = mean over time steps and features of (x - x̂)².
    Batched prediction avoids OOM on large arrays.

    Parameters
    ----------
    model:
        Trained LSTM-AE.
    X:
        float32 array (N, 60, 7).
    timestamps:
        DatetimeIndex of end-of-window timestamps (length N).
    threshold:
        Reconstruction error above which a sequence is flagged.
    batch_size:
        Prediction batch size (default 1024).

    Returns
    -------
    pd.DataFrame with columns lstm_recon_error and lstm_flag,
    indexed by timestamps.
    """
    _require_tf()

    n = len(X)
    recon_errors = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X[start:end]
        pred = model.predict(batch, verbose=0)
        # MSE over time axis, then mean over features
        mse_per_feat = np.mean((batch - pred) ** 2, axis=1)   # (B, 7)
        recon_errors[start:end] = np.mean(mse_per_feat, axis=1)  # (B,)

    flag = (recon_errors > threshold).astype(float)

    return pd.DataFrame(
        {"lstm_recon_error": recon_errors, "lstm_flag": flag},
        index=timestamps,
    )


def compute_lstm_threshold(
    model: "keras.Model",
    X_train: np.ndarray,
    percentile: float = 95,
    batch_size: int = 1024,
) -> float:
    """
    Compute the reconstruction-error threshold from TRAINING sequences only.

    Using training-set percentile prevents test-set anomalies from raising
    the bar, which would reduce recall on the test set.
    """
    _require_tf()
    n = len(X_train)
    errors = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X_train[start:end]
        pred = model.predict(batch, verbose=0)
        mse = np.mean((batch - pred) ** 2, axis=1)
        errors[start:end] = np.mean(mse, axis=1)
    return float(np.percentile(errors, percentile))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_lstm_ae(
    model: "keras.Model",
    threshold: float,
    path_dir: Path,
) -> None:
    """
    Save model in Keras v3 format and threshold to a JSON sidecar.

    Keras v3 (.keras) bundles architecture + weights in one file and is
    forward-compatible with future TF releases.
    """
    _require_tf()
    path_dir.mkdir(parents=True, exist_ok=True)
    model_path = path_dir / "lstm_ae.keras"
    model.save(str(model_path))
    sidecar = path_dir / "lstm_ae_threshold.json"
    sidecar.write_text(
        json.dumps({"threshold": threshold, "percentile": 95}, indent=2),
        encoding="utf-8",
    )
    print(f"LSTM-AE model     -> {model_path}")
    print(f"LSTM-AE threshold -> {sidecar}  ({threshold:.6f})")


def load_lstm_ae(path_dir: Path) -> "tuple[keras.Model, float]":
    """Load a saved LSTM-AE and its threshold."""
    _require_tf()
    model = keras.models.load_model(str(path_dir / "lstm_ae.keras"))
    threshold = json.loads(
        (path_dir / "lstm_ae_threshold.json").read_text()
    )["threshold"]
    return model, threshold
