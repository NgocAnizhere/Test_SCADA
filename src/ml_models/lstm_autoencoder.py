"""
LSTM Autoencoder - phát hiện bất thường bằng reconstruction error.

Sử dụng PyTorch để xây dựng LSTM Encoder-Decoder architecture.
Nếu PyTorch không khả dụng, fallback sang TensorFlow/Keras.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Kiểm tra framework khả dụng
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    BACKEND = "pytorch"
except ImportError:
    BACKEND = "tensorflow"
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        BACKEND = "none"
        logger.warning("Không có PyTorch hay TensorFlow. LSTM sẽ không hoạt động.")


# ==============================================================================
# PyTorch implementation
# ==============================================================================

if BACKEND == "pytorch":
    import torch
    import torch.nn as nn

    class _LSTMEncoder(nn.Module):
        """LSTM Encoder."""

        def __init__(self, n_features: int, encoding_dim: int = 64) -> None:
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
            self.lstm2 = nn.LSTM(128, encoding_dim, batch_first=True)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", Tuple]:
            out, _ = self.lstm1(x)
            out, hidden = self.lstm2(out)
            return out, hidden

    class _LSTMDecoder(nn.Module):
        """LSTM Decoder."""

        def __init__(self, n_features: int, encoding_dim: int = 64) -> None:
            super().__init__()
            self.lstm1 = nn.LSTM(encoding_dim, encoding_dim, batch_first=True)
            self.lstm2 = nn.LSTM(encoding_dim, 128, batch_first=True)
            self.fc = nn.Linear(128, n_features)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm1(x)
            out, _ = self.lstm2(out)
            out = self.fc(out)
            return out

    class _LSTMAutoencoderModel(nn.Module):
        """LSTM Autoencoder Model (PyTorch)."""

        def __init__(self, n_features: int, encoding_dim: int = 64) -> None:
            super().__init__()
            self.encoder = _LSTMEncoder(n_features, encoding_dim)
            self.decoder = _LSTMDecoder(n_features, encoding_dim)

        def forward(
            self, x: "torch.Tensor"
        ) -> "torch.Tensor":
            enc_out, _ = self.encoder(x)
            # Dùng output của encoder để decode
            dec_out = self.decoder(enc_out)
            return dec_out


class LSTMAutoencoder:
    """LSTM Autoencoder cho phát hiện bất thường trong chuỗi thời gian SCADA.

    Architecture:
        Input (sequence_length, n_features)
        → LSTM Encoder (128) → LSTM (encoding_dim)
        → LSTM Decoder (encoding_dim) → LSTM (128)
        → Dense(n_features)

    Attributes:
        sequence_length (int): Độ dài chuỗi đầu vào.
        n_features (int): Số đặc trưng.
        encoding_dim (int): Chiều của encoding.
        threshold (float): Ngưỡng reconstruction error cho anomaly.
    """

    def __init__(
        self,
        sequence_length: int = 48,
        n_features: int = 4,
        encoding_dim: int = 64,
    ) -> None:
        """Khởi tạo LSTMAutoencoder.

        Args:
            sequence_length: Độ dài chuỗi đầu vào.
            n_features: Số đặc trưng.
            encoding_dim: Chiều encoding.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.threshold: Optional[float] = None
        self.model: Optional[object] = None
        self._backend = BACKEND
        logger.debug(
            "LSTMAutoencoder: seq_len=%d, n_features=%d, encoding_dim=%d, backend=%s",
            sequence_length, n_features, encoding_dim, BACKEND,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_model(
        self,
        sequence_length: Optional[int] = None,
        n_features: Optional[int] = None,
        encoding_dim: Optional[int] = None,
    ) -> object:
        """Xây dựng model LSTM Autoencoder.

        Args:
            sequence_length: Ghi đè self.sequence_length nếu không None.
            n_features: Ghi đè self.n_features nếu không None.
            encoding_dim: Ghi đè self.encoding_dim nếu không None.

        Returns:
            Model đã xây dựng.
        """
        seq_len = sequence_length or self.sequence_length
        n_feat = n_features or self.n_features
        enc_dim = encoding_dim or self.encoding_dim

        if self._backend == "pytorch":
            self.model = _LSTMAutoencoderModel(n_feat, enc_dim)
            logger.info("LSTM Autoencoder (PyTorch) đã xây dựng")
        elif self._backend == "tensorflow":
            self.model = self._build_keras_model(seq_len, n_feat, enc_dim)
            logger.info("LSTM Autoencoder (Keras) đã xây dựng")
        else:
            raise RuntimeError("Không có PyTorch hay TensorFlow khả dụng.")

        return self.model

    def _build_keras_model(
        self, seq_len: int, n_features: int, enc_dim: int
    ) -> object:
        """Xây dựng model bằng Keras.

        Args:
            seq_len: Độ dài chuỗi.
            n_features: Số đặc trưng.
            enc_dim: Chiều encoding.

        Returns:
            Keras model.
        """
        from tensorflow import keras

        inputs = keras.Input(shape=(seq_len, n_features))
        # Encoder
        x = keras.layers.LSTM(128, return_sequences=True)(inputs)
        x = keras.layers.LSTM(enc_dim, return_sequences=True)(x)
        # Decoder
        x = keras.layers.LSTM(enc_dim, return_sequences=True)(x)
        x = keras.layers.LSTM(128, return_sequences=True)(x)
        outputs = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        learning_rate: float = 1e-3,
        patience: int = 10,
    ) -> object:
        """Huấn luyện model với EarlyStopping.

        Args:
            X_train: Array shape (n_samples, sequence_length, n_features).
            epochs: Số epochs tối đa.
            batch_size: Batch size.
            validation_split: Tỉ lệ validation.
            learning_rate: Learning rate.
            patience: Số epochs chờ trước khi early stop.

        Returns:
            History object.
        """
        if self.model is None:
            self.build_model()

        if self._backend == "pytorch":
            return self._train_pytorch(
                X_train, epochs, batch_size, validation_split, learning_rate, patience
            )
        else:
            return self._train_keras(
                X_train, epochs, batch_size, validation_split, patience
            )

    def _train_pytorch(
        self,
        X_train: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float,
        learning_rate: float,
        patience: int,
    ) -> dict:
        """Huấn luyện bằng PyTorch.

        Args:
            X_train: Dữ liệu train.
            epochs: Số epochs.
            batch_size: Batch size.
            validation_split: Tỉ lệ validation.
            learning_rate: Learning rate.
            patience: Số epochs cho early stopping.

        Returns:
            Dict history với train_loss, val_loss.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Tách validation
        n_val = max(1, int(len(X_train) * validation_split))
        X_val_arr = X_train[-n_val:]
        X_tr_arr = X_train[:-n_val]

        X_tr_t = torch.FloatTensor(X_tr_arr)
        X_val_t = torch.FloatTensor(X_val_arr)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, X_tr_t),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, X_val_t),
            batch_size=batch_size,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate  # type: ignore[union-attr]
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        best_state = None

        self.model.train()  # type: ignore[union-attr]
        for epoch in range(epochs):
            # Train
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)  # type: ignore[operator]
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()  # type: ignore[union-attr]
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)  # type: ignore[operator]
                    loss = criterion(output, y_batch)
                    val_losses.append(loss.item())
            self.model.train()  # type: ignore[union-attr]

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone()
                    for k, v in self.model.state_dict().items()  # type: ignore[union-attr]
                }
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch [%d/%d] train_loss=%.6f, val_loss=%.6f",
                    epoch + 1, epochs, train_loss, val_loss,
                )

            if patience_counter >= patience:
                logger.info("Early stopping tại epoch %d", epoch + 1)
                break

        # Khôi phục best model
        if best_state is not None:
            self.model.load_state_dict(best_state)  # type: ignore[union-attr]

        return history

    def _train_keras(
        self,
        X_train: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float,
        patience: int,
    ) -> object:
        """Huấn luyện bằng Keras.

        Args:
            X_train: Dữ liệu train.
            epochs: Số epochs.
            batch_size: Batch size.
            validation_split: Tỉ lệ validation.
            patience: Số epochs cho early stopping.

        Returns:
            Keras History object.
        """
        from tensorflow import keras

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        ]
        history = self.model.fit(  # type: ignore[union-attr]
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )
        return history

    # ------------------------------------------------------------------
    # Reconstruction error
    # ------------------------------------------------------------------

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Tính MSE reconstruction error cho từng mẫu.

        Args:
            X: Array shape (n_samples, sequence_length, n_features).

        Returns:
            Array shape (n_samples,) với MSE của từng mẫu.
        """
        if self.model is None:
            raise RuntimeError("Model chưa được build/train.")

        if self._backend == "pytorch":
            import torch
            self.model.eval()  # type: ignore[union-attr]
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                X_rec = self.model(X_t).numpy()  # type: ignore[operator]
        else:
            X_rec = self.model.predict(X, verbose=0)  # type: ignore[union-attr]

        # MSE theo từng mẫu
        mse = np.mean((X - X_rec) ** 2, axis=(1, 2))
        return mse

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Phát hiện bất thường dựa trên reconstruction error.

        Args:
            X: Array shape (n_samples, sequence_length, n_features).
            threshold: Ngưỡng. Dùng self.threshold nếu None.

        Returns:
            Tuple (is_anomaly, reconstruction_errors):
                - is_anomaly: Boolean array.
                - reconstruction_errors: MSE array.
        """
        errors = self.get_reconstruction_error(X)

        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            # Tự động tính ngưỡng
            threshold = np.mean(errors) + 3 * np.std(errors)
            logger.info("Tự động tính threshold=%.6f", threshold)

        is_anomaly = errors > threshold
        logger.info(
            "LSTM Anomaly detection: %d/%d anomalies (threshold=%.6f)",
            np.sum(is_anomaly), len(X), threshold,
        )
        return is_anomaly, errors

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def set_threshold(
        self,
        X_normal: np.ndarray,
        percentile: float = 99.0,
    ) -> float:
        """Tự động xác định ngưỡng từ dữ liệu bình thường.

        Args:
            X_normal: Dữ liệu bình thường (không có anomaly).
            percentile: Phần trăm để xác định ngưỡng.

        Returns:
            Threshold value đã được set.
        """
        errors = self.get_reconstruction_error(X_normal)
        self.threshold = float(np.percentile(errors, percentile))
        logger.info(
            "Threshold set tại percentile %.1f = %.6f", percentile, self.threshold
        )
        return self.threshold

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------

    def plot_reconstruction(
        self,
        X: np.ndarray,
        idx: int = 0,
        feature_idx: int = 0,
    ) -> Optional[object]:
        """Visualize gốc vs reconstructed cho một mẫu.

        Args:
            X: Array shape (n_samples, sequence_length, n_features).
            idx: Index của mẫu cần vẽ.
            feature_idx: Index của feature cần vẽ.

        Returns:
            matplotlib Figure hoặc None.
        """
        try:
            import matplotlib.pyplot as plt

            if self.model is None:
                logger.warning("Model chưa được train.")
                return None

            if self._backend == "pytorch":
                import torch
                self.model.eval()  # type: ignore[union-attr]
                with torch.no_grad():
                    X_t = torch.FloatTensor(X[idx:idx + 1])
                    X_rec = self.model(X_t).numpy()[0]  # type: ignore[operator]
            else:
                X_rec = self.model.predict(X[idx:idx + 1], verbose=0)[0]  # type: ignore[union-attr]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(X[idx, :, feature_idx], label="Original", color="blue")
            ax.plot(X_rec[:, feature_idx], label="Reconstructed", color="red", linestyle="--")
            ax.set_title(f"LSTM Reconstruction - Sample {idx}, Feature {feature_idx}")
            ax.legend()
            ax.set_xlabel("Time step")
            ax.set_ylabel("Value")
            plt.tight_layout()
            return fig
        except ImportError:
            logger.warning("matplotlib chưa cài. Không thể plot.")
            return None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Lưu model.

        Args:
            path: Đường dẫn file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "pytorch":
            import torch
            torch.save({
                "model_state_dict": self.model.state_dict(),  # type: ignore[union-attr]
                "config": {
                    "sequence_length": self.sequence_length,
                    "n_features": self.n_features,
                    "encoding_dim": self.encoding_dim,
                },
                "threshold": self.threshold,
            }, str(path))
        elif self._backend == "tensorflow":
            self.model.save(str(path))  # type: ignore[union-attr]

        logger.info("LSTM model lưu tại: %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Tải model.

        Args:
            path: Đường dẫn file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        if self._backend == "pytorch":
            import torch
            checkpoint = torch.load(str(path), map_location="cpu")
            config = checkpoint["config"]
            self.sequence_length = config["sequence_length"]
            self.n_features = config["n_features"]
            self.encoding_dim = config["encoding_dim"]
            self.threshold = checkpoint.get("threshold")
            self.build_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[union-attr]
        elif self._backend == "tensorflow":
            from tensorflow import keras
            self.model = keras.models.load_model(str(path))

        logger.info("LSTM model đã tải: %s", path)
