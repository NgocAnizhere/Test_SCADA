"""
Kaggle Downloader - tải dataset Wind Turbine SCADA từ Kaggle API.

Module này cung cấp class KaggleDownloader để tải và nạp dataset
"noobtube99/wind-turbine-scada-data" từ Kaggle.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class KaggleDownloader:
    """Tải và nạp dữ liệu Wind Turbine SCADA từ Kaggle.

    Attributes:
        dataset (str): Tên dataset Kaggle.
        raw_data_path (Path): Đường dẫn thư mục lưu dữ liệu gốc.
    """

    DATASET = "noobtube99/wind-turbine-scada-data"
    COLUMNS = {
        "datetime": "Date/Time",
        "active_power": "LV ActivePower (kW)",
        "wind_speed": "Wind Speed (m/s)",
        "theoretical_power": "Theoretical_Power_Curve (KWh)",
        "wind_direction": "Wind Direction (°)",
    }

    def __init__(self, raw_data_path: str = "data/raw") -> None:
        """Khởi tạo KaggleDownloader.

        Args:
            raw_data_path: Đường dẫn thư mục lưu dữ liệu gốc.
        """
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        logger.info("KaggleDownloader khởi tạo, raw_data_path=%s", self.raw_data_path)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def download_dataset(self) -> bool:
        """Tải dataset từ Kaggle API về thư mục raw_data_path.

        Returns:
            True nếu tải thành công, False nếu thất bại.

        Raises:
            RuntimeError: Khi không thể xác thực hoặc kết nối Kaggle API.
        """
        try:
            # Import ở đây để tránh lỗi nếu kaggle chưa được cài đặt
            import kaggle  # noqa: F401
            from kaggle.api.kaggle_api_extended import KaggleApiExtended

            api = KaggleApiExtended()
            api.authenticate()
            logger.info("Đã xác thực Kaggle API thành công")

            logger.info("Bắt đầu tải dataset: %s", self.DATASET)
            api.dataset_download_files(
                self.DATASET,
                path=str(self.raw_data_path),
                unzip=True,
            )
            logger.info("Tải dataset hoàn tất, lưu tại: %s", self.raw_data_path)
            return True

        except ImportError:
            logger.error(
                "Thư viện 'kaggle' chưa được cài đặt. Chạy: pip install kaggle"
            )
            return False
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Lỗi khi tải dataset: %s", exc)
            return False

    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Đọc file CSV và parse datetime, trả về DataFrame.

        Args:
            filename: Tên file CSV trong raw_data_path. Nếu None, tự động
                tìm file CSV đầu tiên.

        Returns:
            DataFrame với index là DatetimeIndex.

        Raises:
            FileNotFoundError: Khi không tìm thấy file CSV.
            ValueError: Khi file không có cột datetime hợp lệ.
        """
        csv_path = self._resolve_csv_path(filename)
        logger.info("Đọc dữ liệu từ: %s", csv_path)

        try:
            df = pd.read_csv(csv_path)
            df = self._parse_datetime(df)
            logger.info("Đã nạp %d hàng, %d cột", len(df), len(df.columns))
            return df
        except Exception as exc:
            logger.error("Lỗi khi đọc file CSV: %s", exc)
            raise

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Thống kê cơ bản về DataFrame.

        Args:
            df: DataFrame đã nạp từ load_data().

        Returns:
            Dict chứa: shape, dtypes, missing_values, date_range, describe.
        """
        datetime_col = self.COLUMNS["datetime"]

        # Xác định khoảng thời gian
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "total_days": (df.index.max() - df.index.min()).days,
            }
        elif datetime_col in df.columns:
            date_range = {
                "start": str(df[datetime_col].min()),
                "end": str(df[datetime_col].max()),
            }
        else:
            date_range = {}

        info: Dict[str, Any] = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "date_range": date_range,
            "describe": df.describe().to_dict(),
        }

        logger.info(
            "Thống kê dữ liệu - shape=%s, missing_total=%d",
            info["shape"],
            sum(info["missing_values"].values()),
        )
        return info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_csv_path(self, filename: Optional[str]) -> Path:
        """Xác định đường dẫn file CSV cần đọc.

        Args:
            filename: Tên file hoặc None để tự tìm.

        Returns:
            Path tới file CSV.

        Raises:
            FileNotFoundError: Khi không tìm thấy file.
        """
        if filename is not None:
            path = self.raw_data_path / filename
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file: {path}")
            return path

        # Tự động tìm file CSV đầu tiên
        csv_files = list(self.raw_data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Không có file CSV nào trong: {self.raw_data_path}"
            )
        logger.info("Tự động chọn file: %s", csv_files[0])
        return csv_files[0]

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse cột datetime và đặt làm index.

        Args:
            df: DataFrame gốc.

        Returns:
            DataFrame với DatetimeIndex.
        """
        datetime_col = self.COLUMNS["datetime"]
        if datetime_col not in df.columns:
            logger.warning("Không tìm thấy cột '%s', bỏ qua parse datetime", datetime_col)
            return df

        # Thử parse với nhiều format khác nhau
        df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
        df = df.set_index(datetime_col)
        df = df.sort_index()
        logger.info("Đã parse datetime, index từ %s đến %s", df.index.min(), df.index.max())
        return df
