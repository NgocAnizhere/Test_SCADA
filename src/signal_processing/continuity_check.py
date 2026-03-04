"""
Continuity Checker - kiểm tra tính liên tục của dữ liệu SCADA.

Phát hiện mốc thời gian thiếu, khoảng trống dữ liệu, timestamp
trùng lặp và tần số lấy mẫu không đều.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ContinuityChecker:
    """Kiểm tra tính liên tục dữ liệu SCADA.

    Attributes:
        expected_freq (str): Tần số lấy mẫu mong đợi (pandas freq string).
    """

    def __init__(self, expected_freq: str = "10min") -> None:
        """Khởi tạo ContinuityChecker.

        Args:
            expected_freq: Tần số lấy mẫu mong đợi theo pandas (vd: '10min').
        """
        self.expected_freq = expected_freq
        logger.debug("ContinuityChecker khởi tạo, expected_freq=%s", expected_freq)

    # ------------------------------------------------------------------
    # Missing timestamps
    # ------------------------------------------------------------------

    def check_missing_timestamps(
        self,
        df: pd.DataFrame,
        expected_freq: Optional[str] = None,
    ) -> Dict:
        """Phát hiện mốc thời gian bị thiếu.

        Args:
            df: DataFrame với DatetimeIndex.
            expected_freq: Tần số lấy mẫu. Dùng self.expected_freq nếu None.

        Returns:
            Dict với 'missing_count', 'missing_timestamps', 'missing_pct'.
        """
        expected_freq = expected_freq or self.expected_freq
        self._validate_datetime_index(df)

        # Tạo dãy thời gian đầy đủ
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=expected_freq,
        )

        missing = full_index.difference(df.index)
        missing_pct = len(missing) / len(full_index) * 100 if len(full_index) > 0 else 0.0

        logger.info(
            "Missing timestamps: %d/%d (%.2f%%)",
            len(missing), len(full_index), missing_pct,
        )
        return {
            "missing_count": len(missing),
            "missing_timestamps": missing.tolist(),
            "total_expected": len(full_index),
            "missing_pct": round(missing_pct, 4),
        }

    # ------------------------------------------------------------------
    # Data gaps
    # ------------------------------------------------------------------

    def check_data_gaps(
        self,
        df: pd.DataFrame,
        threshold_minutes: float = 20.0,
    ) -> Dict:
        """Phát hiện khoảng trống dữ liệu lớn hơn ngưỡng.

        Args:
            df: DataFrame với DatetimeIndex.
            threshold_minutes: Ngưỡng tối thiểu (phút) để coi là gap.

        Returns:
            Dict với 'gap_count', 'gaps' (list dict: start, end, duration_min).
        """
        self._validate_datetime_index(df)

        diffs = df.index.to_series().diff().dropna()
        threshold_td = pd.Timedelta(minutes=threshold_minutes)

        gaps = []
        for idx, gap in diffs[diffs > threshold_td].items():
            gap_start = idx - gap
            gaps.append({
                "start": str(gap_start),
                "end": str(idx),
                "duration_minutes": round(gap.total_seconds() / 60, 2),
            })

        logger.info(
            "Data gaps (>%.0f min): %d gaps tìm thấy", threshold_minutes, len(gaps)
        )
        return {
            "gap_count": len(gaps),
            "gaps": gaps,
            "threshold_minutes": threshold_minutes,
        }

    # ------------------------------------------------------------------
    # Duplicate timestamps
    # ------------------------------------------------------------------

    def check_duplicate_timestamps(self, df: pd.DataFrame) -> Dict:
        """Phát hiện timestamp trùng lặp.

        Args:
            df: DataFrame với DatetimeIndex.

        Returns:
            Dict với 'duplicate_count', 'duplicate_timestamps'.
        """
        self._validate_datetime_index(df)

        duplicates = df.index[df.index.duplicated(keep=False)]
        unique_dups = df.index[df.index.duplicated(keep="first")]

        logger.info("Duplicate timestamps: %d", len(unique_dups))
        return {
            "duplicate_count": len(unique_dups),
            "duplicate_timestamps": unique_dups.tolist(),
            "total_duplicate_rows": len(duplicates),
        }

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate_missing(
        self,
        df: pd.DataFrame,
        method: str = "linear",
        expected_freq: Optional[str] = None,
    ) -> pd.DataFrame:
        """Nội suy dữ liệu thiếu.

        Args:
            df: DataFrame với DatetimeIndex.
            method: Phương pháp nội suy: 'linear', 'cubic', 'ffill'.
            expected_freq: Tần số lấy mẫu. Dùng self.expected_freq nếu None.

        Returns:
            DataFrame đã được nội suy.
        """
        self._validate_datetime_index(df)
        expected_freq = expected_freq or self.expected_freq

        # Reindex theo dãy thời gian đầy đủ
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=expected_freq,
        )
        df_reindexed = df.reindex(full_index)

        if method == "ffill":
            df_filled = df_reindexed.ffill().bfill()
        elif method in ("linear", "cubic", "spline"):
            df_filled = df_reindexed.interpolate(method=method, limit_direction="both")
        else:
            df_filled = df_reindexed.interpolate(method="linear", limit_direction="both")

        logger.info(
            "Interpolate (%s): %d -> %d hàng",
            method, len(df), len(df_filled),
        )
        return df_filled

    # ------------------------------------------------------------------
    # Sampling rate consistency
    # ------------------------------------------------------------------

    def check_sampling_rate_consistency(self, df: pd.DataFrame) -> Dict:
        """Kiểm tra tần số lấy mẫu có đều không.

        Args:
            df: DataFrame với DatetimeIndex.

        Returns:
            Dict với 'is_consistent', 'mode_interval_min', 'std_interval_min',
            'irregular_count'.
        """
        self._validate_datetime_index(df)

        diffs_minutes = (
            df.index.to_series().diff().dropna().dt.total_seconds() / 60.0
        )
        mode_val = diffs_minutes.mode()[0] if len(diffs_minutes) > 0 else 0
        std_val = diffs_minutes.std()
        irregular = diffs_minutes[np.abs(diffs_minutes - mode_val) > 1.0]

        is_consistent = len(irregular) == 0

        logger.info(
            "Sampling rate: mode=%.2f min, std=%.2f min, irregular=%d",
            mode_val, std_val, len(irregular),
        )
        return {
            "is_consistent": is_consistent,
            "mode_interval_min": round(float(mode_val), 4),
            "std_interval_min": round(float(std_val), 4),
            "irregular_count": len(irregular),
            "total_intervals": len(diffs_minutes),
        }

    # ------------------------------------------------------------------
    # Continuity report
    # ------------------------------------------------------------------

    def generate_continuity_report(
        self,
        df: pd.DataFrame,
        expected_freq: Optional[str] = None,
    ) -> Dict:
        """Báo cáo tổng hợp tính liên tục dữ liệu.

        Args:
            df: DataFrame với DatetimeIndex.
            expected_freq: Tần số lấy mẫu mong đợi.

        Returns:
            Dict tổng hợp: missing_count, gap_count, duplicate_count,
            continuity_score (0-100%).
        """
        expected_freq = expected_freq or self.expected_freq

        missing_info = self.check_missing_timestamps(df, expected_freq)
        gap_info = self.check_data_gaps(df)
        dup_info = self.check_duplicate_timestamps(df)
        sampling_info = self.check_sampling_rate_consistency(df)

        # Tính continuity score (0-100%)
        total_expected = missing_info["total_expected"]
        if total_expected > 0:
            present_pct = (total_expected - missing_info["missing_count"]) / total_expected
        else:
            present_pct = 1.0

        # Penalize cho gaps và duplicates
        gap_penalty = min(gap_info["gap_count"] * 0.01, 0.2)
        dup_penalty = min(dup_info["duplicate_count"] / max(len(df), 1) * 0.5, 0.1)
        sampling_penalty = 0.05 if not sampling_info["is_consistent"] else 0.0

        continuity_score = max(
            0.0,
            (present_pct - gap_penalty - dup_penalty - sampling_penalty) * 100
        )

        report = {
            "missing_count": missing_info["missing_count"],
            "missing_pct": missing_info["missing_pct"],
            "gap_count": gap_info["gap_count"],
            "duplicate_count": dup_info["duplicate_count"],
            "is_sampling_consistent": sampling_info["is_consistent"],
            "mode_interval_min": sampling_info["mode_interval_min"],
            "continuity_score": round(continuity_score, 2),
            "total_records": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            },
        }

        logger.info(
            "Continuity report: score=%.2f%%, missing=%d, gaps=%d, dups=%d",
            continuity_score,
            missing_info["missing_count"],
            gap_info["gap_count"],
            dup_info["duplicate_count"],
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_datetime_index(df: pd.DataFrame) -> None:
        """Kiểm tra DataFrame có DatetimeIndex không.

        Args:
            df: DataFrame cần kiểm tra.

        Raises:
            ValueError: Khi index không phải DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame phải có DatetimeIndex. "
                "Hãy set_index() với cột datetime trước."
            )
