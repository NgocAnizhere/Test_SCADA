"""Data Ingestion Layer - tải và lưu trữ dữ liệu từ Kaggle."""

from .kaggle_downloader import KaggleDownloader

__all__ = ["KaggleDownloader"]
