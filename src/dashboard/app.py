"""
Streamlit Dashboard - Wind Turbine SCADA Fault Detection System.

Tabs:
    1. 📊 Data Overview
    2. 🔧 Signal Processing
    3. 🚨 Anomaly Detection
    4. 🤖 ML Fault Classification
    5. 📈 Reports
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Thêm thư mục gốc vào sys.path để import các module
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.signal_processing.noise_filter import NoiseFilter
from src.signal_processing.frequency_analysis import FrequencyAnalyzer
from src.signal_processing.continuity_check import ContinuityChecker
from src.anomaly_detection.detector import AnomalyDetector

logger = logging.getLogger(__name__)

# ==============================================================================
# Page config
# ==============================================================================
st.set_page_config(
    page_title="Wind Turbine SCADA Fault Detection",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Load config
# ==============================================================================
@st.cache_data
def load_config() -> dict:
    """Tải cấu hình từ config.yaml."""
    config_path = ROOT_DIR / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


CONFIG = load_config()
SIGNAL_COLS = [
    "LV ActivePower (kW)",
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
]

# ==============================================================================
# Helper functions
# ==============================================================================

@st.cache_data
def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Đọc DataFrame từ bytes (uploaded file).

    Args:
        file_bytes: Nội dung file.
        filename: Tên file.

    Returns:
        DataFrame đã parse datetime.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    datetime_col = CONFIG.get("data", {}).get("columns", {}).get("datetime", "Date/Time")
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
        df = df.set_index(datetime_col).sort_index()
    return df


def load_local_data() -> Optional[pd.DataFrame]:
    """Tải dữ liệu từ data/raw/ nếu có.

    Returns:
        DataFrame hoặc None nếu không có file.
    """
    raw_path = ROOT_DIR / "data" / "raw"
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        return None

    df = pd.read_csv(csv_files[0])
    datetime_col = CONFIG.get("data", {}).get("columns", {}).get("datetime", "Date/Time")
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
        df = df.set_index(datetime_col).sort_index()
    return df


def get_numeric_cols(df: pd.DataFrame) -> list:
    """Lấy danh sách cột số trong DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ==============================================================================
# Sidebar
# ==============================================================================

st.sidebar.title("💨 SCADA Fault Detection")
st.sidebar.markdown("---")

tab_names = [
    "📊 Data Overview",
    "🔧 Signal Processing",
    "🚨 Anomaly Detection",
    "🤖 ML Fault Classification",
    "📈 Reports",
]
selected_tab = st.sidebar.radio("Navigation", tab_names)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Wind Turbine SCADA**\n"
    "Dataset: noobtube99/wind-turbine-scada-data\n"
    "Sampling: 10 phút"
)

# ==============================================================================
# Data loading section (shared)
# ==============================================================================

st.title("💨 Wind Turbine SCADA Fault Detection System")

# Session state cho DataFrame
if "df" not in st.session_state:
    st.session_state.df = None

with st.expander("📂 Tải dữ liệu", expanded=st.session_state.df is None):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df_loaded = load_dataframe(uploaded_file.read(), uploaded_file.name)
            st.session_state.df = df_loaded
            st.success(f"✅ Đã tải {len(df_loaded):,} hàng từ {uploaded_file.name}")
    with col2:
        if st.button("📁 Load từ data/raw/"):
            df_local = load_local_data()
            if df_local is not None:
                st.session_state.df = df_local
                st.success(f"✅ Đã tải {len(df_local):,} hàng từ data/raw/")
            else:
                st.warning("⚠️ Không tìm thấy file CSV trong data/raw/")

df: Optional[pd.DataFrame] = st.session_state.df

# ==============================================================================
# Tab 1: Data Overview
# ==============================================================================

if selected_tab == "📊 Data Overview":
    st.header("📊 Data Overview")

    if df is None:
        st.info("👆 Vui lòng tải dữ liệu ở trên")
        st.stop()

    # Thống kê mô tả
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số bản ghi", f"{len(df):,}")
    with col2:
        st.metric("Số cột", len(df.columns))
    with col3:
        if isinstance(df.index, pd.DatetimeIndex):
            days = (df.index.max() - df.index.min()).days
            st.metric("Số ngày", days)
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("Giá trị thiếu", f"{missing:,}")

    st.subheader("📋 Thống kê mô tả")
    st.dataframe(df.describe(), use_container_width=True)

    # Time series plot
    st.subheader("📈 Time Series")
    numeric_cols = get_numeric_cols(df)
    if numeric_cols:
        selected_cols = st.multiselect(
            "Chọn cột",
            numeric_cols,
            default=numeric_cols[:min(2, len(numeric_cols))],
        )
        if selected_cols:
            # Downsample nếu quá nhiều điểm
            plot_df = df[selected_cols]
            if len(plot_df) > 5000:
                plot_df = plot_df.iloc[::len(plot_df) // 5000]

            fig = px.line(
                plot_df,
                title="Time Series của tín hiệu SCADA",
                labels={"value": "Giá trị", "index": "Thời gian"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Missing data heatmap
    st.subheader("🔍 Missing Data Heatmap")
    missing_pct = (df[numeric_cols].isnull().resample("D").mean() * 100
                   if isinstance(df.index, pd.DatetimeIndex) and numeric_cols
                   else df[numeric_cols].isnull() * 100)

    if isinstance(missing_pct, pd.DataFrame) and not missing_pct.empty:
        fig_missing = px.imshow(
            missing_pct.T,
            labels={"x": "Ngày", "y": "Cột", "color": "Missing %"},
            title="Tỉ lệ dữ liệu thiếu theo ngày",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            fig_bar = px.bar(
                x=missing_counts.index,
                y=missing_counts.values,
                title="Số giá trị thiếu theo cột",
                labels={"x": "Cột", "y": "Số giá trị thiếu"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("✅ Không có giá trị thiếu")

# ==============================================================================
# Tab 2: Signal Processing
# ==============================================================================

elif selected_tab == "🔧 Signal Processing":
    st.header("🔧 Signal Processing")

    if df is None:
        st.info("👆 Vui lòng tải dữ liệu ở trên")
        st.stop()

    numeric_cols = get_numeric_cols(df)
    if not numeric_cols:
        st.warning("Không có cột số nào trong dữ liệu")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        selected_signal = st.selectbox("Chọn tín hiệu", numeric_cols)
    with col2:
        filter_type = st.selectbox(
            "Chọn bộ lọc",
            ["Butterworth Lowpass", "Wavelet", "Kalman", "Moving Average", "Savitzky-Golay"],
        )

    signal_data = df[selected_signal].dropna().values
    if len(signal_data) > 10000:
        signal_data = signal_data[:10000]

    noise_filter = NoiseFilter()

    # Áp dụng bộ lọc
    with st.spinner("Đang lọc tín hiệu..."):
        if filter_type == "Butterworth Lowpass":
            cutoff = st.slider("Cutoff frequency", 0.0001, 0.01, 0.001, 0.0001)
            filtered = noise_filter.butterworth_lowpass(signal_data, cutoff)
        elif filter_type == "Wavelet":
            wavelet_type = st.selectbox("Wavelet", ["db4", "db8", "sym4", "haar"])
            filtered = noise_filter.wavelet_denoise(signal_data, wavelet=wavelet_type)
        elif filter_type == "Kalman":
            filtered = noise_filter.kalman_filter(signal_data)
        elif filter_type == "Moving Average":
            window = st.slider("Window size", 3, 100, 10)
            filtered = noise_filter.moving_average(signal_data, window)
        else:  # Savitzky-Golay
            win = st.slider("Window length", 5, 51, 11, step=2)
            filtered = noise_filter.savitzky_golay_filter(signal_data, win)

    # So sánh trước/sau
    col1, col2 = st.columns(2)
    with col1:
        fig_before = go.Figure()
        fig_before.add_trace(go.Scatter(y=signal_data[:500], name="Original"))
        fig_before.update_layout(title="Trước lọc", height=300)
        st.plotly_chart(fig_before, use_container_width=True)
    with col2:
        fig_after = go.Figure()
        fig_after.add_trace(go.Scatter(y=filtered[:500], name="Filtered", line=dict(color="red")))
        fig_after.update_layout(title="Sau lọc", height=300)
        st.plotly_chart(fig_after, use_container_width=True)

    # SNR
    snr = noise_filter.compute_snr(signal_data[:len(filtered)], filtered)
    st.metric("SNR (dB)", f"{snr:.2f}")

    # FFT Spectrum
    st.subheader("📡 FFT Spectrum")
    freq_analyzer = FrequencyAnalyzer()
    freqs, mags = freq_analyzer.compute_fft(signal_data)
    fig_fft = px.line(
        x=freqs[:len(freqs) // 2],
        y=mags[:len(mags) // 2],
        labels={"x": "Frequency (Hz)", "y": "Magnitude"},
        title=f"FFT Spectrum - {selected_signal}",
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    # Spectrogram
    st.subheader("🌈 Spectrogram")
    try:
        f_spec, t_spec, sxx = freq_analyzer.compute_spectrogram(signal_data)
        fig_spec = px.imshow(
            10 * np.log10(sxx + 1e-12),
            x=t_spec,
            y=f_spec,
            labels={"x": "Time", "y": "Frequency (Hz)", "color": "Power (dB)"},
            title="Spectrogram",
            origin="lower",
            aspect="auto",
        )
        st.plotly_chart(fig_spec, use_container_width=True)
    except Exception as e:
        st.warning(f"Không thể tính spectrogram: {e}")

# ==============================================================================
# Tab 3: Anomaly Detection
# ==============================================================================

elif selected_tab == "🚨 Anomaly Detection":
    st.header("🚨 Anomaly Detection")

    if df is None:
        st.info("👆 Vui lòng tải dữ liệu ở trên")
        st.stop()

    numeric_cols = get_numeric_cols(df)
    selected_cols = st.multiselect(
        "Chọn cột để phát hiện bất thường",
        numeric_cols,
        default=numeric_cols[:min(2, len(numeric_cols))],
    )

    z_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
    detector = AnomalyDetector(z_threshold=z_threshold)

    if st.button("🔍 Chạy Anomaly Detection"):
        with st.spinner("Đang phát hiện bất thường..."):
            report_df = detector.generate_anomaly_report(df, signal_cols=selected_cols)
            st.session_state.anomaly_report = report_df

    if "anomaly_report" in st.session_state:
        report_df = st.session_state.anomaly_report

        if report_df.empty:
            st.success("✅ Không phát hiện bất thường nào")
        else:
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng bất thường", len(report_df))
            with col2:
                high_count = (report_df["severity"] == "high").sum()
                st.metric("🔴 Nghiêm trọng cao", high_count)
            with col3:
                types = report_df["type"].nunique()
                st.metric("Số loại bất thường", types)

            # Timeline plot
            if isinstance(df.index, pd.DatetimeIndex) and "timestamp" in report_df.columns:
                st.subheader("📅 Anomaly Timeline")
                color_map = {"high": "red", "medium": "orange", "low": "green"}
                for col in selected_cols:
                    col_report = report_df[report_df["column"] == col]
                    if col_report.empty:
                        continue

                    fig_timeline = go.Figure()
                    signal_plot = df[col].iloc[:5000] if len(df) > 5000 else df[col]
                    fig_timeline.add_trace(go.Scatter(
                        x=signal_plot.index,
                        y=signal_plot.values,
                        name=col,
                        line=dict(color="blue"),
                    ))

                    for severity, color in color_map.items():
                        sv_pts = col_report[col_report["severity"] == severity]
                        if not sv_pts.empty:
                            fig_timeline.add_trace(go.Scatter(
                                x=sv_pts["timestamp"],
                                y=sv_pts["value"],
                                mode="markers",
                                name=f"Anomaly ({severity})",
                                marker=dict(color=color, size=8, symbol="x"),
                            ))

                    fig_timeline.update_layout(title=f"Anomalies - {col}", height=350)
                    st.plotly_chart(fig_timeline, use_container_width=True)

            # Bảng anomalies với filter
            st.subheader("📋 Bảng Anomalies")
            filter_type_sel = st.multiselect(
                "Lọc theo type",
                report_df["type"].unique().tolist(),
                default=report_df["type"].unique().tolist(),
            )
            filter_sev = st.multiselect(
                "Lọc theo severity",
                ["high", "medium", "low"],
                default=["high", "medium", "low"],
            )
            filtered_report = report_df[
                report_df["type"].isin(filter_type_sel)
                & report_df["severity"].isin(filter_sev)
            ]
            st.dataframe(filtered_report, use_container_width=True)

            # Pie chart phân phối
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    report_df,
                    names="type",
                    title="Phân phối loại bất thường",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_sev = px.pie(
                    report_df,
                    names="severity",
                    title="Phân phối mức độ nghiêm trọng",
                    color_discrete_map={"high": "red", "medium": "orange", "low": "green"},
                )
                st.plotly_chart(fig_sev, use_container_width=True)

# ==============================================================================
# Tab 4: ML Fault Classification
# ==============================================================================

elif selected_tab == "🤖 ML Fault Classification":
    st.header("🤖 ML Fault Classification")

    if df is None:
        st.info("👆 Vui lòng tải dữ liệu ở trên")
        st.stop()

    st.info(
        "Module này cần dữ liệu đã được gán nhãn lỗi. "
        "Nhãn được tạo tự động từ kết quả anomaly detection."
    )

    FAULT_LABELS = {
        0: "Normal", 1: "Sensor fault", 2: "Electrical fault",
        3: "Mechanical fault", 4: "Wind condition anomaly", 5: "Communication fault",
    }

    if st.button("🏷️ Tạo nhãn tự động từ Anomaly Detection"):
        with st.spinner("Đang tạo nhãn..."):
            detector = AnomalyDetector()
            numeric_cols = get_numeric_cols(df)
            report = detector.generate_anomaly_report(df, signal_cols=numeric_cols[:4])

            # Tạo nhãn đơn giản dựa trên loại bất thường
            labels = np.zeros(len(df), dtype=int)
            if not report.empty:
                for _, row in report.iterrows():
                    try:
                        idx = df.index.get_loc(row["timestamp"])
                        atype = row["type"]
                        if atype == "spike":
                            labels[idx] = 1  # Sensor fault
                        elif atype == "missing":
                            labels[idx] = 5  # Communication fault
                        elif atype == "drift":
                            labels[idx] = 2  # Electrical fault
                        elif atype == "flatline":
                            labels[idx] = 1  # Sensor fault
                        elif atype == "std_deviation":
                            labels[idx] = 3  # Mechanical fault
                    except (KeyError, TypeError):
                        pass

            st.session_state.labels = labels

            label_series = pd.Series(labels)
            label_counts = label_series.value_counts()
            fig_labels = px.bar(
                x=[FAULT_LABELS.get(i, str(i)) for i in label_counts.index],
                y=label_counts.values,
                title="Phân phối nhãn lỗi",
                labels={"x": "Loại lỗi", "y": "Số lượng"},
            )
            st.plotly_chart(fig_labels, use_container_width=True)

    # Model training section
    if "labels" in st.session_state:
        st.subheader("🏋️ Huấn luyện Model")
        model_type = st.selectbox(
            "Chọn model",
            ["Random Forest", "Gradient Boosting"],
        )

        if st.button("🚀 Train Model"):
            with st.spinner(f"Đang train {model_type}..."):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                    from sklearn.metrics import confusion_matrix, classification_report
                    import plotly.figure_factory as ff

                    # Feature engineering đơn giản
                    numeric_cols = get_numeric_cols(df)
                    X = df[numeric_cols[:4]].fillna(0).values
                    y = st.session_state.labels

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                    )

                    if model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
                    else:
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Confusion matrix
                    classes = sorted(np.unique(y))
                    cm = confusion_matrix(y_test, y_pred, labels=classes)
                    class_names = [FAULT_LABELS.get(i, str(i)) for i in classes]

                    fig_cm = px.imshow(
                        cm,
                        x=class_names,
                        y=class_names,
                        title="Confusion Matrix",
                        color_continuous_scale="Blues",
                        text_auto=True,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # Classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)

                    # Feature importance
                    if hasattr(model, "feature_importances_"):
                        importance_df = pd.DataFrame({
                            "Feature": numeric_cols[:4],
                            "Importance": model.feature_importances_,
                        }).sort_values("Importance", ascending=True)
                        fig_imp = px.bar(
                            importance_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title="Feature Importance",
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                    st.session_state.trained_model = model
                    st.success("✅ Train xong!")

                except Exception as e:
                    st.error(f"Lỗi khi train: {e}")

    # Real-time prediction
    if "trained_model" in st.session_state:
        st.subheader("🔮 Real-time Prediction")
        predict_file = st.file_uploader("Upload dữ liệu mới để predict", type=["csv"], key="predict")
        if predict_file:
            df_pred = load_dataframe(predict_file.read(), predict_file.name)
            numeric_cols_pred = df_pred.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols_pred:
                X_pred = df_pred[numeric_cols_pred[:4]].fillna(0).values
                y_pred = st.session_state.trained_model.predict(X_pred)
                df_pred["prediction"] = [FAULT_LABELS.get(p, str(p)) for p in y_pred]
                st.dataframe(df_pred[numeric_cols_pred[:4] + ["prediction"]].head(100), use_container_width=True)

                pred_counts = pd.Series(df_pred["prediction"]).value_counts()
                fig_pred = px.pie(values=pred_counts.values, names=pred_counts.index, title="Kết quả phân loại")
                st.plotly_chart(fig_pred, use_container_width=True)

# ==============================================================================
# Tab 5: Reports
# ==============================================================================

elif selected_tab == "📈 Reports":
    st.header("📈 Reports")

    if df is None:
        st.info("👆 Vui lòng tải dữ liệu ở trên")
        st.stop()

    # Trend analysis
    st.subheader("📆 Trend Analysis theo tháng")
    numeric_cols = get_numeric_cols(df)

    if isinstance(df.index, pd.DatetimeIndex) and numeric_cols:
        monthly_mean = df[numeric_cols].resample("M").mean()
        fig_trend = px.line(
            monthly_mean,
            title="Xu hướng tháng của các tín hiệu",
            labels={"value": "Giá trị trung bình", "index": "Tháng"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Continuity report
    st.subheader("🔗 Báo cáo Continuity")
    if isinstance(df.index, pd.DatetimeIndex):
        checker = ContinuityChecker()
        try:
            report = checker.generate_continuity_report(df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Điểm liên tục", f"{report['continuity_score']:.1f}%")
            col2.metric("Thiếu timestamp", report["missing_count"])
            col3.metric("Khoảng trống", report["gap_count"])
            col4.metric("Trùng lặp", report["duplicate_count"])
        except Exception as e:
            st.warning(f"Không thể tạo báo cáo continuity: {e}")

    # Export Excel
    st.subheader("📤 Export Báo cáo")
    if st.button("📥 Export Excel"):
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.describe().to_excel(writer, sheet_name="Statistics")
                if numeric_cols:
                    df[numeric_cols].head(1000).to_excel(writer, sheet_name="Data Sample")

            buffer.seek(0)
            st.download_button(
                label="💾 Download Excel",
                data=buffer,
                file_name="scada_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Lỗi khi export: {e}")

    # Alert history (placeholder)
    st.subheader("🚨 Alert History")
    if "anomaly_report" in st.session_state and not st.session_state.anomaly_report.empty:
        alert_df = st.session_state.anomaly_report
        st.dataframe(
            alert_df[alert_df["severity"] == "high"].head(50),
            use_container_width=True,
        )
    else:
        st.info("Chưa có lịch sử cảnh báo. Chạy Anomaly Detection trước.")
