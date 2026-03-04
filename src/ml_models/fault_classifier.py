"""
Fault Classifier - phân loại dạng lỗi tuabin gió.

Hỗ trợ: Random Forest, XGBoost, SVM, Gradient Boosting,
Ensemble Voting, SHAP explanations.

Nhãn lỗi:
    0: Normal operation
    1: Sensor fault (noise spike, flatline)
    2: Electrical fault (power anomaly)
    3: Mechanical fault (vibration anomaly)
    4: Wind condition anomaly
    5: Communication fault (missing data, gaps)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


class FaultClassifier:
    """Phân loại dạng lỗi tuabin gió.

    Attributes:
        fault_labels (Dict): Mapping nhãn số sang tên lỗi.
        models (Dict): Các model đã train.
        random_state (int): Seed ngẫu nhiên.
    """

    FAULT_LABELS: Dict[int, str] = {
        0: "Normal operation",
        1: "Sensor fault",
        2: "Electrical fault",
        3: "Mechanical fault",
        4: "Wind condition anomaly",
        5: "Communication fault",
    }

    def __init__(self, random_state: int = 42) -> None:
        """Khởi tạo FaultClassifier.

        Args:
            random_state: Seed ngẫu nhiên cho reproducibility.
        """
        self.random_state = random_state
        self.models: Dict[str, object] = {}
        logger.debug("FaultClassifier khởi tạo, random_state=%d", random_state)

    # ------------------------------------------------------------------
    # Training methods
    # ------------------------------------------------------------------

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
    ) -> RandomForestClassifier:
        """Huấn luyện Random Forest với GridSearchCV.

        Args:
            X_train: Ma trận đặc trưng train.
            y_train: Nhãn train.
            cv: Số fold cross-validation.

        Returns:
            Model Random Forest đã train tốt nhất.
        """
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced"],
        }
        base = RandomForestClassifier(random_state=self.random_state)
        clf = GridSearchCV(base, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        self.models["random_forest"] = best_model
        logger.info(
            "Random Forest best params: %s, CV score: %.4f",
            clf.best_params_, clf.best_score_,
        )
        return best_model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
    ) -> object:
        """Huấn luyện XGBoost với cross-validation.

        Args:
            X_train: Ma trận đặc trưng train.
            y_train: Nhãn train.
            cv: Số fold cross-validation.

        Returns:
            Model XGBoost đã train.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost chưa cài. Chạy: pip install xgboost")
            return None  # type: ignore[return-value]

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.3],
            "use_label_encoder": [False],
        }
        base = XGBClassifier(
            random_state=self.random_state,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        clf = GridSearchCV(base, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        self.models["xgboost"] = best_model
        logger.info(
            "XGBoost best params: %s, CV score: %.4f",
            clf.best_params_, clf.best_score_,
        )
        return best_model

    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3,
    ) -> SVC:
        """Huấn luyện SVM với RBF kernel.

        Args:
            X_train: Ma trận đặc trưng train.
            y_train: Nhãn train.
            cv: Số fold cross-validation.

        Returns:
            Model SVM đã train.
        """
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced"],
        }
        base = SVC(probability=True, random_state=self.random_state)
        clf = GridSearchCV(base, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        self.models["svm"] = best_model
        logger.info(
            "SVM best params: %s, CV score: %.4f",
            clf.best_params_, clf.best_score_,
        )
        return best_model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
    ) -> GradientBoostingClassifier:
        """Huấn luyện Gradient Boosting.

        Args:
            X_train: Ma trận đặc trưng train.
            y_train: Nhãn train.
            cv: Số fold cross-validation.

        Returns:
            Model Gradient Boosting đã train.
        """
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted")
        clf.fit(X_train, y_train)
        self.models["gradient_boosting"] = clf
        logger.info(
            "Gradient Boosting CV score: %.4f ± %.4f",
            scores.mean(), scores.std(),
        )
        return clf

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Đánh giá model với confusion matrix và ROC-AUC.

        Args:
            model: Model đã train.
            X_test: Ma trận đặc trưng test.
            y_test: Nhãn test.

        Returns:
            Dict với: confusion_matrix, classification_report, roc_auc, accuracy.
        """
        y_pred = model.predict(X_test)  # type: ignore[union-attr]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=[self.FAULT_LABELS.get(i, str(i)) for i in sorted(np.unique(y_test))],
            output_dict=True,
        )

        # ROC-AUC cho multi-class
        try:
            classes = sorted(np.unique(y_test))
            y_bin = label_binarize(y_test, classes=classes)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)  # type: ignore[union-attr]
                if len(classes) == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
            else:
                roc_auc = None
        except Exception as e:
            logger.warning("Không tính được ROC-AUC: %s", e)
            roc_auc = None

        accuracy = report.get("accuracy", 0.0)
        logger.info(
            "Model evaluation: accuracy=%.4f, roc_auc=%s",
            accuracy, f"{roc_auc:.4f}" if roc_auc else "N/A",
        )

        return {
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
        }

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán bằng Voting Ensemble của tất cả models.

        Args:
            X: Ma trận đặc trưng.

        Returns:
            Mảng nhãn dự đoán.
        """
        if not self.models:
            raise RuntimeError("Chưa có model nào được train. Hãy train trước.")

        available_models = [
            (name, model) for name, model in self.models.items()
            if model is not None
        ]
        if len(available_models) == 1:
            return available_models[0][1].predict(X)  # type: ignore[union-attr]

        voting_clf = VotingClassifier(
            estimators=available_models,
            voting="soft" if all(
                hasattr(m, "predict_proba") for _, m in available_models
            ) else "hard",
        )
        # Fit với dữ liệu dummy nếu chưa fit
        # (giả sử các model đã được fit trước)
        # Dùng predict riêng và vote manually
        predictions = np.stack([
            m.predict(X) for _, m in available_models  # type: ignore[union-attr]
        ])

        # Majority vote
        from scipy.stats import mode as scipy_mode
        result = scipy_mode(predictions, axis=0, keepdims=False)
        return result.mode.flatten()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self, model: object, path: Union[str, Path]) -> None:
        """Lưu model với joblib.

        Args:
            model: Model cần lưu.
            path: Đường dẫn file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Model đã lưu: %s", path)

    def load_model(self, path: Union[str, Path]) -> object:
        """Tải model từ file joblib.

        Args:
            path: Đường dẫn file.

        Returns:
            Model đã tải.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {path}")
        model = joblib.load(path)
        logger.info("Model đã tải: %s", path)
        return model

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        model: object,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
    ) -> Optional[object]:
        """Giải thích dự đoán bằng SHAP values.

        Args:
            model: Model đã train (Random Forest hoặc XGBoost).
            X: Dữ liệu cần giải thích.
            feature_names: Tên các đặc trưng.

        Returns:
            SHAP Explainer hoặc None nếu không khả dụng.
        """
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            logger.info("SHAP values đã được tính")
            return shap_values
        except ImportError:
            logger.warning("shap chưa cài. Chạy: pip install shap")
            return None
        except Exception as e:
            logger.warning("Không tính được SHAP: %s", e)
            return None

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        model: object,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Lấy feature importance từ tree-based model.

        Args:
            model: Model đã train.
            feature_names: Tên các đặc trưng.

        Returns:
            DataFrame với cột 'feature' và 'importance', sắp xếp giảm dần.
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model không có feature_importances_")
            return pd.DataFrame()

        importances = model.feature_importances_  # type: ignore[union-attr]
        df = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        logger.info("Feature importance: top=%s (%.4f)", df.iloc[0]["feature"], df.iloc[0]["importance"])
        return df
