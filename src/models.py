"""
Model definitions for Bank Loan Default Prediction:
1. Baseline Logistic Regression (Interpretable linear log-odds & calibrated probability)
2. Baseline Decision Tree (Non-linear branching rules & risk segmentation)
3. Tree-Embedding Logistic Hybrid (Tree-discovered non-linear segment embeddings + regularized Logistic Regression)
4. Stacking / Ensemble Hybrid (Blended probability meta-ensemble)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    LOGISTIC_MODEL_PATH, DECISION_TREE_PATH, HYBRID_MODEL_PATH,
    STACKING_MODEL_PATH, RANDOM_SEED
)


class TreeEmbeddingLogisticHybrid(BaseEstimator, ClassifierMixin):
    """
    Hybrid Credit Scoring Model combining Decision Tree segmentation
    with regularized Logistic Regression scoring.
    
    Architecture:
    1. Decision Tree learns non-linear risk interaction splits and assigns applicants to leaf nodes.
    2. Leaf node indices are transformed into one-hot encoded interaction features.
    3. The original features and leaf embeddings are fused and passed to Logistic Regression.
    """
    def __init__(
        self,
        tree_max_depth: int = 5,
        tree_min_samples_leaf: int = 15,
        tree_min_samples_split: int = 25,
        lr_c: float = 0.5,
        lr_penalty: str = "l2",
        class_weight: str = "balanced",
        random_state: int = RANDOM_SEED
    ):
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.tree_min_samples_split = tree_min_samples_split
        self.lr_c = lr_c
        self.lr_penalty = lr_penalty
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.tree_max_depth,
            min_samples_leaf=self.tree_min_samples_leaf,
            min_samples_split=self.tree_min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.leaf_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.logistic_ = LogisticRegression(
            C=self.lr_c,
            solver="lbfgs",
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=self.random_state
        )
        self.classes_ = np.array([0, 1])

    def _extract_leaf_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        # Apply decision tree to get leaf node indices
        leaf_indices = self.tree_.apply(X).reshape(-1, 1)
        if fit:
            return self.leaf_encoder_.fit_transform(leaf_indices)
        return self.leaf_encoder_.transform(leaf_indices)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1. Fit Decision Tree
        self.tree_.fit(X, y)
        
        # 2. Extract and encode leaf interactions
        leaf_features = self._extract_leaf_features(X, fit=True)
        
        # 3. Augment original features with tree structural embeddings
        X_augmented = np.hstack([X, leaf_features])
        
        # 4. Fit regularized Logistic Regression on augmented space
        self.logistic_.fit(X_augmented, y)
        self.classes_ = self.logistic_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        leaf_features = self._extract_leaf_features(X, fit=False)
        X_augmented = np.hstack([X, leaf_features])
        return self.logistic_.predict_proba(X_augmented)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_leaf_index(self, X: np.ndarray) -> np.ndarray:
        return self.tree_.apply(X)


def build_baseline_logistic(random_state: int = RANDOM_SEED) -> LogisticRegression:
    """Builds standard baseline Logistic Regression with class-balanced weighting."""
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state
    )


def build_baseline_decision_tree(random_state: int = RANDOM_SEED) -> DecisionTreeClassifier:
    """Builds transparent baseline Decision Tree with controlled depth."""
    return DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=25,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=random_state
    )


def build_stacking_hybrid(random_state: int = RANDOM_SEED) -> StackingClassifier:
    """Builds Stacking Classifier ensemble combining LR and DT with a meta Logistic Regression."""
    estimators = [
        ("logistic", build_baseline_logistic(random_state)),
        ("tree", build_baseline_decision_tree(random_state))
    ]
    final_estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_state)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method="predict_proba"
    )


def train_and_save_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    """
    Trains all candidate models (Logistic Regression, Decision Tree,
    Tree-Embedding Hybrid, Stacking Ensemble) and persists them to disk.
    """
    LOGISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print("Training Baseline 1: Logistic Regression...")
    lr_model = build_baseline_logistic()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, LOGISTIC_MODEL_PATH)
    
    print("Training Baseline 2: Decision Tree...")
    dt_model = build_baseline_decision_tree()
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, DECISION_TREE_PATH)
    
    print("Training Proposed: Tree-Embedding Logistic Hybrid Model...")
    hybrid_model = TreeEmbeddingLogisticHybrid()
    hybrid_model.fit(X_train, y_train)
    joblib.dump(hybrid_model, HYBRID_MODEL_PATH)
    
    print("Training Reference: Stacking Ensemble...")
    stacking_model = build_stacking_hybrid()
    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, STACKING_MODEL_PATH)
    
    print("All models trained and persisted successfully!")
    return {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
        "Hybrid Model": hybrid_model,
        "Stacking Hybrid": stacking_model
    }


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline
    from config import TARGET_COLUMN
    
    train_df, test_df, preprocessor = run_preprocessing_pipeline()
    X_train = preprocessor.transform(train_df)
    y_train = train_df[TARGET_COLUMN].values
    
    train_and_save_all_models(X_train, y_train)
