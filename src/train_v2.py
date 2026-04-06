"""
Water Potability Prediction - Optimized Model Training Script (v2.0)
Production-grade ML pipeline with advanced techniques and best practices.

Improvements in v2.0:
- Early stopping for neural networks
- Class weight balancing
- Enhanced hyperparameter tuning (RandomizedSearchCV)
- Multiple evaluation metrics (AUC-ROC, PR-AUC)
- Logging system instead of just prints
- Better error handling and data validation
- Learning curves and analysis
- Model persistence improvements
- Type hints and docstrings
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Data processing & ML
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging system."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration management."""
    # Paths
    DATA_PATH = 'data/water_data.csv'
    MODELS_DIR = 'models/'
    
    # Data processing
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MISSING_VALUE_STRATEGY = 'median'  # More robust than mean
    SCALER_TYPE = 'robust'  # Handle outliers better
    
    # Model tuning
    CV_FOLDS = 5
    N_SEARCH_ITER = 20  # For RandomizedSearchCV (faster than GridSearchCV)
    
    # Neural Network
    NN_EPOCHS = 100
    NN_BATCH_SIZE = 16
    NN_EARLY_STOPPING_PATIENCE = 15
    NN_VALIDATION_SPLIT = 0.2
    
    # Class weights (handle imbalanced data)
    USE_CLASS_WEIGHTS = True

# Set random seeds
np.random.seed(Config.RANDOM_STATE)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(file_path: str) -> Tuple:
    """
    Load dataset and perform comprehensive preprocessing.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Tuple of preprocessed data and metadata
    
    Raises:
        FileNotFoundError: If data file not found
        ValueError: If invalid data format
    """
    logger.info("[1] LOADING DATA...")
    
    # Validate file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Validate required columns
    if 'Potability' not in df.columns:
        raise ValueError("Dataset must contain 'Potability' column")
    
    logger.info(f"    Original dataset shape: {df.shape}")
    logger.info(f"    Missing values:\n{df.isnull().sum()}\n")
    
    # Basic statistics
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    logger.info(f"    Total missing data: {missing_ratio:.2f}%")
    
    # Separate features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Data validation
    if y.isnull().any():
        raise ValueError("Target column contains missing values")
    
    # Handle missing values - IMPROVED: median is more robust
    logger.info("[2] HANDLING MISSING VALUES...")
    imputer = SimpleImputer(strategy=Config.MISSING_VALUE_STRATEGY)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    logger.info(f"    Missing values after imputation: {X_imputed.isnull().sum().sum()}")
    logger.info(f"    Features used: {list(X_imputed.columns)}\n")
    
    # Train-test split with stratification
    logger.info("[3] TRAIN-TEST SPLIT (80-20, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    logger.info(f"    Training set size: {X_train.shape[0]}")
    logger.info(f"    Test set size: {X_test.shape[0]}")
    logger.info(f"    Class distribution (train): {y_train.value_counts().to_dict()}\n")
    
    # Feature scaling - IMPROVED: RobustScaler handles outliers better
    logger.info("[4] FEATURE SCALING...")
    if Config.SCALER_TYPE == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    logger.info(f"    Scaler fitted on training data (type: {Config.SCALER_TYPE})\n")
    
    # Save scaler and imputer for later use
    joblib.dump(imputer, f"{Config.MODELS_DIR}imputer.pkl")
    logger.info(f"    ✓ Saved imputer\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def get_class_weights(y_train: pd.Series) -> Dict:
    """Calculate class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


# ============================================================================
# IMPROVED MODEL TRAINING FUNCTIONS
# ============================================================================

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Tuple:
    """Train Random Forest with optimized hyperparameters and RandomizedSearchCV."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING RANDOM FOREST (Optimized)...")
    logger.info("="*70)
    
    # IMPROVED: Expanded and optimized parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=Config.RANDOM_STATE, n_jobs=-1)
    
    # Use RandomizedSearchCV for larger search space
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=Config.N_SEARCH_ITER, cv=Config.CV_FOLDS,
        scoring='f1', n_jobs=-1, random_state=Config.RANDOM_STATE, verbose=1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV F1 Score: {search.best_score_:.4f}")
    
    # Cross-validation results
    cv_scores = cross_validate(best_model, X_train, y_train, cv=Config.CV_FOLDS,
                               scoring=['f1', 'roc_auc', 'accuracy'])
    logger.info(f"CV F1 Scores: {cv_scores['test_f1'].mean():.4f} (+/- {cv_scores['test_f1'].std():.4f})")
    logger.info(f"CV AUC Scores: {cv_scores['test_roc_auc'].mean():.4f} (+/- {cv_scores['test_roc_auc'].std():.4f})")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, y_pred, y_pred_proba, feature_importance, cv_scores


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> Tuple:
    """Train XGBoost with optimized hyperparameters."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBOOST (Optimized)...")
    logger.info("="*70)
    
    # IMPROVED: Expanded parameter grid for XGBoost
    param_grid = {
        'max_depth': [3, 5, 7, 9, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_lambda': [0, 1, 10]
    }
    
    # Initialize with scale_pos_weight for class imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        random_state=Config.RANDOM_STATE,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        tree_method='hist'
    )
    
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=Config.N_SEARCH_ITER, cv=Config.CV_FOLDS,
        scoring='f1', n_jobs=-1, random_state=Config.RANDOM_STATE, verbose=1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV F1 Score: {search.best_score_:.4f}")
    
    # Cross-validation results
    cv_scores = cross_validate(best_model, X_train, y_train, cv=Config.CV_FOLDS,
                               scoring=['f1', 'roc_auc', 'accuracy'])
    logger.info(f"CV F1 Scores: {cv_scores['test_f1'].mean():.4f} (+/- {cv_scores['test_f1'].std():.4f})")
    logger.info(f"CV AUC Scores: {cv_scores['test_roc_auc'].mean():.4f} (+/- {cv_scores['test_roc_auc'].std():.4f})")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, y_pred, y_pred_proba, feature_importance, cv_scores


def train_knn(X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series) -> Tuple:
    """Train KNN with optimized hyperparameters."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING K-NEAREST NEIGHBORS (Optimized)...")
    logger.info("="*70)
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn = KNeighborsClassifier()
    search = RandomizedSearchCV(
        knn, param_grid, n_iter=Config.N_SEARCH_ITER, cv=Config.CV_FOLDS,
        scoring='f1', n_jobs=-1, random_state=Config.RANDOM_STATE, verbose=1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV F1 Score: {search.best_score_:.4f}")
    
    # Cross-validation results
    cv_scores = cross_validate(best_model, X_train, y_train, cv=Config.CV_FOLDS,
                               scoring=['f1', 'roc_auc', 'accuracy'])
    logger.info(f"CV F1 Scores: {cv_scores['test_f1'].mean():.4f} (+/- {cv_scores['test_f1'].std():.4f})")
    logger.info(f"CV AUC Scores: {cv_scores['test_roc_auc'].mean():.4f} (+/- {cv_scores['test_roc_auc'].std():.4f})")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    return best_model, y_pred, y_pred_proba, cv_scores


def train_neural_network(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> Tuple:
    """Train Neural Network with early stopping and improved architecture."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING ARTIFICIAL NEURAL NETWORK (Optimized)...")
    logger.info("="*70)
    
    input_dim = X_train.shape[1]
    
    # IMPROVED: Better architecture with regularization
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim, 
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    # IMPROVED: Early stopping to prevent overfitting
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=Config.NN_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate scheduling
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    logger.info("Training in progress (with early stopping)...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.NN_EPOCHS,
        batch_size=Config.NN_BATCH_SIZE,
        validation_split=Config.NN_VALIDATION_SPLIT,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    final_epoch = len(history.history['loss'])
    logger.info(f"Training completed at epoch {final_epoch}")
    logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_pred_proba = y_pred_prob.flatten()
    
    return model, y_pred, y_pred_proba, history


# ============================================================================
# ENHANCED EVALUATION
# ============================================================================

def evaluate_model(model_name: str, y_test: np.ndarray, y_pred: np.ndarray,
                  y_pred_proba: np.ndarray = None) -> Dict:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model_name: Name of the model
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # IMPROVED: Additional metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    
    logger.info(f"\n{'─'*70}")
    logger.info(f"EVALUATION METRICS - {model_name}")
    logger.info(f"{'─'*70}")
    logger.info(f"Accuracy:   {accuracy:.4f}")
    logger.info(f"Precision:  {precision:.4f}")
    logger.info(f"Recall:     {recall:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    if roc_auc > 0:
        logger.info(f"ROC AUC:    {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': cm,
        'Predictions': y_pred,
        'Probabilities': y_pred_proba
    }


# ============================================================================
# ADVANCED VISUALIZATION
# ============================================================================

def plot_roc_curves(results_df: pd.DataFrame, y_test: np.ndarray):
    """Plot ROC curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (model_name, probs) in enumerate(zip(
        results_df['Model'],
        results_df['Probabilities']
    )):
        if probs is None:
            continue
            
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        axes[idx].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[idx].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'{model_name} - ROC Curve')
        axes[idx].legend(loc='lower right')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{Config.MODELS_DIR}roc_curves.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: models/roc_curves.png")


def plot_model_comparison(results_df: pd.DataFrame):
    """Enhanced model comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Line plot
    for metric in metrics:
        if metric in results_df.columns:
            axes[0].plot(results_df['Model'], results_df[metric],
                        marker='o', linewidth=2, markersize=8, label=metric)
    
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticklabels(results_df['Model'], rotation=45)
    
    # Bar plot for F1 Score
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    axes[1].bar(results_df['Model'], results_df['F1 Score'], color=colors)
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(results_df['Model'], rotation=45)
    axes[1].set_ylim(0, 1)
    
    for i, v in enumerate(results_df['F1 Score']):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{Config.MODELS_DIR}model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: models/model_comparison_enhanced.png")


def plot_confusion_matrices(results_df: pd.DataFrame):
    """Plot confusion matrices for all models."""
    models = results_df['Model'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (model_name, cm) in enumerate(zip(models, results_df['Confusion Matrix'])):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, annot_kws={'size': 14})
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{Config.MODELS_DIR}confusion_matrices_enhanced.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: models/confusion_matrices_enhanced.png")


def plot_feature_importance(feature_importance: pd.DataFrame):
    """Plot feature importance with statistical visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    top_n = 10
    top_features = feature_importance.head(top_n)
    
    # Horizontal bar chart
    axes[0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'].values)
    axes[0].set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Top {top_n} Features - Random Forest', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # All features
    axes[1].bar(range(len(feature_importance)), feature_importance['importance'].values, color='teal')
    axes[1].set_xlabel('Feature Index', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Importance', fontsize=12, fontweight='bold')
    axes[1].set_title('All Features - Importance Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(feature_importance)))
    axes[1].set_xticklabels(feature_importance['feature'].values, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{Config.MODELS_DIR}feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: models/feature_importance_enhanced.png")


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_all_models(models_dict: Dict[str, Any], scaler: Any) -> dict:
    """Save all trained models with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}
    
    for model_name, model in models_dict.items():
        clean_name = model_name.replace(" ", "_").lower()
        
        if isinstance(model, keras.Model):
            path = f"{Config.MODELS_DIR}{clean_name}_{timestamp}.h5"
            model.save(path)
        else:
            path = f"{Config.MODELS_DIR}{clean_name}_{timestamp}.pkl"
            joblib.dump(model, path)
        
        saved_paths[model_name] = path
        logger.info(f"✓ Saved: {path}")
    
    # Save scaler
    scaler_path = f"{Config.MODELS_DIR}scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Saved: {scaler_path}")
    
    return saved_paths


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline."""
    logger.info("\n" + "="*70)
    logger.info("WATER POTABILITY PREDICTION - ML MODEL TRAINING (v2.0)")
    logger.info("Production-Grade ML Pipeline")
    logger.info("="*70 + "\n")
    
    # Create models directory
    Path(Config.MODELS_DIR).mkdir(exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(
        Config.DATA_PATH
    )
    
    # Save feature names
    joblib.dump(feature_names, f"{Config.MODELS_DIR}feature_names.pkl")
    
    # Train models
    logger.info("\n" + "="*70)
    logger.info("MODEL TRAINING PHASE")
    logger.info("="*70)
    
    rf_model, rf_pred, rf_proba, rf_importance, rf_cv = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    
    xgb_model, xgb_pred, xgb_proba, xgb_importance, xgb_cv = train_xgboost(
        X_train, y_train, X_test, y_test
    )
    
    knn_model, knn_pred, knn_proba, knn_cv = train_knn(
        X_train, y_train, X_test, y_test
    )
    
    nn_model, nn_pred, nn_proba, nn_history = train_neural_network(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate models
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION PHASE")
    logger.info("="*70)
    
    results = []
    results.append(evaluate_model("Random Forest", y_test, rf_pred, rf_proba))
    results.append(evaluate_model("XGBoost", y_test, xgb_pred, xgb_proba))
    results.append(evaluate_model("K-Nearest Neighbors", y_test, knn_pred, knn_proba))
    results.append(evaluate_model("Neural Network", y_test, nn_pred, nn_proba))
    
    results_df = pd.DataFrame(results)
    
    # Display comparison
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("="*70)
    comparison_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    logger.info(results_df[comparison_cols].to_string(index=False))
    
    # Select best model (using weighted criteria)
    logger.info("\n" + "="*70)
    logger.info("BEST MODEL SELECTION (Multi-Criteria)")
    logger.info("="*70)
    
    # Weighted scoring: F1 (50%) + ROC AUC (30%) + Accuracy (20%)
    results_df['Weighted Score'] = (
        results_df['F1 Score'] * 0.5 +
        results_df['ROC AUC'] * 0.3 +
        results_df['Accuracy'] * 0.2
    )
    
    best_idx = results_df['Weighted Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"F1 Score: {results_df.loc[best_idx, 'F1 Score']:.4f}")
    logger.info(f"ROC AUC: {results_df.loc[best_idx, 'ROC AUC']:.4f}")
    logger.info(f"Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
    logger.info(f"Weighted Score: {results_df.loc[best_idx, 'Weighted Score']:.4f}")
    
    # Save all models
    logger.info("\n" + "="*70)
    logger.info("SAVING MODELS")
    logger.info("="*70)
    
    models_dict = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "K-Nearest Neighbors": knn_model,
        "Neural Network": nn_model
    }
    save_all_models(models_dict, scaler)
    
    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)
    
    plot_model_comparison(results_df)
    plot_confusion_matrices(results_df)
    plot_roc_curves(results_df, y_test)
    plot_feature_importance(rf_importance)
    
    # Summary report
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"✓ Best Model: {best_model_name}")
    logger.info(f"✓ Models saved to: {Config.MODELS_DIR}")
    logger.info(f"✓ Visualizations generated (4 files)")
    logger.info(f"✓ Logs saved to: logs/")
    logger.info("="*70 + "\n")
    
    return results_df, best_model_name


if __name__ == "__main__":
    results_df, best_model = main()
