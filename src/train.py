"""
Water Potability Prediction - Model Training Script
This script trains, evaluates, and compares multiple ML models for water potability prediction.
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from datetime import datetime

# Data processing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = 'data/water_data.csv'
MODELS_DIR = 'models/'
TEST_SIZE = 0.2
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(file_path):
    """
    Load dataset from CSV and handle missing values.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("[1] LOADING DATA...")
    df = pd.read_csv(file_path)
    print(f"    Original dataset shape: {df.shape}")
    print(f"    Missing values:\n{df.isnull().sum()}\n")
    
    # Separate features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Handle missing values using imputation (mean strategy)
    print("[2] HANDLING MISSING VALUES...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    print(f"    Missing values after imputation: {X_imputed.isnull().sum().sum()}")
    print(f"    Features used: {list(X_imputed.columns)}\n")
    
    # Train-test split
    print("[3] TRAIN-TEST SPLIT (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"    Training set size: {X_train.shape[0]}")
    print(f"    Test set size: {X_test.shape[0]}")
    print(f"    Training set class distribution:\n{y_train.value_counts().to_dict()}\n")
    
    # Feature scaling
    print("[4] FEATURE SCALING...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    print(f"    Scaler fitted on training data\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning."""
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST...")
    print("="*70)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, y_pred, feature_importance


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning."""
    print("\n" + "="*70)
    print("TRAINING XGBOOST...")
    print("="*70)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, y_pred, feature_importance


def train_knn(X_train, y_train, X_test, y_test):
    """Train K-Nearest Neighbors with hyperparameter tuning."""
    print("\n" + "="*70)
    print("TRAINING K-NEAREST NEIGHBORS...")
    print("="*70)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    return best_model, y_pred


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train Artificial Neural Network using TensorFlow/Keras."""
    print("\n" + "="*70)
    print("TRAINING ARTIFICIAL NEURAL NETWORK...")
    print("="*70)
    
    input_dim = X_train.shape[1]
    
    # Build the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Train the model
    print("Training in progress...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Make predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    return model, y_pred


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model_name, y_test, y_pred):
    """
    Compute evaluation metrics for a model.
    
    Args:
        model_name (str): Name of the model
        y_test: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing all metrics
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'─'*70}")
    print(f"EVALUATION METRICS - {model_name}")
    print(f"{'─'*70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_model_comparison(results_df):
    """Plot model accuracy comparison."""
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for metric in metrics:
        plt.plot(
            results_df['Model'],
            results_df[metric],
            marker='o',
            linewidth=2,
            markersize=8,
            label=metric
        )
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: models/model_comparison.png")
    plt.show()


def plot_confusion_matrices(results_df):
    """Plot confusion matrices for all models."""
    models = results_df['Model'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (model_name, cm) in enumerate(zip(models, results_df['Confusion Matrix'])):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            cbar=False, annot_kws={'size': 12}
        )
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: models/confusion_matrices.png")
    plt.show()


def plot_feature_importance(feature_importance):
    """Plot Random Forest feature importance."""
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    
    plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title('Random Forest - Top 10 Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: models/feature_importance.png")
    plt.show()


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model, model_name, scaler=None):
    """Save trained model and scaler to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    if isinstance(model, keras.Model):
        model_path = f"{MODELS_DIR}{model_name}_{timestamp}.h5"
        model.save(model_path)
    else:
        model_path = f"{MODELS_DIR}{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
    
    # Save scaler
    if scaler is not None:
        scaler_path = f"{MODELS_DIR}scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"\n✓ Saved: {scaler_path}")
    
    print(f"✓ Saved: {model_path}")
    return model_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("WATER POTABILITY PREDICTION - ML MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(DATA_PATH)
    
    # Train models
    print("\n" + "="*70)
    print("MODEL TRAINING PHASE")
    print("="*70)
    
    # Random Forest
    rf_model, rf_pred, rf_importance = train_random_forest(X_train, y_train, X_test, y_test)
    
    # XGBoost
    xgb_model, xgb_pred, xgb_importance = train_xgboost(X_train, y_train, X_test, y_test)
    
    # KNN
    knn_model, knn_pred = train_knn(X_train, y_train, X_test, y_test)
    
    # Neural Network
    nn_model, nn_pred = train_neural_network(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    print("\n" + "="*70)
    print("MODEL EVALUATION PHASE")
    print("="*70)
    
    results = []
    results.append(evaluate_model("Random Forest", y_test, rf_pred))
    results.append(evaluate_model("XGBoost", y_test, xgb_pred))
    results.append(evaluate_model("K-Nearest Neighbors", y_test, knn_pred))
    results.append(evaluate_model("Neural Network", y_test, nn_pred))
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].to_string(index=False))
    
    # Identify best model
    best_model_idx = results_df['F1 Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_f1_score = results_df.loc[best_model_idx, 'F1 Score']
    best_accuracy = results_df.loc[best_model_idx, 'Accuracy']
    
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"✓ Model: {best_model_name}")
    print(f"✓ F1 Score: {best_f1_score:.4f}")
    print(f"✓ Accuracy: {best_accuracy:.4f}")
    print(f"\nWhy {best_model_name}?")
    print(f"  • Highest F1 Score: {best_f1_score:.4f}")
    print(f"  • Balanced precision and recall")
    print(f"  • Best overall generalization performance")
    
    # Select best model for saving
    if best_model_name == "Random Forest":
        best_model = rf_model
    elif best_model_name == "XGBoost":
        best_model = xgb_model
    elif best_model_name == "K-Nearest Neighbors":
        best_model = knn_model
    else:
        best_model = nn_model
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    save_model(best_model, best_model_name.replace(" ", "_").lower())
    save_model(scaler, "scaler")
    
    # Save feature names for the app
    joblib.dump(feature_names, f"{MODELS_DIR}feature_names.pkl")
    print(f"✓ Saved: {MODELS_DIR}feature_names.pkl")
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    plot_model_comparison(results_df)
    plot_confusion_matrices(results_df)
    plot_feature_importance(rf_importance)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Models trained and evaluated")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ All visualizations saved to 'models/' directory")
    print(f"✓ Ready to use with Streamlit app!")
    print("="*70 + "\n")
    
    return results_df, best_model_name


if __name__ == "__main__":
    results_df, best_model = main()
