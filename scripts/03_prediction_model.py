import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer
import os
import joblib
import mlflow
import mlflow.xgboost

# --- CONFIGURATION ---
INPUT_PATH = os.path.join("data", "processed", "training_data_with_clusters.csv")
MODEL_DIR = "models"
CHURN_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model_optimized.json")
MLFLOW_EXPERIMENT = "Uber_Driver_Churn_Prediction"

def run_optimization_and_training():
    print("--- Starting Full Optimization & Training Pipeline ---")

    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}. Run script 03 first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Data Loaded. Shape: {df.shape}")

    # 2. Feature Engineering & Split
    df['cluster_label'] = df['cluster_label'].astype('category')
    features = [
        'avg_earnings_per_hour_online', 'trip_utilization_rate', 'surge_reliance_score',
        'premium_trip_ratio', 'quest_completion_rate', 'cancellation_rate',
        'acceptance_rate', 'pro_tier_status', 'cluster_label'
    ]
    
    X = df[features]
    y = df['Churned']

    # 80% for training/tuning, 20% for final, unbiased evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate Class Imbalance Ratio
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Class Imbalance Ratio (scale_pos_weight): {ratio:.2f}")

    # --- PART A: HYPERPARAMETER OPTIMIZATION (Randomized Search) ---
    print("\n" + "="*50)
    print("1. RUNNING HYPERPARAMETER TUNING...")
    print("="*50)
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'gamma': [0, 0.1, 0.5, 1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0]
    }

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        enable_categorical=True, 
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False 
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    random_search = RandomizedSearchCV(
        estimator=xgb_base, 
        param_distributions=param_dist,
        n_iter=50, # Number of parameter settings sampled
        scoring=scorer, 
        cv=cv, 
        verbose=0,
        random_state=42,
        n_jobs=-1 
    )

    random_search.fit(X_train, y_train)

    # 3. Extract Best Parameters
    best_params = random_search.best_params_
    best_params_cv_score = random_search.best_score_
    
    # Add the fixed parameters back
    best_params.update({
        'objective': 'binary:logistic',
        'scale_pos_weight': ratio,
        'enable_categorical': True, 
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False
    })
    
    print("\n" + "="*50)
    print("✨ OPTIMIZATION COMPLETE")
    print(f"Best ROC AUC Score on CV: {best_params_cv_score:.4f}")
    print("Best Hyperparameters:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")
    print("="*50)

    # --- PART B: FINAL MODEL TRAINING & EVALUATION ---
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run():
        print("\n2. TRAINING FINAL XGBOOST MODEL with Best Parameters...")

        # Initialize and train the final model using the discovered parameters
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        # 4. Evaluation
        # Training Performance
        train_preds = final_model.predict(X_train)
        train_proba = final_model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, train_preds)
        train_roc_auc = roc_auc_score(y_train, train_proba)
        
        # Test Performance (Unbiased)
        test_preds = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, test_preds)
        test_roc_auc = roc_auc_score(y_test, test_proba)
        
        print("\n" + "="*45)
        print("✅ FINAL OPTIMIZED MODEL PERFORMANCE (Test Set)")
        print("="*45)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:     {test_acc:.4f}")
        print("-" * 45)
        print(f"Training ROC AUC:  {train_roc_auc:.4f}") 
        print(f"Test ROC AUC:      {test_roc_auc:.4f}")  
        print("-" * 45)
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_preds))

        # 5. Logging to MLflow
        for key, value in random_search.best_params_.items():
            mlflow.log_param(key, value)
        mlflow.log_param("scale_pos_weight", ratio)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("cv_best_roc_auc", best_params_cv_score)

        mlflow.xgboost.log_model(final_model, "churn_xgb_model_optimized")
        
        # 6. Save Local Model
        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save_model(CHURN_MODEL_PATH)
        print(f"\nModel saved locally to {CHURN_MODEL_PATH}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    run_optimization_and_training()