import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import track_emissions
import os
from datetime import datetime

# Set up results storage
results = []
carbon_results = []

def load_feature_data(version):
    """Load feature data for specified version"""
    if version == 1:
        df = pd.read_csv("data/feature_v1.csv")
        feature_cols = ['age', 'height', 'weight', 'deadlift', 'backsq', 'pullups']
    else:
        df = pd.read_csv("data/feature_v2.csv")
        feature_cols = ['fran', 'helen', 'grace', 'run400', 'run5k', 'snatch']
    
    X = df[feature_cols]
    y = df['is_high_performer']
    return X, y, feature_cols

@track_emissions(project_name="athlete_ml_pipeline")
def train_and_evaluate(algorithm, hyperparams, feature_version, X, y, feature_cols):
    """Train model and return results"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize model
    if algorithm == "RandomForest":
        model = RandomForestClassifier(**hyperparams, random_state=42)
    else:  # LogisticRegression
        model = LogisticRegression(**hyperparams, random_state=42, max_iter=1000)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'algorithm': algorithm,
        'hyperparams': str(hyperparams),
        'feature_version': f'v{feature_version}',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, metrics, X_test, y_test, y_pred, y_pred_proba, feature_cols

if __name__ == "__main__":
    print("🚀 Starting ML Pipeline with Carbon Tracking...")
    
    # Define algorithms and hyperparameters
    algorithms = {
        "RandomForest": [
            {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
            {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2}
        ],
        "LogisticRegression": [
            {"C": 1.0, "penalty": "l2"},
            {"C": 0.1, "penalty": "l1", "solver": "liblinear"}
        ]
    }
    
    print("Training 4 model combinations...")
    print("=" * 50)

    # Run experiments for both feature versions
    for feature_version in [1, 2]:
        print(f"\n📊 Feature Version {feature_version}")
        X, y, feature_cols = load_feature_data(feature_version)
        print(f"Data shape: {X.shape}, Target balance: {y.value_counts().to_dict()}")
        
        for algo_name, param_sets in algorithms.items():
            for i, params in enumerate(param_sets, 1):
                print(f"\n🔥 Training {algo_name} (Set {i}) on Features V{feature_version}")
                
                model, metrics, X_test, y_test, y_pred, y_pred_proba, cols = train_and_evaluate(
                    algo_name, params, feature_version, X, y, feature_cols
                )
                
                results.append(metrics)
                
                # Print results
                print(f"Accuracy: {metrics['accuracy']:.3f}")
                print(f"F1-Score: {metrics['f1']:.3f}")
                print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("experiment_results.csv", index=False)
    
    print("\n" + "="*50)
    print("🎯 EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    print(results_df.round(3))
    
    print(f"\n✅ Results saved to experiment_results.csv")
    print(f"📈 Carbon emissions tracked in emissions.csv")