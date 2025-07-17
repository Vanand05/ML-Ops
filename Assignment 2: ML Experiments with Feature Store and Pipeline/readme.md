Athlete Performance Prediction - Feature Store ML Pipeline

A complete MLOps pipeline using Feast Feature Store to predict high-performing athletes based on CrossFit competition data.

## Dataset Note
The original `athletes.csv` (82MB) exceeds GitHub's file size limit. The processed feature datasets (`feature_v1.csv` and `feature_v2.csv`) are included. Original dataset available upon request.

## Project Overview

This project implements a machine learning pipeline with feature store capabilities to predict athlete performance classification. The system uses two different feature versions, multiple algorithms, and tracks carbon emissions for sustainable ML operations.

### Assignment Requirements Met
- Feature Store implementation with ML Pipeline (Feast)
- 2 different feature versions created and deployed
- 4 model combinations: 2 algorithms × 2 hyperparameter sets
- Quantitative and qualitative model comparison
- Carbon emissions tracking for all experiments

## Dataset

**Source:** Athletes.csv (423,006 records, 27 features)
**Target:** Binary classification - High Performer vs Regular Performer
**Criteria:** Athletes with Deadlift ≥350lbs, Snatch ≥150lbs, Pullups ≥20 reps

## Architecture
Feature Store (Feast) → ML Pipeline → Model Training → Results Analysis
↓
Carbon Tracking (CodeCarbon)

### Feature Versions
- **Version 1 (Physical):** age, height, weight, deadlift, backsq, pullups
- **Version 2 (Performance):** fran, helen, grace, run400, run5k, snatch

## Quick Start

### Prerequisites
```bash
pip install feast pandas scikit-learn matplotlib seaborn codecarbon
Setup and Run
bash# 1. Setup Feast Feature Store
feast apply

# 2. Run complete ML Pipeline
python ml_pipeline.py

# 3. Generate Analysis Report
python analysis_report.py
📈 Results Summary
Model Performance
AlgorithmFeature VersionAccuracyF1-ScoreROC-AUCRandomForestV1 (Physical)0.9290.9200.972RandomForestV1 (Physical)0.9270.9180.971RandomForestV2 (Performance)0.8580.8820.898RandomForestV2 (Performance)0.8490.8740.897LogisticRegressionV2 (Performance)0.8340.8550.888LogisticRegressionV2 (Performance)0.8300.8520.885LogisticRegressionV1 (Physical)0.6700.5710.749LogisticRegressionV1 (Physical)0.6160.3230.713
Key Findings
🏆 Best Model: RandomForest with Feature Version 1 (Physical attributes)
🌱 Most Efficient: LogisticRegression Set1 V2 (lowest carbon footprint)
📊 Feature Impact: Physical attributes (V1) outperform performance metrics (V2)
⚡ Dataset Size Effect: V1 (33K samples) vs V2 (4.6K samples) significantly impacts performance
Carbon Emissions Analysis

Range: 1.29e-10 to 2.90e-06 kg CO2 (22,000x difference)
Most Efficient: LR_Set1_V2
Least Efficient: LR_Set2_V1
Insight: RandomForest models consume more energy due to ensemble nature

📁 Project Structure
feature_repo/
├── README.md                    # This file
├── feature_store.yaml          # Feast configuration
├── athlete_feature_v1.py       # Feature Version 1 definition
├── athlete_feature_v2.py       # Feature Version 2 definition
├── ml_pipeline.py              # Main training pipeline
├── analysis_report.py          # Results analysis and visualization
├── athletes_pipeline.ipynb     # Data exploration notebook
├── experiment_results.csv      # Model performance metrics
├── emissions.csv               # Carbon emissions data
├── comprehensive_analysis.png  # Results visualization
└── data/
    ├── athletes.csv            # Original dataset
    ├── feature_v1.csv         # Physical attributes features
    └── feature_v2.csv         # Performance metrics features
🛠️ Technical Implementation
Feature Store (Feast)

Provider: Local SQLite
Entity: athlete_id (INT64)
Features: Two versioned feature views with different TTL settings
Storage: Local file system with timestamp-based retrieval

Machine Learning Pipeline

Algorithms: RandomForest, LogisticRegression
Hyperparameters: 2 configurations per algorithm
Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Cross-validation: 80/20 train-test split with stratification

Carbon Tracking

Tool: CodeCarbon
Metrics: CPU, RAM, and GPU energy consumption
Granularity: Per-experiment tracking with detailed breakdowns

📊 Visualizations
The comprehensive_analysis.png includes:

Performance heatmaps by algorithm and feature version
Metrics distribution analysis
Carbon emissions comparison
Performance vs carbon trade-off analysis
Hyperparameter impact assessment
Energy consumption breakdowns

🔬 Key Insights
Quantitative Analysis

RandomForest consistently outperforms LogisticRegression
Feature Version 1 achieves 8-9% higher accuracy than Version 2
Hyperparameter tuning shows minimal impact on RandomForest
LogisticRegression highly sensitive to regularization parameters

Qualitative Analysis

Physical attributes are stronger predictors than performance metrics
Dataset size significantly impacts model performance
L1 regularization with low C severely degrades LogisticRegression
Carbon efficiency varies dramatically between model configurations

Business Impact

Physical measurements provide reliable performance prediction
RandomForest offers best accuracy-efficiency balance
Carbon tracking reveals significant environmental cost differences
Feature engineering strategy significantly impacts both performance and sustainability

🔄 Reproducibility
All experiments use fixed random seeds (random_state=42) for reproducible results. The complete pipeline can be re-executed with:
bashfeast apply && python ml_pipeline.py && python analysis_report.py
📝 Future Improvements

Implement feature selection techniques
Add model explainability (SHAP, LIME)
Explore ensemble methods combining both feature versions
Implement automated hyperparameter optimization
Add real-time prediction serving capabilities

👨‍💻 Author: Veera Anand

Project completed: July 17, 2025
Assignment: Feature Store ML Pipeline Implementation
