# Housing Price Prediction

Predict California house prices using linear regression and feature scaling.

## Problem Statement
Given 8 features (location, rooms, population, etc.), predict median house price.

## Dataset
- **Source:** California Housing Dataset (scikit-learn)
- **Size:** 20,640 samples, 8 features
- **Target:** Median house price (in $100k)

## Results
- **R² Score:** 0.5757
- **RMSE:** $73,000
- **MAE:** $53,000

## Tech Stack
- Python 3.8+
- scikit-learn (model training)
- pandas, NumPy (data processing)
- matplotlib (visualization)

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python train.py
```

This will:
1. Load and split data
2. Scale features
3. Train LinearRegression
4. Evaluate on test set
5. Save `predictions.png` and `housing_model.pkl`

## Project Structure
├── train.py              # Training script
├── requirements.txt      # Dependencies
├── predictions.png       # Output visualization
├── housing_model.pkl     # Saved model
└── README.md            # This file

## Key Learnings
- Feature scaling (StandardScaler) improves performance
- Linear regression is good baseline for regression tasks
- Train-test split prevents overfitting
- Cross-validation would improve robustness

## Future Improvements
- Implement Random Forest for better accuracy
- Add hyperparameter tuning (GridSearchCV)
- Use cross-validation for robust evaluation
- Explore feature engineering