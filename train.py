from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

print("Loading dataset...")
# Load data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(data.feature_names)}")

# Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse*100000:.0f}")
print(f"MAE: ${mae*100000:.0f}")
print("="*50)

# Visualize
print("\nCreating visualization...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (in $100k)', fontsize=12)
plt.ylabel('Predicted Price (in $100k)', fontsize=12)
plt.title(f'Housing Price Predictions\nR² Score: {r2:.4f}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
print("✓ Saved: predictions.png")

# Save model
print("Saving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'r2_score': r2,
    'rmse': rmse
}
with open('housing_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✓ Saved: housing_model.pkl")

print("\n✅ Training complete!")