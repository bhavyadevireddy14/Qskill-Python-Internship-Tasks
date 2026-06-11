# ==========================================
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("house_prices.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# STEP 2: HANDLE MISSING VALUES
# ==========================================

# Numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# STEP 3: SELECT FEATURES AND TARGET
# ==========================================

features = ['Size', 'Bedrooms', 'Bathrooms', 'Location']
target = 'Price'

X = df[features]
y = df[target]

# ==========================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ==========================================

X = pd.get_dummies(X, drop_first=True)

print("\nFeatures After Encoding:")
print(X.head())

# ==========================================
# STEP 5: TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# STEP 6: TRAIN LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# STEP 7: MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

print("\nSample Predictions:")
for i in range(min(5, len(y_pred))):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]:.2f}")

# ==========================================
# STEP 8: EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ==========================================
# STEP 9: ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 10: FEATURE IMPORTANCE
# ==========================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# ==========================================
# STEP 11: PREDICT NEW HOUSE PRICE
# ==========================================

new_house = pd.DataFrame({
    'Size': [1800],
    'Bedrooms': [3],
    'Bathrooms': [2]
})

# Add missing encoded location columns
for col in X.columns:
    if col not in new_house.columns:
        new_house[col] = 0

new_house = new_house[X.columns]

predicted_price = model.predict(new_house)

print("\nPredicted House Price:")
print(f"₹ {predicted_price[0]:,.2f}")