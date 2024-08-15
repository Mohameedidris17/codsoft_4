import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv('advertising_train.csv')
test_data = pd.read_csv('advertising_test.csv')

print("\nMissing values in Train Data:")
print(train_data.isnull().sum())
print("\nMissing values in Test Data:")
print(test_data.isnull().sum())

X = train_data.drop('Sales', axis=1)
y = train_data['Sales']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

test_predictions = model.predict(test_data_scaled)

print("\nTest Data Predictions:")
print(test_predictions)