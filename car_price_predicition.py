import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('car_data.csv')
print("First 5 rows:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Preprocessing
df['Current Year'] = 2020
df['Car_Age'] = df['Current Year'] - df['Year']
df.drop(['Year', 'Current Year'], axis=1, inplace=True)

# Convert categorical data
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
print(f"\nR-squared Score: {r2:.2f}")

# Show actual vs predicted
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:")
print(result.head())

# Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.grid(True)
plt.show(block=True)
