import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle


# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Extract year and month from 'date' if it exists
if 'date' in df.columns:
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month

# Step 3: Drop unnecessary columns safely
columns_to_drop = ['date', 'street', 'country']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

# Step 4: One-hot encode 'city' and 'statezip' if they exist
for col in ['city', 'statezip']:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Step 5: Split features and target
X = df.drop('price', axis=1)
y = df['price']

#  Step 6: Train-test split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 7: Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully.")
print(" Mean Squared Error:", round(mse, 2))
print(" R² Score:", round(r2, 4))



with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model, scaler, and feature list saved to 'model/' folder.")
