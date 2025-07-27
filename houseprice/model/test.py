import pandas as pd
import pickle

# Step 1: Load model, scaler, and feature column names
with open("model/house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Step 2: New input data (example)
new_input = pd.DataFrame([{
    'area': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'floors': 1,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'grade': 7,
    'sqft_above': 1500,
    'sqft_basement': 500,
    'yr_built': 2005,
    'yr_renovated': 0,
    'zipcode': 98178,
    'lat': 47.5112,
    'long': -122.257,
    'sqft_living15': 1500,
    'sqft_lot15': 4000,
    'month': 6,
    'year': 2024,
    'city': 'Seattle',
    'statezip': 'WA 98178'
}])
# Step 3: One-hot encode new input
new_input_encoded = pd.get_dummies(new_input)
# Step 4: Add missing columns
missing_cols = [col for col in feature_columns if col not in new_input_encoded.columns]
missing_df = pd.DataFrame(0, index=new_input_encoded.index, columns=missing_cols)
new_input_encoded = pd.concat([new_input_encoded, missing_df], axis=1)

# Step 5: Remove extra/unseen columns not in training
new_input_encoded = new_input_encoded[feature_columns]

# Step 6: Scale input using saved scaler
new_input_scaled = scaler.transform(new_input_encoded)

# Step 7: Predict
predicted_price = model.predict(new_input_scaled)
print("🏠 Predicted House Price: $", round(predicted_price[0], 2))
