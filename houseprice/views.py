from django.shortcuts import render
import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'houseprice', 'model')
model = pickle.load(open(os.path.join(MODEL_DIR, 'house_price_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
feature_columns = pickle.load(open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'rb'))

def predict_price(request):
    prediction = None
    if request.method == 'POST':
        form_data = {
            'area': float(request.POST['area']),
            'bedrooms': int(request.POST['bedrooms']),
            'bathrooms': int(request.POST['bathrooms']),
            'floors': int(request.POST['floors']),
            'zipcode': request.POST['zipcode'],
            'city': request.POST['city'],
            'statezip': request.POST['statezip'],
            'waterfront': 0,
            'view': 0,
            'condition': 3,
            'grade': 7,
            'sqft_above': 1500,
            'sqft_basement': 500,
            'yr_built': 2005,
            'yr_renovated': 0,
            'lat': 47.5,
            'long': -122.2,
            'sqft_living15': 1500,
            'sqft_lot15': 4000,
            'month': 6,
            'year': 2024
        }

        df = pd.DataFrame([form_data])
        df_encoded = pd.get_dummies(df)
        missing_cols = [col for col in feature_columns if col not in df_encoded.columns]
        for col in missing_cols:
            df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]
        df_scaled = scaler.transform(df_encoded)
        prediction = round(model.predict(df_scaled)[0], 2)

    return render(request, 'frontend.html', {'prediction': prediction})
