from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import numpy as np
import xgboost as xgb
import os
import requests
import pandas as pd
import hopsworks
import json
import datetime
import pickle

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    host=os.environ["HOPSWORKS_HOST"]  # Ensure you have the HOPSWORKS_HOST environment variable set
)

fs = project.get_feature_store()


aqi_fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)  

query = aqi_fg.select_all()

print("Fetching all historical data for training...")
training_df = query.read(online=False)
print(f"Successfully retrieved {len(training_df)} rows.")

training_df.drop(columns=['date'], inplace=True)

Y = training_df['aqi']
X = training_df.drop(columns=['aqi'], axis=1)

split_point = int(len(training_df)*0.9)

X_train, X_test = X[:split_point], X[split_point:]
Y_train, Y_test = Y[:split_point], Y[split_point:]

print(f"Defining the model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05    # XGBoost handles NaNs by default, no special parameter needed
)

print("Training the model...")
model.fit(X_train, Y_train,
          eval_set=[(X_test, Y_test)],
          verbose=False)

# 1. Make predictions on the test data
y_pred = model.predict(X_test)

# 2. Calculate the metrics
print("Calculating metrics...")
mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
r2 = r2_score(Y_test, y_pred)

mr = project.get_model_registry()

metrics = {
    "mean_absolute_error": mae,
    "root_mean_squared_error": rmse,
    "r_squared": r2
    }

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

filename = 'aqi_xgboost_model.pkl'
model_dir = "aqi_models"

# 2. Open the file in write-binary ('wb') mode and save the model
with open(f"{model_dir}/{filename}", 'wb') as file:
    pickle.dump(model, file)

input_schema = Schema(X_train)
output_schema = Schema(Y_train)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# Create a new model object to register
sklearn_model = mr.sklearn.create_model(
    name="aqi_forecast_model",
    metrics=metrics,
    description="Random Forest model for aqi forecast in London.",
    model_schema=model_schema,
)

sklearn_model.save(model_dir)
