import os
import requests
import pandas as pd
import hopsworks

project = hopsworks.login(
    api_key_value="wn1dqr4luNONtCPW.y2wmDAXEIgNdRiaqict0iNRgTh12zIoRZmL8olFjSWDmyypo5CwklmbCvVAu43Hm",
    project="AQIndex_forecast"
)

fs = project.get_feature_store()

def get_aqi_data(city, api_token):
    url = f"https://api.waqi.info/feed/{city}/?token={api_token}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']
        aqi = data['aqi']
        time = data['time']['s']
        return pd.DataFrame([{'city': city, 'time': time, 'aqi': aqi}])
    else:
        print("Failed to fetch data")
        return pd.DataFrame()
    
AQICN_API_TOKEN = "e3c818ba649b427837fea729a38d987d67526f33"
CITY = "tashkent"

# Fetch the data
aqi_df = get_aqi_data(CITY, AQICN_API_TOKEN)

if not aqi_df.empty:
    # Basic Feature Engineering: Add a timestamp
    aqi_df['timestamp'] = pd.to_datetime('now').round('H')
    
    # --- Storing Features in Hopsworks ---
    # Create or get a feature group
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Air Quality Index Features"
    )
    
    # Insert data into the feature group
    aqi_fg.insert(aqi_df)
    print("Successfully inserted new AQI data into the feature store.")