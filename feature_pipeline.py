import os
import requests
import pandas as pd
import hopsworks
import json
import datetime

project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

fs = project.get_feature_store()

def get_aqi_data(city, api_token):
    url = f"https://api.waqi.info/feed/{city}/?token={api_token}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']
        aqi = data['aqi']
        date = data['time']['s'].split(' ')[0] # Extract date in 'YYYY-MM-DD' format
        return pd.DataFrame([{'date': date, 'aqi': aqi}])
    else:
        print("Failed to fetch data")
        return pd.DataFrame()
    
AQICN_API_TOKEN = "your_api_token_here"  # Replace with your actual API token
CITY = "london"

# Fetch the data
aqi_df = get_aqi_data(CITY, AQICN_API_TOKEN)

if not aqi_df.empty:
    # Basic Feature Engineering: Add a timestamp
    aqi_df['date'] = pd.to_datetime(aqi_df['date'])
    aqi_df['aqi'] = aqi_df['aqi'].astype(float)

    aqi_fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    seven_days_ago = now_utc - datetime.timedelta(days=14)

    query = aqi_fg.select_all()

    query = query.filter(aqi_fg.date > seven_days_ago)

    historical_df = query.read()
    historical_df["date"] = pd.to_datetime(historical_df["date"]).dt.date
    aqi_df["date"] = aqi_df["date"].dt.date

    combined_df = pd.concat([historical_df, aqi_df], ignore_index=True)
    combined_df.sort_values(by='date', inplace=True)

    for i in range(1, 8):
        combined_df[f'aqi_lag_{i}'] = combined_df['aqi'].shift(i)
    
    combined_df["aqi_rolling_mean_6"] = combined_df['aqi'].shift(1).rolling(window=7).mean()
    combined_df["aqi_rolling_std_6"] = combined_df['aqi'].shift(1).rolling(window=7).std()

    last_row = combined_df.tail(1).copy()
    last_row["date"] = pd.to_datetime(pd.to_datetime(last_row['date']).dt.date)
    last_row.reset_index(drop=True, inplace=True)
    
    aqi_fg.insert(last_row)
    print("Successfully inserted new AQI data into the feature store.")