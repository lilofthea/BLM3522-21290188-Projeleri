import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

data = pd.read_csv("gs://my_first_project_bucket_for_cloud/data/data.csv")
data.dropna(inplace=True)
data.drop(index=1, inplace=True)
data.drop(columns=['Series Name', 'Series Code', 'Country Name', 'Country Code'], inplace=True)
long_data = pd.melt(data, var_name='Year', value_name='Value')

# Extract just the numeric year from the 'Year' column.
long_data['Year'] = long_data['Year'].str.extract(r'(\d{4})').astype(int)
long_data.sort_values(by='Year', inplace=True)
long_data.reset_index(drop=True, inplace=True)


X = long_data[['Year']]  
y = long_data['Value'] 

model = LinearRegression()
model.fit(X, y)

# Predict values for the years in the data and add as a new column in the long DataFrame
long_data['Predicted'] = model.predict(X)

future_years = pd.DataFrame({'Year': np.arange(2024, 2029)})
future_predictions = model.predict(future_years)
future_years['Predicted_Value'] = future_predictions / 1e6

joblib.dump(model, 'linear_model1.pkl')
#!gsutil cp linear_model1.pkl gs://my_first_project_bucket_for_cloud/models/model.joblib