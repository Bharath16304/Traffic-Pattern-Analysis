import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv('traffic_data.csv')  # columns: timestamp, vehicle_count

# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# Traffic by Hour
hourly = df.groupby('hour')['vehicle_count'].sum()
hourly.plot(kind='line', title='Traffic Density by Hour')
plt.xlabel('Hour')
plt.ylabel('Vehicle Count')
plt.show()

# Optional: Predict Peak Hours
from sklearn.linear_model import LinearRegression
X = hourly.index.values.reshape(-1,1)
y = hourly.values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.plot(hourly.index, hourly.values, label='Actual')
plt.plot(hourly.index, y_pred, label='Predicted', color='red')
plt.title('Traffic Prediction')
plt.legend()
plt.show()
