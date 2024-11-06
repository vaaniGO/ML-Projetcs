import yfinance as yf
import pandas as pd

# enter information here
start_date = '2013-01-01'
end_date = '2023-01-02'
ticker = 'ADM'
start_year = 13
end_year = 22

stock_data = yf.download(ticker, start_date, end_date)

# Calculate the percentage change for the 'Close' column as compared to the previous day
stock_data['Close'] = stock_data['Close'].pct_change() * 100
stock_data = stock_data.loc[:, ['Close']]
stock_data = stock_data.drop(index='2013-01-02 00:00:00+00:00')

print(stock_data.head())

rain_data = pd.DataFrame()

for i in range(start_year, end_year):
    file_path = f'gom_rain/gom_rain_{i}-{(i+1)}.csv'
    data = pd.read_csv(file_path)
    rain_data = pd.concat([rain_data, data], axis = 0, ignore_index = True)

rain_data.loc[:, ['Rain (mm)']]

rain_data['Time'] = pd.to_datetime(rain_data['Time'], format='%b %d %Y')

# Reformat the date to 'YYYY-MM-DD' format
rain_data['Time'] = rain_data['Time'].dt.strftime('%Y-%m-%d')


# Make sure both 'Time' in rain_data and the date index in stock_data are DateTime types
rain_data['Time'] = pd.to_datetime(rain_data['Time'])
rain_data = rain_data.set_index('Time')
stock_data.index = pd.to_datetime(stock_data.index)
# Step 1: Reset the index to make 'Date' a regular column again
stock_data = stock_data.reset_index()

# Step 2: Rename the 'Date' column to 'Time'
stock_data = stock_data.rename(columns={'Date': 'Time'})

# Step 3: Set 'Time' back as the index
stock_data = stock_data.set_index('Time')
stock_data.columns = ['_'.join(col) for col in stock_data.columns]
stock_data.index = pd.to_datetime(stock_data.index)
if stock_data.index.tz is not None:
    stock_data.index = stock_data.index.tz_localize(None)

# Merge the DataFrames on 'Time' column (only matching dates will be retained)
# Shift the 'rain_data' time by one week earlier
rain_data['Time_shifted'] = rain_data.index + pd.Timedelta(days=0)

# Perform the merge on the shifted time column
combined_data = pd.merge_asof(rain_data.sort_values('Time_shifted'), 
                               stock_data.sort_values('Time'), 
                               left_on='Time_shifted', 
                               right_on='Time', 
                               direction='nearest')

# Print the result
print(combined_data.head())

x = combined_data['Rain (mm)']
y = combined_data[f'Close_{ticker}']

# Step 1: Import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Prepare the data
x = combined_data['Rain (mm)'].values.reshape(-1, 1)  # Reshape to 2D array
y = combined_data[f'Close_{ticker}'].values.reshape(-1, 1)  # Reshape to 2D array

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R^2 Score

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Optional: Visualize the regression line
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Rain (mm)')
plt.ylabel('Stock Close Price')
plt.title('Linear Regression: Rainfall vs Stock Price')
plt.legend()
plt.show()