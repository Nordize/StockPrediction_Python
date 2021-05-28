# Introduction  
Idea of this project is about to trying to predict stock price by using Recurrent Neural Network and Machine Learning.  
This project is only about Python programming using financial data.  
Source from: https://www.youtube.com/watch?v=PuZY9q-aKLw&list=WL&index=145&t=447s
# How it work
We use ['Close'] data from Yahoo Finance from 2012-1-1 to 2021-5-28
```
# Load Data
stock_symbol = 'TSLA'

start_date = dt.datetime(2012, 1, 1)
end_date = dt.datetime(2021, 5, 28)
```
with 60 of prediction days
```
# how many days for the prediction days
prediction_days = 60
```
Then created 3 layers of LSTM with 50 nodes
```
# Build the model
model = Sequential()

# 1st layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
# 2nd layers
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# 3rd layers
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output Node, Prediction of the next closing price
```
Load testing Data
```
# Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(stock_symbol, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values
```
Plot 
```
# Plot the Test predictions
plt.plot(actual_prices, color='black', label=f"Actual {stock_symbol} price")
plt.plot(predicted_prices, color='green', label=f"Predict {stock_symbol} price")
plt.title(f"{stock_symbol} share price")
plt.xlabel('time')
plt.ylabel(f'{stock_symbol} share price')
plt.legend()
plt.show()
```
Predict price for next day
```
# Predicting the future price (next day)
real_data = [model_inputs[len(model_inputs)+1 - prediction_days:len(model_inputs+1)],0]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
```