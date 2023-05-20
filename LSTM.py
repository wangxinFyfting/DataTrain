import pandas_datareader.data as web
import datetime
from sklearn.preprocessing import StandardScaler
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2021, 9, 1)
df = web.DataReader('GOOGL', 'stooq', start, end)
df.dropna(inplace=True)
df.sort_index(inplace=True)
pre_days = 10
df['label'] = df['Close'].shift(pre_days)
df.dropna(inplace=True)
# print(df)
scaler = StandardScaler()
scal_X = scaler.fit_transform(df.iloc[:,:-1])
# print(scal_X)
mem_his_days = 5
deq = deque(maxlen=mem_his_days)
X = []
for i in scal_X:
    deq.append(i)
    if len(deq) == mem_his_days:
        X.append(list(deq))
X_lately = X[-pre_days:]
X = X[:-pre_days]
print(len(X))
print(len(X_lately))

Y = df['label'].values[mem_his_days-1:-pre_days]

print(len(Y))


x = numpy.array(X)
y = numpy.array(Y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
model = Sequential()

model.add(LSTM(10, input_shape=x.shape[1:], activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.compile(optimizer='adam', loss='mse', metrics=['mape'])
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)
