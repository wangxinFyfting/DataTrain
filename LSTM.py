import pandas_datareader.data as web
import datetime
from sklearn.preprocessing import StandardScaler
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy
def Stock_Price_LSTM_Data_Precesing(df, mem_his_days, pre_days):
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    df['label'] = df['Close'].shift(-pre_days)
    scaler = StandardScaler()
    scal_X = scaler.fit_transform(df.iloc[:,:-1])
    deq = deque(maxlen=mem_his_days)
    X = []
    for i in scal_X:
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X.append(list(deq))
    X_lately = X[-pre_days:]
    X = X[:-pre_days]

    y = df['label'].values[mem_his_days-1:-pre_days]
    
    X = numpy.array(X)
    y = numpy.array(y)

    return X, y, X_lately

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# model = Sequential()

# model.add(LSTM(10, input_shape=x.shape[1:], activation='relu', return_sequences=True))
# model.add(Dropout(0.1))
# model.add(LSTM(10, activation='relu', return_sequences=True))
# model.add(Dropout(0.1))
# model.add(LSTM(10, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.1))
# model.compile(optimizer='adam', loss='mse', metrics=['mape'])
# model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))
# y_pred = model.predict(X_test)


if __name__ == '__main__':

    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2021, 9, 1)
    df = web.DataReader('GOOGL', 'stooq', start, end)
    pre_days =10
    mem_days = [5, 10, 15]
    lstm_layers = [1, 2, 3]
    dense_layers = [1, 2, 3]
    uints = [8, 16, 32]
    for the_mem_days in mem_days:
        for the_lstm_layers in lstm_layers:
            for the_dense_layers in dense_layers:
                for the_uints in uints:
                    filepath = './models/{val_loss:.2f}_{epoch:02d}_' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_uints}'
                    checkpoint = ModelCheckpoint(
                        filepath = filepath,
                        save_weights_only = True,
                        monitor = "val_mape",
                        mode = "min",
                        save_best_only = True)
                    
                    X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)
                    model = Sequential()
                    model.add(LSTM(the_uints, input_shape=X.shape[1:], activation='relu', return_sequences=True))
                    model.add(Dropout(0.1))

                    for i in range(the_lstm_layers):
                        model.add(LSTM(the_uints, activation='relu', return_sequences=True))
                        model.add(Dropout(0.1))

                    model.add(LSTM(the_uints, activation='relu'))
                    model.add(Dropout(0.1))

                    for i in range(the_dense_layers):
                        model.add(Dense(the_uints, activation='relu'))
                        model.add(Dropout(0.1))

                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse', metrics=['mape'])
                    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])
                    # y_pred = model.predict(X_test)
