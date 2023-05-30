import pandas_datareader.data as web
import datetime
from sklearn.preprocessing import StandardScaler
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy
import pandas
import os
import matplotlib.pyplot as plt
def Stock_Price_LSTM_Data_Precesing(df, mem_his_days, pre_days):
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    # df['Overall Index (US)'] = df['Overall Index (US)'].shift(-pre_days)
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

    y = df['Overall Index (US)'].values[mem_his_days-1:-pre_days]
    
    X = numpy.array(X)
    y = numpy.array(y)

    return X, y, X_lately

def save_train_png(y1, y2):
    dir_path = './models'
    # 获取目录下所有子目录
    sub_dirs = os.listdir(dir_path)
    dir = sub_dirs[0]
    lastest = 0

    for name in sub_dirs:
        ctime = os.path.getctime(dir_path + '/' + name)
        if ctime > lastest:
            lastest = ctime
            dir = name
    plt.plot(df.index[-len(y1):], y1, color='red', label='trueData')
    plt.plot(df.index[-len(y1):], y2, color='green', label='predictData')
    plt.lepend()
    plt.savefig('png' + name + '.png', bbox_inches='tight')
    
if __name__ == '__main__':

    # df = pandas.read_csv('./data_csv/RUL/RUL_label_100_warningtime600.csv')
    # df = df.loc[0:3300, :]
    # df.drop(["Time Point"], axis = 1, inplace=True)
    # df.drop(["Primary Pollutant"], axis = 1, inplace=True)
    # df.drop(["label"], axis = 1, inplace=True)
    # # print(df.head(10))

    # X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, 5, 10)
    # best_model = load_model("./models/32.62_05_mem_5_lstm_1_dense_1_unit_32")
    # pre = best_model.predict(X)
    # df.reset_index(drop=True)
    # plt.plot(df.index[-len(y):], y, color='red', label='trueData')
    # plt.plot(df.index[-len(y):], pre, color='green', label='predictData')
    # plt.savefig('lstm.png', bbox_inches='tight')    


    df = pandas.read_csv('./data_csv/RUL/RUL_label_100_warningtime600.csv')
    df = df.loc[0:3300, :]
    df.drop(["Time Point"], axis = 1, inplace=True)
    df.drop(["Primary Pollutant"], axis = 1, inplace=True)
    df.drop(["label"], axis = 1, inplace=True)
    df_train = df.loc[0:3000,:]
    df_test = df.loc[3000:,:]

    # pre_days =10
    # mem_days = [5]
    # lstm_layers = [1]
    # dense_layers = [1]
    # uints = [32]

    pre_days = 60*24
    mem_days = [60, 60*12, 60*24, 60*24*2, 60*24*7]
    lstm_layers = [1, 3, 4, 5]
    uints = [[32], [14, 28, 14], [14, 28, 56, 14], [14, 28, 56, 28, 14]]
    for the_mem_days in mem_days:
        for the_lstm_layers in lstm_layers:
            filepath = './models/{val_loss:.2f}_{epoch:02d}_' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_unit_{the_uints}'
            checkpoint = ModelCheckpoint(
                filepath = filepath,
                save_weights_only = False,
                monitor = "val_mape",
                mode = "min",
                save_best_only = True)
            
            X_train, y_train, _ = Stock_Price_LSTM_Data_Precesing(df_train, the_mem_days, pre_days)
            X_test, y_test, _ = Stock_Price_LSTM_Data_Precesing(df_test, the_mem_days, pre_days)
            model = Sequential()
            model.add(LSTM(14, input_shape=X_train.shape[1:], activation='relu', return_sequences=True))
            model.add(Dropout(0.1))

            for i in range(the_lstm_layers):
                model.add(LSTM(uints[i], activation='relu', return_sequences=True))
                model.add(Dropout(0.1))

            # model.add(LSTM(the_uints, activation='relu'))
            # model.add(Dropout(0.1))

            model.add(Dense(14, activation='relu'))
            model.add(Dropout(0.1))

            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mape'])
            model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

            y_pred = model.predict(X_test)
            save_train_png(y_test, y_pred)