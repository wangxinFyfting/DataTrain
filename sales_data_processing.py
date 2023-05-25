import pandas
import numpy
import time
import datetime
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
def slaes_wtimes(dForm, warnTime):
    # Calculate how many pieces of data will be in the warning time
    delta = dForm.loc[1, "Time Point"] - dForm.loc[0, "Time Point"]
    wtimes = int(warnTime / int(delta))
    return wtimes

def label_data(data, pop, times):

    failed_form = pandas.DataFrame()
    # Collect all rows that exceed the Overall Index threshold 
    # and reverse order them based on the index value
    for i in range(0, len(pop)):
        if len(pop) == 1:
            break
        temp_labe = pandas.DataFrame()
        temp_labe = data.loc[(data["Overall Index (US)"] > pop[i]) & (data["Overall Index (US)"] < pop[i+1])].copy()
        temp_labe.loc[:, "label"] = i + 1
        failed_form = pandas.concat([failed_form, temp_labe], axis=0)
        if i + 1 >= len(pop) - 1:
            break

    temp_labe = pandas.DataFrame()
    temp_labe = data.loc[data["Overall Index (US)"] > pop[-1]].copy()
    temp_labe.loc[:, "label"] = len(pop)
    failed_form = pandas.concat([failed_form, temp_labe], axis=0)

    failed_form = failed_form.sort_values(by="label", ascending=False)
    failed_num = failed_form.shape[0]
    # ~~~bug~~~ print(data.loc[failed_labe.index[1] -15 : failed_labe.index[1]])

    # Mark all data within the warning time
    data["label"] = 0
    for i in range(0, failed_num):
        index = failed_form.index[i]
        preTime = index - times
        if preTime < 0 :
            preTime = 0
        data.loc[range(preTime, index + 1), "label"] = failed_form.loc[index, "label"]

def sales_Data_RUL(data_path, warning_time, pop):
    # Obtain raw data and format the time
    data_form = pandas.read_csv(data_path)
    data_form["Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%d/%m/%Y %H:%M')
    data_form["Time Point"] = data_form["Time Point"].apply(lambda x:time.mktime(x.timetuple()))
    data_form = data_form.sort_values(by="Time Point", ascending=True)
    data_form.reset_index(drop=True)
    wtimes = slaes_wtimes(data_form, warning_time)    
    label_data(data_form, pop, wtimes)

    return data_form

def sales_data_timeShift(data_path, warning_time, pop):
    # Obtain raw data and format the time
    data_form = pandas.read_csv(data_path)
    data_form["Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%d/%m/%Y %H:%M')
    data_form = data_form.sort_values(by="Time Point", ascending=True)
    data_form.reset_index(drop=True)
    wtimes = slaes_wtimes(data_form, warning_time)
    
    y = data_form["Overall Index (US)"]
    y = y.drop(y.index[0:wtimes], axis=0)
    y = y.reset_index(drop=True)
    data_form = data_form.drop("Overall Index (US)", axis=1)
    data_form = data_form.drop(data_form.index[-wtimes:], axis=0)
    data_form = data_form.reset_index(drop=True)

    data_form = pandas.concat([data_form, y], axis=1)
    data_form = data_form.reset_index(drop=True)

    label_data(data_form, pop, wtimes)
    data_form.to_csv('newTimeShift.csv')
def new():
    sales_Data_RUL('Khery_7.csv', 600, numpy.array([100, 150]))
    # sales_Data_RUL('Khery_7.csv', 600, numpy.array([100]))
    # sales_Data_RUL('Khery_7.csv', 600, numpy.array([150]))
    # 读取数据
    data_form = pandas.read_csv("newRULData.csv")
    # 将 time 列转换为日期类型
    data_form['Time Point'] = pandas.to_datetime(data_form['Time Point'])

    # 绘制时间序列图
    plt.plot(data_form['Time Point'], data_form['label'])

    # 设置x轴和y轴标签
    plt.xlabel('TTime Pointime')
    plt.ylabel('Label')

    # 设置图形标题
    plt.title('Label over Time Point')

    # 显示图形
    plt.show()
# warning time 6h 4h 2h 1h 30m 10m 0

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 1 
    y_pred = K.clip(y_pred, 0, 1)
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

def Stock_Price_LSTM_Data_Precesing(df, mem_his_days, pre_days):
    df.drop(axis=1, columns="Overall Index (US)", inplace=True)
    df.drop(axis=1, columns="Primary Pollutant", inplace=True)
    print(df.head(10))
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
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

if __name__ == '__main__':
    
    df = sales_Data_RUL('Khery_7.csv', 600, numpy.array([100, 150]))
    print("data ok")
    pre_days =60*24
    # mem_days = [10*1, 30*1, 60*1, 60*12, 60*24]
    # lstm_layers = [1, 2, 4]  #   
    # units = [32,64]  # 

    mem_days = [10*1]
    lstm_layers = [1]  # 
    units = [32]  # 
    
    for the_mem_days in mem_days:
        for the_lstm_layers in lstm_layers:
            filepath = './models/{val_loss:.2f}_{epoch:02d}_' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}'
            checkpoint = ModelCheckpoint(
            filepath = filepath,
            save_weights_only = True,
            min_delta=0,
            patience=50,
            verbose=0,
            mode = "auto",
            save_best_only = True)
            
            X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)
            model = Sequential()        
            for i in range(the_lstm_layers):
                model.add(LSTM(input_shape=X.shape[1:], units=10, return_sequences=True))
                model.add(Dropout(0.2))
            
            model.add(Dense(units=1, input_shape=X.shape[1:], activation='linear'))
            adam = tf.keras.optimizers.legacy.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            mse = tf.keras.losses.MeanSquaredError()
            model.compile(loss= mse, optimizer=adam, metrics=[fbeta]) # mae

            history = model.fit(X_train, y_train, batch_size=100, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint])
