# Data processing
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # chatgpt 说应该用standardScaler
# LSTM training
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

# Comparison
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# LSTM autoencoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed


# def data_scale(df_air_quality):
#     # data_scale 的方法有很多，这里用的是MinMaxScaler 
#     # 需要update # (-1,1) 和 （0，1）取决与激活函数，如果是tanh，就是-1，1，如果是sigmoid/relue，就是0，1

#     # create a MinMaxScaler object
#     # scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler = StandardScaler() # standardize features by removing the mean and scaling to unit variance

#     # Convert "Time Point" column to datetime format
#     # df_air_quality['Time Point'] = pd.to_datetime(df_air_quality['Time Point']) 前面探究数据分布时有了这一句，就不需要了

#     # fit the MinMaxScaler object with the training data
#     # df_air_quality['km200.rpm10c (µg/m³)'] = scaler.fit_transform(df_air_quality['km200.rpm10c (µg/m³)'].values.reshape(-1,1))
#     df_air_quality['km200.rpm25c (µg/m³)'] = scaler.fit_transform(df_air_quality['km200.rpm25c (µg/m³)'].values.reshape(-1,1))
#     df_air_quality['km203.rtvoc (ppb)'] = scaler.fit_transform(df_air_quality['km203.rtvoc (ppb)'].values.reshape(-1,1))
#     df_air_quality['rco2 (ppm)'] = scaler.fit_transform(df_air_quality['rco2 (ppm)'].values.reshape(-1,1))
#     df_air_quality['rhumid (%)'] = scaler.fit_transform(df_air_quality['rhumid (%)'].values.reshape(-1,1))
#     df_air_quality['rtemp (C)'] = scaler.fit_transform(df_air_quality['rtemp (C)'].values.reshape(-1,1))
#     df_air_quality['Time Point'] = scaler.fit_transform(df_air_quality['Time Point'].values.reshape(-1,1))

#     return df_air_quality

# def LSTM_data_processing(df, mem_his_days, pre_days): 
#     # df.dropna(inplace=True) 空值应该在数据处理部分解决
#     # df.sort_index(inplace=True)
    
#     deq = deque(maxlen=mem_his_days)
#     X = []
#     for i in df.iloc[:,:-1].values:
#         deq.append(list(i))
#         if len(deq) == mem_his_days:
#             X.append(list(deq))
#     X_lately = X[-pre_days:]
#     X = X[:-pre_days]

#     y = df['label'].values[mem_his_days-1:-pre_days]
    
#     X = np.array(X)
#     y = np.array(y)

#     return X, y, X_lately

# def LSTM_training_tunning(df_train, df_test, pre_days, mem_days):
#     # 记得参数调整
#     lstm_layers = [1,2,3]
#     dense_layers = [1,2,3]
#     uints = [8,16,32,64]
#     batch_sizes = [16,32,64]
#     for the_mem_days in mem_days:
#         for the_lstm_layers in lstm_layers:
#             for the_dense_layers in dense_layers:
#                 for the_uints in uints:
#                     for the_batch_sizes in batch_sizes:
#                         for the_pre_days in pre_days:
#                             filepath = './models/{val_loss:.2f}_{epoch:02d}_' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_uints}_batch_{the_batch_sizes}_preday_{the_pre_days}'
#                             #filepath = './models/{val_loss:.2f}_{epoch:02d}_' 
#                             checkpoint = ModelCheckpoint(
#                                 filepath = filepath,
#                                 save_weights_only = False,
#                                 monitor = "val_mape",
#                                 mode = "min",
#                                 save_best_only = True)
                            
#                             X_train, y_train, _ = LSTM_data_processing(df_train, the_mem_days, the_pre_days)
#                             X_test, y_test, _ = LSTM_data_processing(df_test, the_mem_days, the_pre_days)
#                             model = Sequential()
#                             model.add(LSTM(the_uints, input_shape=X_train.shape[1:], activation='relu', return_sequences=True))
#                             model.add(Dropout(0.1))

#                             for i in range(the_lstm_layers):
#                                 model.add(LSTM(the_uints, activation='relu', return_sequences=True))
#                                 model.add(Dropout(0.1))

#                             model.add(LSTM(the_uints, activation='relu'))
#                             model.add(Dropout(0.1))

#                             for i in range(the_dense_layers):
#                                 model.add(Dense(the_uints, activation='relu'))
#                                 model.add(Dropout(0.1))

#                             model.add(Dense(1))
#                             model.compile(optimizer='adam', loss='mse', metrics=['mape'])
#                             model.fit(X_train, y_train, batch_size=the_batch_sizes, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

#                             y_pred = model.predict(X_test)
#                             df_test.reset_index(drop=True)
#     return

# if __name__ == "__main__":
#     #-------------------------------------Overall index > 100------------------------------------------------------------
#     # 这里这个 warning time = 600s是 RUL和TS生成时的参数。 100，150，250是空气质量划分的阈值
#     # need deleted the unnamed column
#     Dataset= pd.read_csv('RUL_100_wt600s.csv')
#     Dataset_AE= pd.read_csv('TS_100_wt600s.csv')
#     #-------------------------------------Overall index > 150------------------------------------------------------------
#     # Dataset= pd.read_csv('RUL_150_wt600s.csv')
#     # Dataset_AE= pd.read_csv('TS_150_wt600s.csv')
#     #-------------------------------------Overall index > 250------------------------------------------------------------
#     # Dataset= pd.read_csv('RUL_250_wt600s.csv')
#     # Dataset_AE= pd.read_csv('TS_250_wt600s.csv')
#     #

#     Dataset= Dataset.drop(Dataset.columns[0], axis=1)
#     Dataset_AE = Dataset_AE.drop(Dataset_AE.columns[0], axis=1)

#     Dataset['Time Point'] = pd.to_datetime(Dataset['Time Point']) # 没有这一句运行会很慢
#     Dataset_AE['Time Point'] = pd.to_datetime(Dataset_AE['Time Point']) # 没有这一句运行会很慢

#     # Divied the Dataset into train and test, the trainset is 90 days, the testset is 30 days
#     Dataset_train = Dataset.iloc[0:129600, :].copy()
#     Dataset_train_AE = Dataset_AE.iloc[0:129600, :].copy()

#     Dataset_test = Dataset.iloc[129600:172800, :].copy()
#     Dataset_test_AE = Dataset_AE.iloc[129600:172800, :].copy()



#     # AE should keep overall index and utilized it as the target
#     Dataset_train_AE = Dataset_train_AE.drop(['km200.rpm10c (µg/m³)','Primary Pollutant'], axis=1)
#     Dataset_test_AE = Dataset_test_AE.drop(['km200.rpm10c (µg/m³)','Primary Pollutant'], axis=1)

#     #------------------------------------------------------------------------------------------
#     Dataset_train = Dataset_train.drop(['Overall Index (US)', 'Primary Pollutant'], axis=1)
#     Dataset_test = Dataset_test.drop(['Overall Index (US)', 'Primary Pollutant'], axis=1)
#     # Dataset_train
#     # Dataset_test

#     Dataset_train = Dataset_train.drop(['km200.rpm10c (µg/m³)'], axis=1)
#     Dataset_test = Dataset_test.drop(['km200.rpm10c (µg/m³)'], axis=1)


#     Dataset_train = data_scale(Dataset_train)
#     Dataset_test = data_scale(Dataset_test)

#     pre_days =[[10],[20],[30]]
#     mem_days = [[5],[10],[20]]
#     args = sys.argv 
#     if args[1] == "0":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[0], mem_days[0])
#     elif args[1] == "1":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[0], mem_days[1])
#     elif args[1] == "2":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[0], mem_days[2])
#     elif args[1] == "3":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[1], mem_days[0])
#     elif args[1] == "4":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[1], mem_days[1])
#     elif args[1] == "5":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[1], mem_days[2])
#     elif args[1] == "6":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[2], mem_days[0])
#     elif args[1] == "7":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[2], mem_days[1])
#     elif args[1] == "8":
#         LSTM_training_tunning(Dataset_train, Dataset_test, pre_days[2], mem_days[2])
#     else:
#         print("error")


