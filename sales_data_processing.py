import pandas
import datetime

# 计算warning time， 返回打标个数
def slaes_wtimes(dForm, warnTime):
    # Calculate how many pieces of data will be in the warning time
    delta = dForm.loc[1, "Time Point"] - dForm.loc[0, "Time Point"]
    delta_seconds = delta.total_seconds()
    wtimes = int(warnTime / delta_seconds)
    return wtimes


#打标
def label_data(data, pop, times, shiftFlag = False):

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

    data["label"] = 0
    for i in range(0, failed_num):
        index = failed_form.index[i]
        preTime = index - times
        if preTime < 0 :
            preTime = 0
        if shiftFlag == False:
            data.loc[range(preTime, index + 1), "label"] = failed_form.loc[index, "label"]
        else:
            data.loc[index, "label"] = failed_form.loc[index, "label"]
    return data

# 转换Primary Pollutant标签
def translatePP(data_form): 
    data_form['Primary Pollutant'].replace(['TVOC', 'PM2.5', 'CO2', 'PM10'], [1, 2, 3, 4], inplace=True)

#RUL数据处理
def sales_Data_RUL(data_form, warning_time, pop, saveName = ""):
    # Obtain raw data and format the time
    data_form.loc[:, "Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%Y/%m/%d %H:%M').copy()
    data_form = data_form.sort_values(by="Time Point", ascending=True)
    data_form = data_form.reset_index(drop=True)
    wtimes = slaes_wtimes(data_form, warning_time)    
    data_form = label_data(data_form, pop, wtimes)
    translatePP(data_form)
    if saveName != "":
        data_form.to_csv(saveName)
    return data_form

# timeshift数据处理
def sales_data_timeShift(data_form, warning_time, pop, saveName = ""):
    # Obtain raw data and format the time
    data_form.loc[:, "Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%Y/%m/%d %H:%M')
    data_form = data_form.sort_values(by="Time Point", ascending=True)
    data_form = data_form.reset_index(drop=True)
    wtimes = slaes_wtimes(data_form, warning_time)
    
    y = data_form["Overall Index (US)"]
    y = y.drop(y.index[0:wtimes], axis=0)
    y = y.reset_index(drop=True)
    data_form = data_form.drop("Overall Index (US)", axis=1)
    data_form = data_form.drop(data_form.index[-wtimes:], axis=0)
    data_form = data_form.reset_index(drop=True)

    data_form = pandas.concat([data_form, y], axis=1)
    data_form = data_form.reset_index(drop=True)

    label_data(data_form, pop, wtimes, True)
    translatePP(data_form)
    if saveName != "":
        data_form.to_csv(saveName)
    return data_form

def buildData(data_form, warning_time, pop):
    for i in range(len(pop)):
        sales_data_timeShift(data_form, warning_time, [pop[i]], saveName = f"timeShift_label_{pop[i]}_warningtime{warning_time}.csv")
        sales_Data_RUL(data_form, warning_time, [pop[i]], saveName = f"RUL_label_{pop[i]}_warningtime{warning_time}.csv")

# if __name__ == '__main__':
#     print("Khery")
#     data_form = pandas.read_csv('Khery_366.csv')
#     data_form = data_form.iloc[0:172800, :].copy()
#     buildData(data_form, 600, [100, 150, 250])
