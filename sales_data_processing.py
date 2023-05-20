import pandas
import numpy
import datetime

def slaes_wtimes(dForm, warnTime):
    # Calculate how many pieces of data will be in the warning time
    delta = dForm.loc[1, "Time Point"] - dForm.loc[0, "Time Point"]
    delta_seconds = delta.total_seconds()
    wtimes = int(warnTime / delta_seconds)
    return wtimes

def label_data(data, pop, times):

    failed_form = pandas.DataFrame()
    # Collect all rows that exceed the Overall Index threshold 
    # and reverse order them based on the index value
    for i in range(0, len(pop)):
        if len(pop) == 1:
            break
        temp_labe = pandas.DataFrame()
        temp_labe = data.loc[data["Overall Index (US)"] > pop[i] and data["Overall Index (US)"] < pop[i+1]].copy()
        temp_labe.loc[:, "reasult"] = i + 1
        failed_form = pandas.concat([failed_form, temp_labe], axis=0)
        if i + 1 >= len(pop) - 1:
            break

    temp_labe = pandas.DataFrame()
    temp_labe = data.loc[data["Overall Index (US)"] > pop[-1]].copy()
    temp_labe.loc[:, "reasult"] = len(pop)
    failed_form = pandas.concat([failed_form, temp_labe], axis=0)

    failed_form = failed_form.sort_values(by="Time Point", ascending=False)
    failed_num = failed_form.shape[0]
    # ~~~bug~~~ print(data.loc[failed_labe.index[1] -15 : failed_labe.index[1]])

    # Mark all data within the warning time
    data["reasult"] = 0
    for i in range(0, failed_num):
        index = failed_form.index[i]
        data.loc[range(index - times, index + 1), "reasult"] = failed_form.loc[index, "reasult"]

def sales_Data_RUL(data_path, warning_time, pop):
    # Obtain raw data and format the time
    data_form = pandas.read_csv(data_path)
    data_form["Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%Y/%m/%d %H:%M')
    wtimes = slaes_wtimes(data_form, warning_time)    
    label_data(data_form, pop, wtimes)

    data_form.to_csv('newRULData.csv')

def sales_data_timeShift(data_path, warning_time, pop):
    # Obtain raw data and format the time
    data_form = pandas.read_csv(data_path)
    data_form["Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%Y/%m/%d %H:%M')
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

if __name__ == '__main__':
    sales_Data_RUL('Khery_366.csv', 600, numpy.array([100]))
    sales_data_timeShift('Khery_366.csv', 600, numpy.array([100]))