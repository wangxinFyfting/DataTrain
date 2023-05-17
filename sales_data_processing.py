import pandas
from datetime import datetime


def sales_Data_RUL(data_path, warning_time):
    data_form = pandas.read_csv(data_path)
    data_form["reasult"] = 0
    data_form["Time Point"] = pandas.to_datetime(data_form["Time Point"], format='%Y/%m/%d %H:%M')
    failed_labe = data_form.loc[data_form["Overall Index (US)"] > 100]
    failed_labe = failed_labe.sort_values(by="Time Point", ascending=False)
    print(failed_labe.head(5))
    # sorted_df = data_form.sort_values(by="Time Point", ascending=False)
    # sorted_df.tail(1)
    # print(sorted_df.head(5))
    # 将字符串转换为datetime对象
    # date_str = '2022/1/22 0:26'
    # date_obj = datetime.strptime(data_labe["Time Point"], '%Y/%m/%d %H:%M')

    # # 计算时间差
    # diff = datetime.now() - date_obj

if __name__ == '__main__':
    sales_Data_RUL('Khery_366.csv', 10)