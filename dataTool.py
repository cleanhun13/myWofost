from datetime import datetime, timedelta
import os
import os.path as Path
import pandas as pd
import numpy as np
from tqdm import tqdm


## 生成日期列表（按日）
def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]


# 时间切片
def data_slice(st, en, data, format="%m-%d"):
    """
    :param beginning: string type of date
    :param ending: string type of date
    :param data: type of pd.DataFrame, index = datetime
    :return pd.DataFrame
    """
    # st = datetime.strptime(beginning, format)
    # en = datetime.strptime(ending, format)
    return data[st: en]


# 读取气象文件
def get_weather(path, format="%Y/%m/%d"):
    """
    :param path: string
    :return pd.DataFrame
    """

    hw = pd.read_csv(path, header=0, encoding="utf-8")

    # 生成日序数
    jdd = np.zeros(hw['date'].shape[0])
    for i in range(0, hw['date'].shape[0]):
        jdd[i] = d2jd(hw.loc[i, 'date'], format)
    # 插入日序数
    hw.insert(1, "DOY", jdd)
    # 设置时间索引
    # hw['date'] = pd.to_datetime(hw['date'], format=format)
    # hw.sort_values('date', inplace=True)
    hw = hw.set_index('DOY')

    try:
        hw.drop('siteNumber', axis=1, inplace=True)
    except KeyError:
        pass
    finally:
        return hw


# 数据拼接
def data_fusion(data1, data2):

    try:
        assert data1.shape[1] == data2.shape[1]
        res = pd.concat([data1, data2], axis=0)
        res.sort_values('DOY', inplace=True)
        return res
    except AssertionError:
        return False


def day_step(date, step=1, format="%Y-%m-%d"):

    strptime, strftime = datetime.strptime, datetime.strftime

    return strftime(strptime(date, format) + timedelta(step), format)
# 日期转儒略日
def d2jd(date: str, format="%Y-%m-%d", mode=1) -> int:

    dt = datetime.strptime(date, format)
    tt = dt.timetuple()
    # return tt.tm_year * 1000 + tt.tm_yday
    if mode == 1:
        return tt.tm_yday
    else:
        return tt.tm_year * 1000 + tt.tm_yday


def data_wash(data):
    """
    data: pd.DataFrame
    """
    # 重新设置索引
    data = data.reset_index(drop=True)
    length = data['date'].shape[0]
    new_doy = list()
    #########
    #日期订正
    ori_date = data.loc[0, 'date']
    year_ = ori_date[0: 4]
    for i in range(0, length):
        try:
            date_ = f"{year_}{(data.loc[i, 'date'])[4: ]}"
        except KeyError:
            continue
        try:
            doy = d2jd(date_, "%Y/%m/%d", 2)
        except ValueError:
            doy = d2jd(date_, "%Y-%m-%d", 2)
        doy = (str(doy))[2: ]
        new_doy.append(doy)
        # data.loc[i, 'date'] = doy
    data.insert(1, "DOY", new_doy)
    #########
    # 数据清洗
    data['Precipitation'] = data['Precipitation'].mask(data['Precipitation'] > 3000, 0)
    data['Precipitation'] = data['Precipitation'].mask(data['Precipitation'] < 0.1, 0)
    lenth = data['Ssr'].shape[0]
    for i in range(lenth):
        if data.loc[i, 'Ssr'] > 50 and i > 0 and i < lenth - 1:
            data.loc[i, 'Ssr'] = data.loc[i-1, 'Ssr'] + (data.loc[i + 1, 'Ssr'] - data.loc[i - 1, 'Ssr'])/2.0

    ## 另一方法实现
    # data['Precipitation'] = np.where(data['Precipitation'] > 3000, 0, data['Precipitation'])
    # data['Precipitation'] = np.where(data['Precipitation'].between(1, 9), 0, data['Precipitation'])
    # df[df[]]
    return data


# 儒略日转日期
def jd2d(jd: str, format="%Y-%m-%d") -> str:
    dt = datetime.strptime(jd, "%Y%j").date()
    return dt.strftime(format)


def split_list(data: list, part: int):
    l = len(data)
    step = int((l + 1) / part)
    res = list()
    for i in range(0, l, step):
        res.append(data[i: i+step])
    return res
