from datetime import datetime
import sys, os
import matplotlib
# matplotlib.style.use("ggplot")
import matplotlib.pyplot as plt
import pandas as pd
data_dir = os.path.join(os.getcwd(), "data")
import pcse
from pcse.fileinput import CABOFileReader, ExcelWeatherDataProvider, YAMLAgroManagementReader, YAMLCropDataProvider
from pcse.exceptions import PCSEError, PartitioningError
from pcse.base import ParameterProvider
from pcse.util import WOFOST72SiteDataProvider
from pcse.models import Wofost72_PP, Wofost72_WLP_FD
import yaml
import numpy as np
# import pandas as pd
from SALib.sample import sobol as sobolSample
from SALib.analyze import sobol, fast
# import matplotlib.pyplot as plt
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_colwidth", 250)
import time
from datetime import datetime
import pickle
from progressbar import printProgressBar, PrintProgressBar
from tqdm import tqdm
from SALib.sample.fast_sampler import sample as efast_sample


def calDays(start, end, format="%Y-%m-%d"):
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return days


def parase_model_ouput(model_outplut: list, var_name: list, pid, idx=0):
    res_ = list()
    for each in var_name:
        try:
            res_.append(model_outplut[idx][each])
        except KeyError as e:
            print("%s\n 不存在key=%s"%(e, each))
            res_.append(np.nan)
        except IndexError:
            res_.append(np.nan)

    res_.append(pid)
    return res_


def create_dict(key_das):
    res_dict = dict()
    res_dict["0"] = list()
    for each in key_das:
        dict_key = str(each)
        res_dict[dict_key] = list()
    return res_dict


def convert2simlab1(input_data: pd.DataFrame, output_file):
    var_name = input_data.columns
    row, col = input_data.shape
    with open(output_file, "w", encoding='utf-8') as sf:
        sf.write("%s\n"%col)
        for each in var_name:
            sf.write("%s\n"%each)
        sf.write("time\t=\tno\n")
        sf.write("%s\n"%row)
        for i in range(row):
            for j in range(col):
                value = input_data.iloc[i][j]
                sf.write("%s"%value)
                if j != col - 1:
                    sf.write("\t")
                elif j == col - 1:
                    sf.write("\n")


def convert2simlab(input_data: pd.DataFrame, output_file):
    input_data.to_csv(output_file, sep="\t", encoding="utf-8", index=0, header=0)
    var_name = input_data.columns
    row, col = input_data.shape
    with open(output_file, "r+", encoding='utf-8') as sf:
        sf.seek(0)
        sf.write("%s\n"%col)
        for each in var_name:
            sf.write("%s\n"%each)
        sf.write("time\t=\tno\n")
        sf.write("%s\n"%row)


def get_wofost_output(wofostModel, var_names, template_dict, pid, var_list=None):
    wofostModel.run_till_terminate()
    r = wofostModel.get_summary_output()
    tmp_res = parase_model_ouput(r, var_name=var_names, pid=pid, idx=0)
    template_dict["0"].append(tmp_res)
    if var_list is None:
        pass
    else:
        days, name_ = var_list
        r = wofostModel.get_output()
        for ii in days:
            # print(ii)
            tmp_res = None
            tmp_res = parase_model_ouput(r, var_name=name_, idx=ii, pid=pid)
            template_dict[str(ii)].append(tmp_res)

    return template_dict


def get_efast_sample(problem, n, M1, seed1):
    return efast_sample(problem, n, M1, seed=seed1)


def run_wofost(variable_name, paramsets: np.ndarray, result_dict, parameters, agrodata, weatherdata, pkl_file,
               target_variables, target_list=None, method=0):
    """

    :param problem: yaml
    :param paramsets:
    :param result_dict:
    :param parameters:
    :param agrodata:
    :param weatherdata:
    :param pkl_file:
    :param target_variables:
    :param target_list:
    :param method: 0: Wofost72PP, 1: Wofost72WLP
    :return:
    """
    total_row = len(paramsets)
    bar_format = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]"
    ncols = 80
    mycropd = {
        "AMAXTB": [0.0, 70.0, 1.25, 70.0, 1.5, 63.0, 1.75, 49.0, 2.0, 21.0],
        "TMPFTB": [0.00, 0.01, 9.00, 0.05, 16.0, 0.80, 18.0, 0.94, 20.0, 1.0, 30.0, 1.0, 36.0, 0.95, 42.0, 0.56],
        "TMNFTB": [5.0, 0.0, 8.0, 1.0],
        "RFSETB": [0.0, 1.0, 1.5, 1.0, 1.75, 0.75, 2.0, 0.25],
        "SLATB": [0.0, 0.0026, 1.0, 0.0012, 2.0, 0.0012],
        "KDIFTB": [0.0, 0.6, 2.0, 0.6],
        "EFFTB": [0.0, 0.45, 40.0, 0.45]

    }
    with tqdm(total=total_row, ncols=ncols, bar_format=bar_format) as _tqdm:
        _tqdm.set_description("{}".format("完成进度"))
        for i, paramset in enumerate(paramsets):
            parameters.clear_override()
            ## 设置TSUM1 TSUM2
            parameters.set_override("TSUMEM", 150.0)
            parameters.set_override("TSUM1", 1290)
            parameters.set_override("TSUM2", 650)
            ## 修改敏感参数值
            value_slatb = mycropd["SLATB"]
            value_slatb[1], value_slatb[3], value_slatb[5] = paramset[0], paramset[1], paramset[2]
            parameters.set_override("SLATB", value_slatb)

            values = mycropd["KDIFTB"]
            values[1], values[3] = paramset[3], paramset[4]
            parameters.set_override("KDIFTB", values)

            values = mycropd["EFFTB"]
            values[1], values[3] = paramset[5], paramset[6]
            parameters.set_override("EFFTB", values)

            values = mycropd["AMAXTB"]
            values[1], values[3], values[5], values[7], values[9] = paramset[10], paramset[7], paramset[8], paramset[9], paramset[11]
            parameters.set_override("AMAXTB", values)

            values = mycropd["TMPFTB"]
            values[1], values[11] = paramset[12], paramset[13]
            parameters.set_override("TMPFTB", values)

            values = mycropd["TMNFTB"]
            values[3] = paramset[14]
            parameters.set_override("TMNFTB", values)

            values = mycropd["RFSETB"]
            values[1], values[7] = paramset[15], paramset[16]
            parameters.set_override("RFSETB", values)
            # print("RFSETB: %s"%values)

            for name, value in zip(variable_name[17: ], paramset[17: ]):
                try:
                    parameters.set_override(name, value)
                except PCSEError as e:
                    print("Error: %s, id:%s"%(e, i))

            try:
                if method == 0:

                    wofostpp = Wofost72_PP(parameters, weatherdata, agrodata)
                    result_dict = get_wofost_output(wofostpp, target_variables, result_dict, pid=i, var_list=target_list)
                else:
                    wofostwlp = Wofost72_WLP_FD(parameters, weatherdata, agrodata)
                    result_dict = get_wofost_output(wofostwlp, target_variables, result_dict, pid=i, var_list=target_list)
            except ZeroDivisionError as e:
                print(e)

            if (i + 1) % 50 == 0 or (i + 1) == total_row:
                with open(pkl_file, "wb") as f:
                    pickle.dump(result_dict, f)

            _tqdm.set_postfix(number = i)
            _tqdm.update(1)

    return result_dict


if __name__ == "__main__":
    ## 2.1 作物参数
    cropfile = os.path.join(data_dir, 'crop', 'maize.yaml')
    cropd = YAMLCropDataProvider()
    cropd.set_active_crop('maize', 'Grain_maize_201')

    ## 土壤参数
    soilfile = os.path.join(data_dir, 'soil', 'ec3.soil')
    soild = CABOFileReader(soilfile)

    ## 站点数据
    sited = WOFOST72SiteDataProvider(WAV=18)

    ## 整合模型参数
    parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)

    ## 管理文件
    agromanagement21 = YAMLAgroManagementReader(os.path.join(data_dir, 'agro', 'maize_2021.agro'))
    agromanagement22 = YAMLAgroManagementReader(os.path.join(data_dir, 'agro', 'maize_2022.agro'))

    ## 气象数据
    weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')
    wdp = ExcelWeatherDataProvider(weatherfile)
    print(wdp)

    ## 敏感性分析设置
    with open(os.path.join(data_dir, 'yaml', 'params1.yaml'), 'r', encoding='utf-8') as f:
        problem_yaml = f.read()

    problem = yaml.safe_load(problem_yaml)

    paramsets = get_efast_sample(problem, 75, 4, 1213)

    ## 设置需要获取的模型输出
    final_target = ["LAIMAX", "TAGP", "TWSO", "DOE", "DOA", "DOM"]
    time_target = ["LAI", "TAGP"]
    das = [18, 27, 37, 48, 56, 65, 75, 85, 95]
    model_start_date = "2021-06-01"
    sowing_date = "2021-06-11"
    base_i = calDays(model_start_date, sowing_date)
    for idx, value_ in enumerate(das):
        das[idx] = base_i + value_

    col1 = ["LAIMAX", "TAGP", "TWSO", "DOE", "DOA", "DOM", "id"]
    col2 = ["LAI", "TAGP", "id"]

    result_pp = create_dict(das)
    result_pp["names1"] = col1
    result_pp["names2"] = col2
    time1 = datetime.now()
    pkl_file = "./data/pickleFile/result_pp2021{0}_{1}_{2}_{3}_{4}.pkl".format(time1.month, time1.day, time1.hour, time1.minute, time1.second)
    result_pp = run_wofost(problem, paramsets, result_pp, parameters, agromanagement21,
                           wdp, pkl_file, final_target, (das, time_target))

    with open(pkl_file, "wb") as pf:
        pickle.dump(result_pp, pf)
