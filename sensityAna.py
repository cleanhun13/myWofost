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
import pickle
from progressbar import printProgressBar, PrintProgressBar

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
            res_.append(None)
        except IndexError:
            res_.append(None)

    res_.append(pid)
    return res_


def create_dict(key_das):
    res_dict = dict()
    res_dict["0"] = list()
    for each in key_das:
        dict_key = str(each)
        res_dict[dict_key] = list()
    return res_dict


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



if __name__ == "__main__":
    # 读取作物参数
    cropfile = os.path.join(data_dir, 'crop', 'maize.yaml')
    cropd = YAMLCropDataProvider()
    cropd.set_active_crop('maize', 'Grain_maize_201')
    # 土壤参数
    soilfile = os.path.join(data_dir, 'soil', 'ec3.soil')
    soild = CABOFileReader(soilfile)
    # 站点数据

    sited = WOFOST72SiteDataProvider(WAV=18)
    # 整合模型参数
    parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)
    # 管理文件
    agromanagement21 = YAMLAgroManagementReader(os.path.join(data_dir, 'agro', 'maize_2021.agro'))
    agromanagement22 = YAMLAgroManagementReader(os.path.join(data_dir, 'agro', 'maize_2022.agro'))
    # 气象数据
    weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')
    wdp = ExcelWeatherDataProvider(weatherfile)
    # 敏感性分析
    with open(os.path.join(data_dir, 'yaml', 'params.yaml'), 'r', encoding='utf-8') as f:
        problem_yaml = f.read()

    problem = yaml.safe_load(problem_yaml)

    calc_second_order = True
    nsamples = 2 ** 6
    paramsets = sobolSample.sample(problem, nsamples, calc_second_order=calc_second_order, seed=666)
    print("We are going to do %s simulations" % len(paramsets))
    # 修改作物参数用
    param_dict = {
        "SLATB": {"SLATB1": [0.00, 1], "SLATB2": [0.50, 3], "SLATB3": [0.78, 5], "SLATB4": [2.00, 7]},
        "KDIFTB": {"KDIFTB1": [0.0, 1], "KDIFTB2": [2.0, 3]},
        "EFFTB": {"EFFTB1": [0.0, 1], "EFFTB2": [40.0, 3]},
        "AMAXTB": {"AMAXTB1": [0.00, 1], "AMAXTB2": [1.25, 3], "AMAXTB3": [1.50, 5], "AMAXTB4": [1.75, 7], "AMAXTB5": [2.00, 9]},
        "TMPFTB": {
            "TMPFTB1": [0.00, 1],
            "TMPFTB2": [4.00, 3],
            "TMPFTB3": [16.00, 5],
            "TMPFTB4": [18.00, 7],
            "TMPFTB5": [20.00, 9],
            "TMPFTB6": [28.00, 11],
            "TMPFTB7": [36.00, 13],
            "TMPFTB8": [42.00, 15]
        },
        "TMNFTB": {"TMNFTB1": [8.00, 3]},
        "RFSETB": {
            "RFSETB1": [1.75, 5],
            "RFSETB2": [2.00, 7],
        },
        "FRTB": {
            "FRTB1": [0.00, 1],
            "FRTB2": [0.70, 15],
            "FRTB3": [0.90, 19]
        },
        "FLTB": {
            "FLTB1": [0.00, 1],
            "FLTB2": [0.33, 3],
            "FLTB3": [0.88, 5],
            "FLTB4": [1.10, 9],
        },
        "FOTB": {"FOTB1": [1.25, 7]},
        "RDRRTB": {
            "RDRRTB1": [1.5001, 5],
            "RDRRTB2": [2.00, 7]
        },
        "RDRSTB": {
            "RDRSTB1": [1.5001, 5],
            "RDRSTB2": [2.00, 7],
        }
    }


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

    nsets = len(paramsets)
    result_pp = create_dict(das)
    result_wlp = create_dict(das)
    pbar = PrintProgressBar(nsets, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█', printEnd="\r")
    pbar.printProgressBar(0)
    # printProgressBar(0, nsets, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i, paramset in enumerate(paramsets):
        parameters.clear_override()
        ## 修改敏感参数值
        for name, value in zip(problem["names"], paramsets[i]):
            try:
                parameters.set_override(name, value)
            except PCSEError:
                tmp_n = name[: -1]
                tem_d1 = param_dict[tmp_n]
                tmp_list = tem_d1[name]
                tmp_value = parameters[tmp_n]
                try:
                    tmp_value[tmp_list[1] -1], tmp_value[tmp_list[1]] = tmp_list[0], value
                except IndexError:
                    tmp_value.extend([tmp_list[0], value])
                    # print(tmp_value)
                parameters.set_override(tmp_n, tmp_value)
        ## 模型运算
        # 潜在产量数据
        tmp_res = None
        try:
            wofostpp = Wofost72_PP(parameters, wdp, agromanagement21)
            wofostwlp = Wofost72_WLP_FD(parameters, wdp, agromanagement21)
            result_pp = get_wofost_output(wofostpp, final_target, result_pp, pid=i, var_list=(das, time_target))
            result_wlp = get_wofost_output(wofostwlp, final_target, result_wlp, pid=i, var_list=(das, time_target))

        except ZeroDivisionError as e:
            print(e)

        if (i + 1) % 50 == 0 or (i + 1) == nsets:
            with open("./data/pickleFile/result_pp2021.pkl", "wb") as f:
                pickle.dump(result_pp, f)
            with open("./data/pickleFile/result_wlp2021.pkl", "wb") as f:
                pickle.dump(result_wlp, f)
        pbar.printProgressBar(i+1)


    result_pp["names1"] = col1
    result_pp["names2"] = col2
    result_wlp["names1"] = col1
    result_wlp["names2"] = col2

    with open("./data/pickleFile/result_pp2021.pkl", "wb") as f:
        pickle.dump(result_pp, f)
    with open("./data/pickleFile/result_wlp2021.pkl", "wb") as f:
        pickle.dump(result_wlp, f)


