import copy
import yaml
import numpy as np
from pcse.fileinput import ExcelWeatherDataProvider, CABOFileReader
from pcse.exceptions import PCSEError, PartitioningError
import pcse
import sys, os
import pandas as pd
from pcse.util import WOFOST72SiteDataProvider
from pcse.base import ParameterProvider
from pcse.fileinput import YAMLAgroManagementReader
from pcse.models import Wofost72_PP, Wofost72_WLP_FD
import pickle
from SALib.analyze import fast
import time
from wofostSA1 import run_wofost


def isdir_demo(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            os.makedirs(dir_path)


def get_sample(file_path:str):
    with open(file_path, "r", encoding="utf-8") as f:
        problem_name, bounds1 = list(), list()
        tmp_list, res_list = list(), list()
        count = 0
        f.seek(0)
        flag = 0
        flag2 = 0
        for num, line in enumerate(f):
            if num == 1:
                total = int(line)
                continue
            elif num == 2:
                num_var = int(line)
                continue

            elif num > 2:

                try:
                    line = line.rstrip()
                    tmp_list = line.split('\t')
                    # print(len(tmp_list))
                    if len(tmp_list) == num_var and count < total:
                        res_list.append(tmp_list)
                        count += 1
                    else:
                        temp_str = f"{num_var} Distributions"
                        if temp_str in line:
                            start1 = num + 2
                            flag = 1
                            count1 = 0
                            count2 = 0
                        
                        if flag and count1 < num_var and num == start1 + count1*6:
                            start2 = num
                            problem_name.append(line)
                            count1 += 1
                            flag2 = 1

                        if flag2 and count2 < num_var and num == start2 + 3:
                            __tmp = line.split("\t")
                            print(__tmp)
                            bounds1.append([float(__tmp[0]), float(__tmp[1])])
                            count2 += 1

                except ValueError:
                    pass

    paramsets = np.array(res_list, dtype=np.float64)
    pro_ = dict()
    pro_["bounds"] = bounds1
    pro_["names"] = problem_name
    pro_["num_vars"] = len(bounds1)
    return (paramsets, pro_)


def my_agro(agro_yaml, n_amount):
    yaml_agro = copy.deepcopy(agro_yaml)
    n1 = n_amount * 0.6
    n2 = n_amount - n1
    yaml_agro = yaml_agro.replace("My_N1", str(n1))
    yaml_agro = yaml_agro.replace("My_N2", str(n2))
    
    return yaml.safe_load(yaml_agro)


def create_dict(key_das):
    res_dict = dict()
    res_dict["0"] = list()
    for each in key_das:
        dict_key = str(each)
        res_dict[dict_key] = list()
    return res_dict


def gen_res_dict(input_dict: dict):
    tmp_df = None
    for key_ in input_dict.keys():
        
        if key_.isnumeric():
            if key_ == "0":
                name = input_dict["names1"]
                tmp_df = pd.DataFrame(input_dict[key_], columns=input_dict["names1"])
            else:
                name = input_dict["names2"]
                tmp_df = pd.DataFrame(input_dict[key_], columns=input_dict["names2"])
            
            yield (key_, name, tmp_df)


def save_pickle(data, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
        print(f"{pkl_path}保存成功\n")


## 两日相隔天数
from datetime import datetime
def calDays(start, end, format="%Y-%m-%d"):
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return days


def main(sample_file, save_dir, n_amounts, flg1=False):
    data_dir = os.path.join(os.getcwd(), "data")

    # 作物参数
    cropfile = os.path.join(data_dir, 'crop', 'MAG202.CAB')
    cropd = CABOFileReader(cropfile)

    # 土壤参数
    soilfile = os.path.join(data_dir, 'soil', 'ec3_copy1.soil')
    soild = CABOFileReader(soilfile)

    # 站点数据
    sited = WOFOST72SiteDataProvider(WAV=18)

    # 整合参数
    parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)

    # 管理文件
    with open("./data/crop/agro2022.pkl", 'rb') as f:
        yaml_agro_2022 = pickle.load(f)
    with open("./data/crop/agro2021.pkl", 'rb') as f:
        yaml_agro_2021 = pickle.load(f)

    # 气象数据
    weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')
    wdp = ExcelWeatherDataProvider(weatherfile)

    final_target = ["LAIMAX", "TAGP", "TWSO"]
    time_target = ["LAI", "TAGP"]
    das = [i for i in range(126)]
    model_start_date = "2021-06-01"
    sowing_date = "2021-06-11"
    base_i = calDays(model_start_date, sowing_date)
    for idx, value_ in enumerate(das):
        das[idx] = base_i + value_

    col1 = ["LAIMAX", "TAGP", "TWSO", "id"]
    col2 = ["LAI", "TAGP", "id"]

    # 选择参数集
    # with open(os.path.join(data_dir, 'yaml', 'params2.yaml'), 'r', encoding='utf-8') as f:
    #     problem_yaml = f.read()
    # problem = yaml.safe_load(problem_yaml)

    # 读取数据样本
    
    paramsets, problem = get_sample(sample_file)

    nsets = len(paramsets)
    result_template = create_dict(das)
    # result_wlp = create_dict(das)
    result_template["names1"] = col1
    result_template["names2"] = col2
    # result_wlp["names1"] = col1
    # result_wlp["names2"] = col2

    
    
    for nn in n_amounts:
        parameters.clear_override()
        result_pp = copy.deepcopy(result_template)
        pklfile = "%sPPN%s2021.pkl" % (save_dir, nn)
        agromanagement21 = my_agro(yaml_agro_2021, nn)
        agromanagement22 = my_agro(yaml_agro_2022, nn)
        result_pp = run_wofost(variable_name=problem["names"], paramsets=paramsets, result_dict=result_pp, parameters=parameters,
                            agrodata=agromanagement21, weatherdata=wdp, pkl_file=pklfile, target_variables=final_target, target_list=(das, time_target), phenology=flg1)
        # parase_res_dict(result_2021pp, "./data/modelOut/2021PP/2021PP_s_.psc")
        save_pickle(result_pp, pklfile)
        pklfile = "%sPPN%s2022.pkl" % (save_dir, nn)
        result_pp = copy.deepcopy(result_template)
        result_pp = run_wofost(variable_name=problem["names"], paramsets=paramsets, result_dict=result_pp, parameters=parameters,
                                agrodata=agromanagement22, weatherdata=wdp, pkl_file=pklfile, target_variables=final_target, target_list=(das, time_target), phenology=flg1)
        save_pickle(result_pp, pklfile)
        # parase_res_dict(result_2022pp, "./data/modelOut/2022PP/2022PP_s_.psc")

    for nn in n_amounts:
        parameters.clear_override()
        result_wlp = copy.deepcopy(result_template)
        pklfile = "%sWLPN%s2021.pkl" % (save_dir, nn)
        agromanagement21 = my_agro(yaml_agro_2021, nn)
        agromanagement22 = my_agro(yaml_agro_2022, nn)

        result_wlp = run_wofost(variable_name=problem["names"], paramsets=paramsets, result_dict=result_wlp, parameters=parameters,
                                agrodata=agromanagement21, weatherdata=wdp, pkl_file=pklfile, target_variables=final_target, target_list=(das, time_target), method=1, phenology=flg1)
        save_pickle(result_pp, pklfile)
        # parase_res_dict(result_2021wlp, "./data/modelOut/2021WLP/2021WLP_s_.psc")
        # time1 = datetime.now()
        pklfile = "%sWLPN%s2022.pkl" % (save_dir, nn)
        result_wlp = copy.deepcopy(result_template)
        result_wlp = run_wofost(variable_name=problem["names"], paramsets=paramsets, result_dict=result_wlp, parameters=parameters,
                                agrodata=agromanagement22, weatherdata=wdp, pkl_file=pklfile, target_variables=final_target, target_list=(das, time_target), method=1, phenology=flg1)
        save_pickle(result_pp, pklfile)                      
        # parase_res_dict(result_2022wlp, "./data/modelOut/2022WLP/2022WLP_s_.psc")


if __name__ == "__main__":
    sample_files = ["./fastsample33.sam", "./fastsample33LAI.sam"]
    save_dirs = ["./data/modelOut/Final_new/", "./data/modelOut/LAI_new/"]
    for each in save_dirs:
        isdir_demo(each)
    flagg = [False, True]
    n_amounts = [180]
    for sample_file, save_dir, flg in zip(sample_files, save_dirs, flagg):
        main(sample_file=sample_file, save_dir=save_dir, n_amounts=n_amounts, flg1=flg)
