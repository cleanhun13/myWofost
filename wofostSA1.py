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
from pcse.util import WOFOST80SiteDataProvider
from pcse.models import Wofost80_NWLP_FD_beta as Wofost80_NWLP_FD
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


def my_crop_dict():

    mycropd = {

        "AMAXTB": [0.0, 70.0, 1.25, 70.0, 1.5, 63.0, 1.75, 49.0, 2.0, 21.0],

        "TMPFTB": [0.00, 0.01, 9.00, 0.05, 16.0, 0.8, 18.0, 0.94, 20.0, 1.0, 30.0, 1.0, 36.0, 0.95, 42.0, 0.56],

        "TMNFTB": [5.0, 0.0, 8.0, 1.0],

        "RFSETB": [0.0, 1.0, 1.5, 1.0, 1.75, 0.75, 2.0, 0.25],

        "SLATB": [0.0, 0.0026, 0.78, 0.001, 2.00, 0.0012],

        "KDIFTB": [0.0, 0.6, 2.0, 0.6],

        "EFFTB": [0.0, 0.45, 40.0, 0.45],

        "FLTB": [0.00, 0.62, 0.33, 0.62, 0.88, 0.15, 0.95, 0.15, 1.10, 0.10, 1.20, 0.00, 2.00, 0.00],

        "FOTB": [0.00, 0.00, 0.33, 0.00, 0.88, 0.00, 0.95, 0.00, 1.10, 0.50, 1.20, 1.00, 2.00, 1.00],

        "FSTB": [0.00, 0.38, 0.33, 0.38, 0.88, 0.85, 0.95, 0.85, 1.10, 0.40, 1.20, 0.00, 2.00, 0.00]

        }

    return mycropd


def run_wofost(variable_name, paramsets: np.ndarray, result_dict, parameters, agrodata, weatherdata, pkl_file,
               target_variables, target_list=None, phenology=False):
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
    # bar_format = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]"
    ncols = 80

    with tqdm(total=total_row, ncols=ncols, postfix="Processing") as _tqdm:
        _tqdm.set_description("{}".format("完成进度"))
        for i, paramset in enumerate(paramsets):
            parameters.clear_override()
            ## 设置TSUM1 TSUM2
            if phenology:
                parameters.set_override("TSUMEM", 125.0)
                parameters.set_override("TSUM1", 1300)
                parameters.set_override("TSUM2", 720)
            ## 修改敏感参数值
            mycropd = my_crop_dict()

            for name, value in zip(variable_name, paramset):
                tmp_name = name.split("00")
                if len(tmp_name) == 2:
                    var_name, idx1 = tmp_name[0], int(tmp_name[1])
                    if var_name == "FLTB" or var_name == "FOTB":
                        mycropd[var_name][idx1] = value
                        mycropd['FSTB'][idx1] = 1 - mycropd['FLTB'][idx1] - mycropd['FOTB'][idx1]
                        parameters.set_override(var_name, mycropd[var_name])
                        parameters.set_override("FSTB", mycropd["FSTB"])
                        # print("%s: %s" % (var_name, parameters[var_name]))
                    else:
                        mycropd[var_name][idx1] = value
                        parameters.set_override(var_name, mycropd[var_name])

                else:
                    var_name = name
                    parameters.set_override(var_name, value)



            try:
                wofostpp = Wofost80_NWLP_FD(parameters, weatherdata, agrodata)
                result_dict = get_wofost_output(wofostpp, target_variables, result_dict, pid=i, var_list=target_list)
            except ZeroDivisionError as e:
                print(e)

            if (i + 1) % 100 == 0 or (i + 1) == total_row:
                with open(pkl_file, "wb") as f:
                    pickle.dump(result_dict, f)

            _tqdm.set_postfix(number = i)
            _tqdm.update(1)

    return result_dict


if __name__ == "__main__":
    print("test")