import yaml
import pcse
#  import sys
import os
import pickle
import copy
# import nlopt
import hyperopt

from hyperopt import hp, fmin, tpe, Trials, partial, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader
from pcse.exceptions import PCSEError, PartitioningError
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.fileinput import YAMLAgroManagementReader
from pcse.fileinput import ExcelWeatherDataProvider
from pcse.models import Wofost71_WLP_FD

import numpy as np
import pandas as pd

data_dir = os.path.join(os.getcwd(), "data")


def my_agro(agro_yaml, n_amount):
    yaml_agro = copy.deepcopy(agro_yaml)
    n1 = n_amount * 0.6
    n2 = n_amount - n1
    yaml_agro = yaml_agro.replace("My_N1", str(n1))
    yaml_agro = yaml_agro.replace("My_N2", str(n2))

    return yaml.safe_load(yaml_agro)['AgroManagement']


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


class ModelRerunner(object):
    parameters_name = ["SLATB001", "SPAN", "EFFTB003", "TMNFTB003", "CVO", "FLTB003", "TDWI", "CVL",
                       "TEFFMX", "EFFTB001", "KDIFTB003"]

    def __init__(self, params, wdp, agro):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.summary = None

    def __call__(self, par_values, flag=False):
        # 核对参数个数是否一致
        if len(par_values) != len(self.parameters_name):
            msg = "Optimizing %i parameters, but only %i values were provided!" % \
                  (len(self.parameters_name), len(par_values))
            raise RuntimeError(msg)
        # 重设输入参数
        self.params.clear_override()
        self.params.set_override("TSUMEM", 125.0)
        self.params.set_override("TSUM1", 1300)
        self.params.set_override("TSUM2", 720)
        crop_dict = my_crop_dict()

        for parname, value in zip(self.parameters_name, par_values):
            tmp_name = parname.split("00")
            if len(tmp_name) == 2:
                var_name, idx1 = tmp_name[0], int(tmp_name[1])
                if var_name == "FLTB" or var_name == "FOTB":
                    crop_dict[var_name][idx1] = value
                    crop_dict['FSTB'][idx1] = 1 - crop_dict['FLTB'][idx1] - crop_dict['FOTB'][idx1]
                    self.params.set_override(var_name, crop_dict[var_name])
                    self.params.set_override("FSTB", crop_dict["FSTB"])
                    # print("%s: %s" % (var_name, parameters[var_name]))
                else:
                    crop_dict[var_name][idx1] = value
                    self.params.set_override(var_name, crop_dict[var_name])

            else:
                var_name = parname
                self.params.set_override(var_name, value)

        # 运行模型
        wofostwlp = Wofost71_WLP_FD(self.params, self.wdp, self.agro)
        wofostwlp.run_till_terminate()
        df = pd.DataFrame(wofostwlp.get_output())
        self.summary = wofostwlp.get_summary_output()
        df.index = pd.to_datetime(df.day)
        df.drop("day", axis=1, inplace=True)
        if flag:
            tmp_re = [df, wofostwlp.get_summary_output()]
            return tmp_re
        else:
            return df


# class ObjectiveFunctionCalculator1(object):
#
#     def __init__(self, params, wdp, agro, observations):
#         self.modelrerunner = ModelRerunner(params, wdp, agro)
#         self.df_observations = observations
#         self.n_calls = 0
#
#     def __call__(self, par_values, grad=None):
#         self.n_calls += 1
#         print(".", end="")
#         # Run the model and collect output
#         self.df_simulations = self.modelrerunner(par_values)
#         # compute the differences by subtracting the DataFrames
#         # Note that the dataframes automatically join on the index (dates) and column names
#         df_differences = self.df_simulations - self.df_observations
#         # Compute the RMSE on the LAI column
#         obj_func = np.sqrt(np.mean(df_differences.LAI ** 2))
#         return obj_func


class ObjectiveFunctionCalculatorLAI(object):

    def __init__(self, params, wdp, agro, observations):
        self.modelrerunner = ModelRerunner(params, wdp, agro)
        self.df_observations = observations
        self.n_calls = 0

    def __call__(self, params1):

        par_values = [params1['SLATB001'], params1['SPAN'], params1['EFFTB003'], params1['TMNFTB003'], params1['CVO'],
                      params1['FLTB003'], params1['TDWI'], params1['CVL'], params1['TEFFMX'],
                      params1['EFFTB001'], params1['KDIFTB003']]
        # Run the model and collect output
        self.df_simulations = self.modelrerunner(par_values)
        # compute the differences by subtracting the DataFrames
        # Note that the dataframes automatically join on the index (dates) and column names
        df_differences = self.df_simulations - self.df_observations
        # Compute the RMSE on the LAI column
        obj_func = np.sqrt(np.mean(df_differences.LAI ** 2))
        return {'loss': obj_func, 'status': STATUS_OK}


class ObjectiveFunctionCalculatorYield(object):

    def __init__(self, params, wdp, agro_yaml, observations):
        # self.modelrerunner = ModelRerunner(params, wdp, agro)
        self.params = params
        self.wdp = wdp
        self.agro_list = []
        self.agro_list.append(my_agro(agro_yaml, 0))
        self.agro_list.append(my_agro(agro_yaml, 180))
        self.observations = observations
        self.n_calls = 0

    def __call__(self, params1):
        par_values = [params1['SLATB001'], params1['SPAN'], params1['EFFTB003'], params1['TMNFTB003'], params1['CVO'],
                      params1['FLTB003'], params1['TDWI'], params1['CVL'], params1['TEFFMX'],
                      params1['EFFTB001'], params1['KDIFTB003']]
        # Run the model and collect output
        self.res = list()

        for j in range(2):
            modelrerunner = ModelRerunner(self.params, self.wdp, self.agro_list[j])
            df_simulations = modelrerunner(par_values, flag=True)
            self.res.append(df_simulations[0]["TWSO"])
        # compute the differences by subtracting the DataFrames
        # Note that the dataframes automatically join on the index (dates) and column names
        df_differences = np.array(self.res) - self.observations
        # Compute the RMSE on the LAI column
        obj_func = np.sqrt(np.mean(df_differences ** 2))
        return {'loss': obj_func, 'status': STATUS_OK}


class ObjectiveFunctionCalculatorYield1(object):

    def __init__(self, params, wdp, agro, observations):
        # self.modelrerunner = ModelRerunner(params, wdp, agro)
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.observations = observations
        self.n_calls = 0
        self.modelrerunner = ModelRerunner(self.params, self.wdp, self.agro)

    def __call__(self, params1):
        par_values = [params1['SLATB001'], params1['SPAN'], params1['EFFTB003'], params1['TMNFTB003'], params1['CVO'],
                      params1['FLTB003'], params1['TDWI'], params1['CVL'], params1['TEFFMX'],
                      params1['EFFTB001'], params1['KDIFTB003']]
        # Run the model and collect output

        df_simulations = self.modelrerunner(par_values, flag=True)
        # compute the differences by subtracting the DataFrames
        # Note that the dataframes automatically join on the index (dates) and column names
        df_simulations[0].LAI = (df_simulations[0].LAI - 0.09126) / (5.811 - 0.09126)
        df_differences = np.abs(df_simulations[0].LAI - self.observations[0].LAI)
        object_fun = np.mean(df_differences)
        sim_yield = df_simulations[1][0]["TWSO"]
        sim_yield = (sim_yield - 6270) / (11892 - 6270)
        sim_yield = np.abs(sim_yield - self.observations[1])
        object_fun = 0.5 * object_fun + 0.5 * sim_yield
        # Compute the RMSE on the LAI column
        return {'loss': object_fun, 'status': STATUS_OK}


if __name__ == "__main__":
    # 读取作物参数
    cropfile = os.path.join(data_dir, 'npkfile', 'wofost_npk.crop')
    cropd = CABOFileReader(cropfile)

    # 土壤参数
    soilfile = os.path.join(data_dir, 'soil', 'ec3_copy1.soil')
    soild = CABOFileReader(soilfile)

    # 站点数据
    sited = WOFOST71SiteDataProvider(WAV=18, NAVAILI=250, PAVAILI=50.0, KAVAILI=250.0)

    # 合并数据
    parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)

    # 管理文件
    with open("./data/npkfile/wofost_npk2022.agro", 'r') as f:
        yaml_agro_2022 = f.read()
    with open("./data/npkfile/wofost_npk2022.agro", 'r') as f:
        yaml_agro_2021 = f.read()

    # 气象数据
    weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')
    wdp = ExcelWeatherDataProvider(weatherfile)

    # 管理文件数据
    agro = my_agro(yaml_agro_2022, 180)

    fspace = {
        "SLATB001": hp.uniform("SLATB001", 0.0013, 0.0039),
        "SPAN": hp.uniform("SPAN", 16.5, 50.5),
        "EFFTB003": hp.uniform("EFFTB003", 0.225, 0.675),
        "TMNFTB003": hp.uniform("TMNFTB003", 0.5, 1.2),
        "CVO": hp.uniform("CVO", 0.4955, 0.9065),
        "FLTB003": hp.uniform("FLTB003", 0.465, 0.675),
        "TDWI": hp.uniform("TDWI", 25, 80),
        "CVL": hp.uniform("CVL", 0.34, 0.92),
        "TEFFMX": hp.uniform("TEFFMX", 35, 45),
        "EFFTB001": hp.uniform("EFFTB001", 0.25, 0.75),
        "KDIFTB003": hp.uniform("KDIFTB003", 0.30, 0.90),

    }

    # 观测数据
    # yield_dict = {
    #     "ZDN180": np.array([8440, 10907]),
    #     "YDN180": np.array([9230, 10387]),
    #     "QSN180": np.array([8541, 11480])
    # }

    yield_dict = {
        "ZDN180": 0.436855212,
        "YDN180": 10387,
        "QSN180": 11480
    }

    file_names = ["ZDN180"]
    for file_name in file_names:
        obs_LAI = pd.read_csv(os.path.join(data_dir, "LAI", "2022", f"{file_name}.csv"))
        obs_LAI.index = pd.to_datetime(obs_LAI.day)
        obs_LAI.drop("day", axis=1, inplace=True)
        obs_data = [obs_LAI, yield_dict[file_name]]

        objfunc_calculator = None

        objfunc_calculator = ObjectiveFunctionCalculatorYield1(parameters, wdp, agro, obs_data)
        trials = Trials()
        best = fmin(fn=objfunc_calculator, space=fspace, algo=tpe.suggest, max_evals=10000, trials=trials)
        print("best: ", best)
        print("trials:")
        col_name = ["rmse", "SLATB001", "SPAN", "EFFTB003", "TMNFTB003", "CVO", "FLTB003", "TDWI", "CVL", "TEFFMX", "EFFTB001", "KDIFTB003"]
        opt_result = list([col_name])
        for trial in trials.trials:
            tmp_list = list()
            tmp_list.append(trial['result']['loss'])
            for parname1 in col_name[1:]:
                tmp_dict = trial["misc"]
                tmp_dict = tmp_dict['vals']

                tmp_list.append(tmp_dict[parname1][0])
            opt_result.append(tmp_list)
        df_res = pd.DataFrame(opt_result)
        df_res.to_csv(os.path.join(data_dir, "opt", f"./opt_{file_name}_result.csv"))


# 叶面积指数用
    # # 观测数据
    # file_names = ["ZDN180", "YDN180", "QSN180"]
    # for file_name in file_names:
    #     obs_data = pd.read_csv(os.path.join(data_dir, "LAI", "2022", f"{file_name}.csv"))
    #     obs_data.index = pd.to_datetime(obs_data.day)
    #     obs_data.drop("day", axis=1, inplace=True)
    #
    #     objfunc_calculator = None
    #
    #     objfunc_calculator = ObjectiveFunctionCalculator(parameters, wdp, agro, obs_data)
    #     trials = Trials()
    #     best = fmin(fn=objfunc_calculator, space=fspace, algo=tpe.suggest, max_evals=5000, trials=trials)
    #     print("best: ", best)
    #     print("trials:")
    #     col_name = ["rmse", "SLATB001", "SPAN", "EFFTB003", "TMNFTB003", "CVO", "FLTB003", "TDWI", "CVL", "TEFFMX"]
    #     opt_result = list([col_name])
    #     for trial in trials.trials:
    #         tmp_list = list()
    #         tmp_list.append(trial['result']['loss'])
    #         for parname1 in col_name[1:]:
    #             tmp_dict = trial["misc"]
    #             tmp_dict = tmp_dict['vals']
    #
    #             tmp_list.append(tmp_dict[parname1][0])
    #         opt_result.append(tmp_list)
    #     df_res = pd.DataFrame(opt_result)
    #     df_res.to_csv(os.path.join(data_dir, "opt", f"./opt_{file_name}_result.csv"))
    #     # print(trial)

