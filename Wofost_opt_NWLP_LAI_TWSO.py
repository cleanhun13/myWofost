import yaml
import pcse
#  import sys
import os
import pickle
import copy
import nlopt
import hyperopt

from hyperopt import hp, fmin, tpe, Trials, partial, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader
from pcse.exceptions import PCSEError, PartitioningError
from pcse.util import WOFOST80SiteDataProvider
from pcse.base import ParameterProvider
from pcse.fileinput import YAMLAgroManagementReader
from pcse.fileinput import ExcelWeatherDataProvider
from pcse.models import Wofost80_NWLP_FD_beta as Wofost80_NWLP_FD

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

        "AMAXTB": [0.0, 70.0,
                   1.25, 70.0,
                   1.5, 63.0,
                   1.75, 49.0,
                   2.0, 21.0],

        "TMPFTB": [0.00, 0.01,
                   9.00, 0.05,
                   16.0, 0.8,
                   18.0, 0.94,
                   20.0, 1.0,
                   30.0, 1.0,
                   36.0, 0.95,
                   42.0, 0.56],

        "TMNFTB": [5.0, 0.0, 8.0, 1.0],

        "RFSETB": [0.0, 1.0,
                   1.5, 1.0,
                   1.75, 0.75,
                   2.0, 0.25],

        "SLATB": [0.0, 0.0026,
                  0.78, 0.001,
                  2.00, 0.0012],

        "KDIFTB": [0.0, 0.6, 2.0, 0.6],

        "EFFTB": [0.0, 0.45, 40.0, 0.45],

        "FLTB": [0.00, 0.62,
                 0.33, 0.62,
                 0.88, 0.15,
                 0.95, 0.15,
                 1.10, 0.10,
                 1.20, 0.00,
                 2.00, 0.00],

        "FOTB": [0.00, 0.00,
                 0.33, 0.00,
                 0.88, 0.00,
                 0.95, 0.00,
                 1.10, 0.50,
                 1.20, 1.00,
                 2.00, 1.00],

        "FSTB": [0.00, 0.38,
                 0.33, 0.38,
                 0.88, 0.85,
                 0.95, 0.85,
                 1.10, 0.40,
                 1.20, 0.00,
                 2.00, 0.00],
        "NMAXLV_TB": [0.0, 0.06,
                      0.4, 0.04,
                      0.7, 0.03,
                      1.0, 0.02,
                      2.0, 0.014,
                      2.1, 0.014,]
        }

    return mycropd


class ModelRerunner(object):

    def __init__(self, params, wdp, agro):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.summary = None

    def __call__(self, par_values, flag=False):
        # 重设输入参数
        self.params.clear_override()
        self.params.set_override("TSUMEM", 125.0)
        self.params.set_override("TSUM1", 1300)
        self.params.set_override("TSUM2", 720)
        crop_dict = my_crop_dict()

        for parname, value in par_values.items():

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
        wofostwlp = Wofost80_NWLP_FD(self.params, self.wdp, self.agro)
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

    def __call__(self, params1):

        # Run the model and collect output
        re_lai = list()
        re_yield = list()
        for nn in [90, 180]:
            obs_yield = self.observations[1][str(nn)]
            for i in range(2):
                agro1 = my_agro(self.agro[i], nn)
                modelrerunner = ModelRerunner(self.params, self.wdp, agro1)
                df_simulations = modelrerunner(params1, flag=True)
                # compute the differences by subtracting the DataFrames
                # Note that the dataframes automatically join on the index (dates) and column names
                if nn == 180:
                    df_simulations[0].LAI = (df_simulations[0].LAI - 0.09126) / (5.811 - 0.09126)
                    obs_lai = self.observations[0][i]
                    df_differences = np.abs(df_simulations[0].LAI - obs_lai.LAI)
                    re_lai.append(np.mean(df_differences))
                sim_yield = df_simulations[1][0]["TWSO"]
                sim_yield = (sim_yield - 6270) / (11892 - 6270)
                sim_yield = np.abs(sim_yield - obs_yield[i])
                re_yield.append(sim_yield)
        object_fun = 0.3 * np.mean(re_lai) + 0.7 * np.mean(re_yield)
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
    sited = WOFOST80SiteDataProvider(WAV=18, NAVAILI=20.5, PAVAILI=50.0, KAVAILI=250.0)

    # 合并数据
    parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)

    # 管理文件
    with open("./data/npkfile/wofost_npk2022.agro", 'r') as f:
        yaml_agro_2022 = f.read()
    with open("./data/npkfile/wofost_npk2021.agro", 'r') as f:
        yaml_agro_2021 = f.read()

    # 气象数据
    weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')
    wdp = ExcelWeatherDataProvider(weatherfile)

    # 管理文件数据
    agro = my_agro(yaml_agro_2022, 180)
    agro_list = (yaml_agro_2021, yaml_agro_2022)
    # bounds = {
    #     "SLATB001": np.arange(0.0013, 0.0039, 0.0001),
    #     "SLATB003": np.arange(0.0005, 0.0015, 0.0001),
    #     "SPAN": np.arange(16.5, 50.5, 0.5),
    #     "EFFTB003": np.arange(0.225, 0.675, 0.001),
    #     "TMNFTB003": np.arange(0.5, 1.2, 0.1),
    #     "CVO": np.arange(0.4955, 0.9065, 0.0001),
    #     "FLTB001": np.arange(0.465, 0.775, 0.001),
    #     "TDWI": np.arange(25, 80, 1),
    #     "CVS": np.arange(0.329, 0.987, 0.001),
    #     "EFFTB001": np.arange(0.25, 0.75, 0.01),
    #     "KDIFTB003": np.arange(0.30, 0.90, 0.01),
    #     "NCRIT_FR": np.arange(0.50, 1.50, 0.01),
    #     "AMAXTB001": np.arange(35.0, 90.0, 2),
    #     "NMAXLV_TB003": np.arange(0.01, 0.09, 0.005),
    #     "NMAXLV_TB007": np.arange(0.01, 0.09, 0.005),
    #     "NRESIDLV": np.arange(0.0020, 0.0080, 0.0001)
    #
    # }

    fspace = {
        "NAVAILI": hp.uniform("NAVAILI", 20.5, 60),
        "SLATB001": hp.uniform("SLATB001", 0.0013, 0.0039),
        "SLATB003": hp.uniform("SLATB003", 0.0005, 0.0015),
        "SPAN": hp.uniform("SPAN", 16.5, 50.5),
        "EFFTB003": hp.uniform("EFFTB003", 0.225, 0.675),
        "TMNFTB003": hp.uniform("TMNFTB003", 0.5, 1.2),
        "CVO": hp.uniform("CVO", 0.4955, 0.9065),
        "FLTB001": hp.uniform("FLTB001", 0.465, 0.775),
        "TDWI": hp.uniform("TDWI", 25, 80),
        "CVS": hp.uniform("CVS", 0.329, 0.987),
        "EFFTB001": hp.uniform("EFFTB001", 0.25, 0.75),
        "KDIFTB003": hp.uniform("KDIFTB003", 0.30, 0.90),
        "NCRIT_FR": hp.uniform("NCRIT_FR", 0.50, 1.50),
        "AMAXTB001": hp.uniform("AMAXTB001", 35.0, 90.0),
        "NMAXLV_TB003": hp.uniform("NMAXLV_TB003", 0.01, 0.09),
        "NMAXLV_TB007": hp.uniform("NMAXLV_TB007", 0.01, 0.09),
        "NRESIDLV": hp.uniform("NRESIDLV", 0.0020, 0.0080)

    }

    # 观测数据
    # yield_dict = {
    #     "ZDN180": np.array([8440, 10907]),
    #     "YDN180": np.array([9230, 10387]),
    #     "QSN180": np.array([8541, 11480])
    # }

    yield_dict = {
        "ZDN180": {"180": [0.593707, 0.383493419], "90": [0.358947, 0.200462469], "0": [0.09356101, 0.000711491]},
        "YDN180": {"180": [0.710423337, 0.524190679], "90": [0.448950551, 0.383493419], "0": [0.117573817, 0.026858769]},
        "QSN180": {"180": [0.79811455, 0.612771256], "90": [0.551938812, 0.408573461], "0": [0.153859836, 0.060476699]}
    }

    file_names = ["ZDN180", "YDN180", "QSN180"]
    for file_name in file_names:

        obs_LAI = pd.read_csv(os.path.join(data_dir, "LAI", "2022", f"{file_name}.csv"))
        obs_LAI.index = pd.to_datetime(obs_LAI.day)
        obs_LAI.drop("day", axis=1, inplace=True)

        obs_LAI1 = pd.read_csv(os.path.join(data_dir, "LAI", "2021", f"{file_name}.csv"))
        obs_LAI1.index = pd.to_datetime(obs_LAI1.day)
        obs_LAI1.drop("day", axis=1, inplace=True)

        obs_data = [[obs_LAI1, obs_LAI], yield_dict[file_name]]

        objfunc_calculator = None

        objfunc_calculator = ObjectiveFunctionCalculatorYield1(parameters, wdp, agro_list, obs_data)
        trials = Trials()
        best = fmin(fn=objfunc_calculator, space=fspace, algo=tpe.suggest, max_evals=5000, trials=trials)
        print("best: ", best)
        print("trials:")
        col_name = ["rmse"]
        for key_ in fspace.keys():
            col_name.append(key_)
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
        df_res.to_csv(os.path.join(data_dir, "opt", "opt1", f"./opt_{file_name}_NWLP_result3.csv"), header=None)


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

