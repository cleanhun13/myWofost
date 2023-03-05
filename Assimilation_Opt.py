import sys
import os
import copy
import datetime as dt
import pcse
from Wofost_optV1 import my_crop_dict, my_agro, ModelRerunner
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader, YAMLAgroManagementReader
from pcse.exceptions import PCSEError, PartitioningError
from hyperopt import hp, fmin, tpe, Trials, partial, STATUS_OK
from pcse.util import WOFOST80SiteDataProvider, WOFOST72SiteDataProvider
from pcse.base import ParameterProvider
from pcse.fileinput import YAMLAgroManagementReader
from pcse.fileinput import ExcelWeatherDataProvider
from pcse.models import Wofost80_NWLP_FD_beta as Wofost80_NWLP_FD
from pcse.models import Wofost72_WLP_FD
import pandas as pd
import numpy as np
import matplotlib


def overwrite_para(params, param_dict):
    params.clear_override()
    params.set_override("TSUMEM", 125.0)
    params.set_override("TSUM1", 1300)
    params.set_override("TSUM2", 720)
    crop_dict = my_crop_dict()

    for parname, value in param_dict.items():

        tmp_name = parname.split("00")
        if len(tmp_name) == 2:
            var_name, idx1 = tmp_name[0], int(tmp_name[1])
            if var_name == "FLTB" or var_name == "FOTB":
                crop_dict[var_name][idx1] = value
                crop_dict['FSTB'][idx1] = 1 - \
                    crop_dict['FLTB'][idx1] - crop_dict['FOTB'][idx1]
                params.set_override(var_name, crop_dict[var_name])
                params.set_override("FSTB", crop_dict["FSTB"])
                # print("%s: %s" % (var_name, parameters[var_name]))
            else:
                crop_dict[var_name][idx1] = value
                params.set_override(var_name, crop_dict[var_name])

        else:
            var_name = parname
            params.set_override(var_name, value)
    return params


def cal_ensembel_mean(df_list):

    num = 0
    for df in copy.deepcopy(df_list):

        df.reset_index(inplace=True)
        try:
            df.insert(loc=0, column="group", value=num)
        except ValueError:
            print("pass")
            pass
        if num == 0:
            concat_df = df
        else:
            concat_df = pd.concat([concat_df, df], axis=0)
        num += 1

    concat_df.reset_index(inplace=True, drop=True)
    concat_df = concat_df.groupby(["day"]).mean()

    return concat_df



def isdir_demo(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            os.makedirs(dir_path)


def add_gaussian_noise(data, data_size, mean=0., std=1.):
    return data + np.random.normal(mean, std, data_size)


def runEnKF(observations_for_DA, variables_for_DA, ensemble, 
            ensemble_size=50, show_output=False):
    
    while len(observations_for_DA):   
        day, obs = observations_for_DA.pop(0)
        for member in ensemble:
            member.run_till(day)
        # print("Ensemble now at day %s" % member.day)
        # print("%s observations left!" % len(observations_for_DA))

        collected_states = []
        for member in ensemble:
            t = {}
            for state in variables_for_DA:
                t[state] = member.get_variable(state)
            collected_states.append(t)
        df_A = pd.DataFrame(collected_states)
        A = np.matrix(df_A).T
        df_A if show_output else None

        P_e = np.matrix(df_A.cov())
        df_A.cov() if show_output else None

        perturbed_obs = []
        for state in variables_for_DA:
            (value, std) = obs[state]
            d = np.random.normal(value, std, (ensemble_size))
            perturbed_obs.append(d)
        df_perturbed_obs = pd.DataFrame(perturbed_obs).T
        df_perturbed_obs.columns = variables_for_DA
        D = np.matrix(df_perturbed_obs).T
        R_e = np.matrix(df_perturbed_obs.cov())
        df_perturbed_obs if show_output else None

        # Here we compute the Kalman gain
        H = np.identity(len(obs))
        K1 = P_e * (H.T)
        K2 = (H * P_e) * H.T
        K = K1 * ((K2 + R_e).I)
        K if show_output else None

        # Here we compute the analysed states
        Aa = A + K * (D - (H * A))
        df_Aa = pd.DataFrame(Aa.T, columns=variables_for_DA)
        df_Aa if show_output else None

        for member, new_states in zip(ensemble, df_Aa.itertuples()):
            r1 = member.set_variable("LAI", new_states.LAI)
            r2 = member.set_variable("TAGP", new_states.TAGP)


    for member in ensemble:
        member.run_till_terminate()

    results = []
    for member in ensemble:
        member_df = pd.DataFrame(member.get_output()).set_index("day")
        results.append(member_df)

    #集合均值
    ensembel_mean_df = cal_ensembel_mean(results)
    return ensembel_mean_df



class AssimilationObj:
    
    def __init__(self, param, wdp, agro, param_dict, df_model, obs_DA):
        self.param = param
        self.wdp = wdp
        self.agro = agro
        self.param_dict = param_dict
        self.df_model = df_model
        self.variables_for_DA = ["LAI", "TAGP"]
        self.obs_DA = obs_DA

    def get_observations_for_DA(self):
     
        dates_of_observation = self.obs_DA.index.to_list()
        observed_lai = self.obs_DA.LAI_ML.to_numpy()
        std_lai = observed_lai * 0.10 # Std. devation is estimated as 10% of observed value                                                                 
        observed_tagp = self.obs_DA.TAGP_ML.to_numpy()
        std_tagp = observed_tagp * 0.10 # Std. devation is estimated as 5% of observed value
        observations_for_DA = []
        # Pack them into a convenient format
        for d, lai, errlai, tagp, errtagp in zip(dates_of_observation, observed_lai, std_lai, observed_tagp, std_tagp):
            observations_for_DA.append((d, {"LAI":(lai, errlai), "TAGP":(tagp, errtagp)}))
        return observations_for_DA

    def __call__(self, std_dict):

        # 数据同化设置
        ensemble_size = 50
        np.random.seed(1354331612)

        # 高斯分布
        override_parameters = {}
        for key, value in self.param_dict.items():
            override_parameters[key] = add_gaussian_noise(value, (50, ), 0, std_dict[key]*value)
        
        # 初始化集合
        ensemble = []
        parameters1 = overwrite_para(self.param, self.param_dict)
        for i in range(ensemble_size):
            p = copy.deepcopy(parameters1)
            tmp_dict = dict()
            for par, distr in override_parameters.items():
                tmp_dict[par] = distr[i]
            p = overwrite_para(p, tmp_dict)
            member = Wofost72_WLP_FD(p, self.wdp, self.agro)
            ensemble.append(member)
        
        show_output = False

        ensembel_mean = runEnKF(self.get_observations_for_DA(), self.variables_for_DA, ensemble, ensemble_size, show_output)

        diff = ensembel_mean.LAI - obs_DA.LAI_ML
        obj_func = np.sqrt(np.mean(diff**2))
        obj_func = obj_func / np.mean(obs_DA.LAI_ML)

        diff = ensembel_mean.TAGP - obs_DA.TAGP_ML
        obj_func1 = np.sqrt(np.mean(diff**2))
        obj_func1 = obj_func1 / np.mean(obs_DA.TAGP_ML)

        return {'loss': np.mean([obj_func, obj_func1]), 'status': STATUS_OK}


if __name__ == '__main__':
    idx = 0
    global data_dir
    data_dir = os.path.join(os.getcwd(), "data")
    cropfile = os.path.join(data_dir, 'npkfile', 'wofost_npk.crop')
    cropd = CABOFileReader(cropfile)
    soilfile = os.path.join(data_dir, 'soil', 'ec3_copy1.soil')
    soild = CABOFileReader(soilfile)
    sited = WOFOST72SiteDataProvider(WAV=18)
    parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)

    with open("./data/npkfile/wofost_npk2022.agro", 'r') as f:
        yaml_agro_2022 = f.read()
    with open("./data/npkfile/wofost_npk2021.agro", 'r') as f:
        yaml_agro_2021 = f.read()
    
    weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')


    wdp = ExcelWeatherDataProvider(weatherfile)
    file_name = "ZDN180"
    # 模型参数读取
    param_df = pd.read_csv(os.path.join(data_dir, "opt", "opt_ZDN180_result.csv"), index_col=0)
    param_df.sort_values(by=['rmse'], ascending=True, inplace=True)
    p_name = param_df.columns.to_list()
    p_name = p_name[1:]

    rows, _ = param_df.shape
    df_dict = dict()
    for nn in [180]:
        agro = my_agro(yaml_agro_2021, nn)
        modelrerunner = None
        modelrerunner = ModelRerunner(parameters, wdp, agro)
        twso_list = list()
        for i in [idx]:
            p_value = dict()
            for each in p_name:
                p_value[each] = param_df.iloc[i][each]
            result = modelrerunner(p_value, flag=True)
            # if result[1][0]['TWSO'] > 8000:
            twso_list.append([i, result[1][0]['TWSO']])
        df1 = pd.DataFrame(twso_list, columns=["id", "TWSO"])
        df1.set_index("id", inplace=True)
        df_dict[str(nn)] = df1

        df_model = result[0]

    # 观测数据
    pathDA = "./data/data4DA/ZDN180Y2021.csv"
    obs_DA = pd.read_csv(pathDA)
    obs_DA.index = pd.to_datetime(obs_DA.day).dt.date
    obs_DA.drop("day", axis=1, inplace=True)
    obs_DA.dropna(axis=0, inplace=True, how="any")


    fspace = {}
    for each in p_name:
        fspace[each] = hp.uniform(each, 0.0, 0.20)

    assiObj = AssimilationObj(parameters, wdp, agro, p_value, df_model, obs_DA)

    trials = Trials()
    best = fmin(assiObj, fspace, algo=tpe.suggest, trials=trials, max_evals=1000)
    
    print(best)
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
    df_res.to_csv(os.path.join(data_dir, "opt", "opt1",
                    f"./opt_{file_name}_DA_result3.csv"), header=None)

