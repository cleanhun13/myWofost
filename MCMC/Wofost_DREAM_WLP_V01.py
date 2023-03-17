import sys, os
import datetime as dt
import pcse
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader, YAMLAgroManagementReader
from pcse.exceptions import PCSEError, PartitioningError
from pcse.util import WOFOST80SiteDataProvider, WOFOST72SiteDataProvider
from pcse.base import ParameterProvider
from pcse.fileinput import YAMLAgroManagementReader
from pcse.fileinput import ExcelWeatherDataProvider
from pcse.models import Wofost80_NWLP_FD_beta as Wofost80_NWLP_FD
from pcse.models import Wofost72_WLP_FD
from tqdm import tqdm
import pandas as pd
import numpy as np
from wofostTool import overwrite_by_frac, isdir_demo, my_agro
import datetime as dt

from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm, uniform
import inspect
import os.path
import pickle

pydream_path = os.path.dirname(inspect.getfile(run_dream))
data_dir = os.path.join("F:\\paper_code\\wofost\\data")

try:
    tabName = sys.argv[1]
    nn = sys.argv[2]
    if tabName is None:
        tabName = "ZDN180"
    if nn is None:
        nn = 180

except:
    tabName = "ZDN180"
    nn = 180
nn = int(nn)
year = "2020"
save_dir = os.path.join("F:\\paper_code\\wofost\\MCMC", tabName + year)
isdir_demo(save_dir)

# print(tabName, nn)
# input("pause: ")
## 验证数据集
def get_LAI(table_name):
    yield_dict = {
        "ZDN180": [9607, 8426],
        "YDN180": [10264, 9217],
        "QSN180": [10757, 9715],
        "ZDN0": [6634, 5864],
        "QSN0": [7135, 6610],
        "YDN0": [6931, 6421],
        "ZDN90": [8288, 7397],
        "YDN90": [8794, 8426],
        "QSN90": [9373, 8567],
        "ZDN270": [9573, 8346],
        "YDN270": [10083.24, 9152],
        "QSN270": [10264.95, 9586],
    }

    lai_dir = os.path.join(data_dir, "LAI/STDLAI/")
    obs_LAI = pd.read_csv(os.path.join(lai_dir, f"{table_name}Y2021_LAI_OBS.csv"))

    obs_LAI.index = pd.to_datetime(obs_LAI.day)
    obs_LAI.drop("day", axis=1, inplace=True)
    obs_LAI1 = pd.read_csv(os.path.join(lai_dir, f"{table_name}Y2022_LAI_OBS.csv"))
    obs_LAI1.index = pd.to_datetime(obs_LAI1.day)
    obs_LAI1.drop("day", axis=1, inplace=True)

    obs_data = [[obs_LAI, obs_LAI1], yield_dict[table_name]]
    return obs_data

[lai_list, yield_list] = get_LAI(tabName)

## WOFOST模型初始化参数
# data_dir = os.path.join("F://paper_code//wofost", "data")
cropfile = os.path.join(data_dir, 'npkfile', 'wofost_npk.crop')
cropd = CABOFileReader(cropfile)
soilfile = os.path.join(data_dir, 'soil', 'ec3_copy1.soil')
soild = CABOFileReader(soilfile)

sited = WOFOST72SiteDataProvider(WAV=18)
global parameters
parameters = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)

with open(os.path.join(data_dir, "npkfile/wofost_npk2020.agro"), 'r') as f:
    yaml_agro_2020 = f.read()
with open(os.path.join(data_dir, "npkfile/wofost_npk2022.agro"), 'r') as f:
    yaml_agro_2022 = f.read()
with open(os.path.join(data_dir, "npkfile/wofost_npk2021.agro"), 'r') as f:
    yaml_agro_2021 = f.read()

weatherfile = os.path.join(data_dir, 'meteo', 'WOFOSTYL.xlsx')
wdp = ExcelWeatherDataProvider(weatherfile)

agro_list = [my_agro(yaml_agro_2021, nn), my_agro(yaml_agro_2022, nn)]

## 观测数据概率分布
# like_doa = norm(loc=doa, scale=1.5)
# like_dom = norm(loc=dom, scale=1.5)

## 估计参数采样设计
ori_params = {
    "TSUMEM": 92.35, "SLATB001":0.0026, "TDWI": 50.0, 
    "TSUM1": 1192.58, "SPAN": 36.5, "FLTB003": 0.62,
    "TSUM2": 788.15, "CVO": 0.671, "CVL": 0.680, 
    "EFFTB001": 0.45, "EFFTB003": 0.45,
    "TMNFTB003": 1.0 
}

# 参数分布设置
# TSUMEM = SampledParam(norm, loc=1, scale=0.09)
# TSUM1 = SampledParam(norm, loc=1, scale=0.019)
# TSUM2 = SampledParam(norm, loc=1, scale=0.03)
# EFFTB001 = SampledParam(uniform, loc=0.85, scale=0.5)
EFFTB003 = SampledParam(norm, loc=1, scale=0.1)
TMNFTB003 = SampledParam(norm, loc=1, scale=0.1)
SLATB001 = SampledParam(norm, loc=1, scale=0.1)
TDWI = SampledParam(norm, loc=1, scale=0.1)
SPAN = SampledParam(norm, loc=1, scale=0.1)
CVO = SampledParam(norm, loc=1, scale=0.1)
CVL = SampledParam(norm, loc=1, scale=0.1)
FLTB003 = SampledParam(norm, loc=1, scale=0.1)
sampled_parameter_names = [ 
                           "EFFTB003", 
                           "TMNFTB003", "SLATB001", 
                           "TDWI", "SPAN", "CVO", 
                           "CVL", "FLTB003"
                           ]

sampled_parameters = [EFFTB003, TMNFTB003, SLATB001, TDWI, SPAN, CVO, CVL, FLTB003]


nchains = 8
niterations = 10000

def likelihood(parameter_vector):
    param_dict = {pname: pvalue for pname, pvalue in zip(sampled_parameter_names, parameter_vector)}
    total_logp = 0
    agro, lai_obs, yield_obs = agro_list[0], lai_list[0], yield_list[0]
    params = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)
    # 设置物候参数
    params.set_override("TSUMEM", 92.35)
    params.set_override("TSUM1", 1192.58)
    params.set_override("TSUM2", 788.15)
    parameters = overwrite_by_frac(params, param_dict, ori_params)
    try:
        wofostmodel = Wofost72_WLP_FD(parameters, wdp, agro)
        wofostmodel.run_till_terminate()
    except:
        return -np.inf
    
    result = pd.DataFrame(wofostmodel.get_output())
    result.index = pd.to_datetime(result.day)
    result.drop("day", axis=1, inplace=True)

    twso = wofostmodel.get_summary_output()[0]["TWSO"]

    try:
        # 产量似然
        log_ps_yield = -0.5 * np.log(2*np.pi)-np.log(yield_obs*0.08) - 0.5*((yield_obs-twso)**2/(yield_obs*0.08)**2)
        # 叶面积似然
        diff = lai_obs.LAI - result.LAI
        diff.dropna(inplace=True)
        (n, ) = diff.shape
        log_ps_lai = -0.5*n*np.log(2*np.pi) - np.sum(np.log((lai_obs.LAI)*0.05)) - 0.5*np.sum((diff/((lai_obs.LAI)*0.05))**2)

        total_logp = log_ps_yield + log_ps_lai

    except Exception:
        total_logp += np.nan

    if np.isnan(total_logp):
        total_logp = -np.inf

    return total_logp


if __name__ == "__main__":
    converged = False
    total_iterations = niterations
    sampled_params, log_ps = run_dream(parameters=sampled_parameters, likelihood=likelihood, niterations=niterations, nchains=nchains,
                                       multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='wofost_dreamzs_5chain', verbose=True,)
    # Save sampling output (sampled parameter values and their corresponding logps).
    pre_name = f'wofost_dreamzs_{nchains}chain_'
    for chain in range(len(sampled_params)):
        pth = os.path.join(save_dir, f"{pre_name}sampled_params_chain_{chain}_{total_iterations}_{tabName}")
        pth1 = os.path.join(save_dir, f"{pre_name}logps_chain_{chain}_{total_iterations}_{tabName}")
        np.save(pth, sampled_params[chain])
        np.save(pth1, log_ps[chain])
    
    #Check convergence and continue sampling if not converged
    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    # np.savetxt('Wofost_dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+ tabName +'.txt', GR)

    old_samples = sampled_params
    if np.any(GR>1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations
            sampled_params, log_ps = run_dream(parameters=sampled_parameters, likelihood=likelihood,
                                               niterations=niterations, nchains=nchains, start=starts, multitry=False, gamma_levels=4,
                                               adapt_gamma=True, history_thin=1, model_name='wofost_dreamzs_5chain',
                                               verbose=True, restart=True)
            
            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                pth = os.path.join(save_dir, f"{pre_name}sampled_params_chain_{chain}_{total_iterations}_{tabName}")
                pth1 = os.path.join(save_dir, f"{pre_name}logps_chain_{chain}_{total_iterations}_{tabName}")
                np.save(pth, sampled_params[chain])
                np.save(pth1, log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ',total_iterations,' GR = ',GR)
            # np.savetxt('wofost_dreamzs_5chain_GelmanRubin_iteration_' +
            #            str(total_iterations) + tabName+'.txt', GR)

            if np.all(GR<1.2):
                converged = True
    with open(os.path.join(save_dir, f"./old_samples_{tabName}.pkl"), "wb") as f:
        pickle.dump(old_samples, f)

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(old_samples[0])
        burnin = total_iterations/2
        burnin = int(burnin)

        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :],
                                 old_samples[2][burnin:, :], old_samples[3][burnin:, :], old_samples[4][burnin:, :], 
                                 old_samples[5][burnin:, :], old_samples[6][burnin:, :], old_samples[7][burnin:, :],))

        ndims = len(sampled_parameter_names)
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
            fig.savefig(os.path.join(save_dir, 'PyDREAM_example_CORM_dimension_'+str(dim) + tabName) )

    except ImportError:
        pass

else:

    run_kwargs = {'parameters': sampled_parameters, 'likelihood': likelihood, 'niterations': niterations, 'nchains': nchains,
                  'multitry': False, 'gamma_levels': 4, 'adapt_gamma': True, 'history_thin': 1, 'model_name': 'wofost_dreamzs_5chain', 'verbose': False}



    




