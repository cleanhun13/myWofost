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
from wofostTool import overwrite_para, isdir_demo, my_crop_dict, my_agro
import datetime as dt

from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm, uniform
import inspect
import os.path
import pickle

pydream_path = os.path.dirname(inspect.getfile(run_dream))


## 验证数据集
doa_list = [64, 61]
dom_list = [107, 102]
doe_list = [7, 6]

## WOFOST模型初始化参数
data_dir = os.path.join("F://paper_code//wofost", "data")
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

agro_list = [my_agro(yaml_agro_2021, 180), my_agro(yaml_agro_2022, 180)]

## 观测数据概率分布
# like_doa = norm(loc=doa, scale=1.5)
# like_dom = norm(loc=dom, scale=1.5)

## 估计参数采样设计
TSUEM = SampledParam(uniform, loc=65, scale=50)
TSUM1 = SampledParam(uniform, loc=500, scale=900)
TSUM2 = SampledParam(uniform, loc=400, scale=600)

sampled_parameter_names = ["TSUMEM", "TSUM1", "TSUM2"]

sampled_parameters = [TSUEM, TSUM1, TSUM2]

nchains = 5
niterations = 3000

def likelihood(parameter_vector):
    param_dict = {pname: pvalue for pname, pvalue in zip(sampled_parameter_names, parameter_vector)}
    total_logp = 0
    for agro, doe, doa, dom in zip(agro_list, doe_list, doa_list, dom_list):

        params = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)
        parameters = overwrite_para(params, param_dict)

        wofostmodel = Wofost72_WLP_FD(parameters, wdp, agro)
        wofostmodel.run_till_terminate()
        result = wofostmodel.get_summary_output()
        try:
            doe_s = (result[0]["DOE"]-result[0]["DOS"]).days
            doa_s = (result[0]["DOA"] - result[0]["DOS"]).days
            dom_s = (result[0]["DOM"] - result[0]["DOS"]).days

            total_logp += -1.5 * np.log(2*np.pi)-np.log(1.5)-np.log(1.5) - np.log(1.5) - 0.5*((doa_s-doa)**2)/(1.5**2) - 0.5*((dom_s-dom)**2)/(1.5**2) - 0.5*((doe_s-doe)**2)/(1.5**2)
        except TypeError:
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
    for chain in range(len(sampled_params)):
        np.save('wofost_dreamzs_5chain_sampled_params_chain_' +
                str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('wofost_dreamzs_5chain_logps_chain_' + str(chain) +
                '_'+str(total_iterations), log_ps[chain])
    
    #Check convergence and continue sampling if not converged
    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('Wofost_dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)

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
                np.save('wofost_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
                np.save('wofost_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ',total_iterations,' GR = ',GR)
            np.savetxt('wofost_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

            if np.all(GR<1.2):
                converged = True
    with open("./old_samples.pkl", "wb") as f:
        pickle.dump(old_samples, f)

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(old_samples[0])
        burnin = total_iterations/2
        burnin = int(burnin)

        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :],
                                 old_samples[2][burnin:, :], old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(sampled_parameter_names)
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
            fig.savefig('PyDREAM_example_CORM_dimension_'+str(dim))

    except ImportError:
        pass

else:

    run_kwargs = {'parameters': sampled_parameters, 'likelihood': likelihood, 'niterations': niterations, 'nchains': nchains,
                  'multitry': False, 'gamma_levels': 4, 'adapt_gamma': True, 'history_thin': 1, 'model_name': 'wofost_dreamzs_5chain', 'verbose': False}



    




