import os
from pcse.base import ParameterProvider
import yaml
import copy


def isdir_demo(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            os.makedirs(dir_path)


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


def overwrite_by_frac(obj_param, param_dict, param_ori_dict):
    """
    Overwrite the parameter dictionary with the values in the parameter dictionary
    :param obj_param: Wofost Parameters object
    :param param_dict: Parameter dictionary
    :return: Wofost Parameters object
    """
    obj_param.clear_override()
    # params.set_override("TSUMEM", 125.0)
    # params.set_override("TSUM1", 1300)
    # params.set_override("TSUM2", 720)
    crop_dict = my_crop_dict()

    for parname, value in param_dict.items():
        
        ori_value = param_ori_dict[parname]

        tmp_name = parname.split("00")
        if len(tmp_name) == 2:
            var_name, idx1 = tmp_name[0], int(tmp_name[1])
            if var_name == "FLTB" or var_name == "FOTB":
                crop_dict[var_name][idx1] = value * ori_value
                crop_dict['FSTB'][idx1] = 1 - \
                    crop_dict['FLTB'][idx1] - crop_dict['FOTB'][idx1]
                obj_param.set_override(var_name, crop_dict[var_name])
                obj_param.set_override("FSTB", crop_dict["FSTB"])
                # print("%s: %s" % (var_name, parameters[var_name]))
            else:
                crop_dict[var_name][idx1] = value * ori_value
                obj_param.set_override(var_name, crop_dict[var_name])

        else:
            var_name = parname
            obj_param.set_override(var_name, value * ori_value)

    return obj_param


def overwrite_param1(obj_param, param_dict):
    """
    Overwrite the parameter dictionary with the values in the parameter dictionary
    :param obj_param: Wofost Parameters object
    :param param_dict: Parameter dictionary
    :return: Wofost Parameters object
    """
    crop_dict = my_crop_dict()
    DVS_name = ["SLATB", "KDIFTB", "EFFTB",
                "AMAXTB", "TMPFTB", "TMNFTB", "RFSETB"]
    for key, values in param_dict.items():
        
        if key in DVS_name:
            tmp_value = crop_dict[key]
            count = 0
            for item in tmp_value:
                if count % 2 == 1:
                    tmp_value[count] = item*values
                count += 1
            obj_param.set_override(key, tmp_value)

        elif key == "DVS":
            
            fotb, fstb, fltb = crop_dict["FOTB"], crop_dict["FSTB"], crop_dict["FLTB"]
            value1 = round(fotb[4] + values, 3)
            value2 = round(fotb[8] + values, 3)
            fotb[4] = round(value1, 3)
            fotb[8] = round(value2, 3)
            fstb[4] = round(value1, 3)
            fstb[8] = round(value2, 3)
            fltb[4] = round(value1, 3)
            fltb[8] = round(value2, 3)
            obj_param.set_override("FOTB", fotb)
            obj_param.set_override("FSTB", fstb)
            obj_param.set_override("FLTB", fltb)

        elif key == "alpha":
            # print(key)
            fotb, fstb, fltb = crop_dict["FOTB"], crop_dict["FSTB"], crop_dict["FLTB"]
            fltb[1] = fltb[1]*values
            fstb[1] = 1 - fltb[1]
            fltb[3] = fltb[3]*values
            fstb[3] = 1 - fltb[3]
            fltb[5] = fltb[5]*values
            fstb[5] = 1 - fltb[5]

            fotb[7] = fotb[7]*values
            fstb[7] = 1 - fotb[7]

            obj_param.set_override("FOTB", fotb)
            obj_param.set_override("FSTB", fstb)
            obj_param.set_override("FLTB", fltb)
        else:
            obj_param.set_override(key, values)

    return obj_param



def overwrite_para(params, param_dict):
    params.clear_override()
    # params.set_override("TSUMEM", 125.0)
    # params.set_override("TSUM1", 1300)
    # params.set_override("TSUM2", 720)
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


def my_agro(agro_yaml, n_amount):
    yaml_agro = copy.deepcopy(agro_yaml)
    n1 = n_amount * 0.6
    n2 = n_amount - n1
    yaml_agro = yaml_agro.replace("My_N1", str(n1))
    yaml_agro = yaml_agro.replace("My_N2", str(n2))

    return yaml.safe_load(yaml_agro)['AgroManagement']
