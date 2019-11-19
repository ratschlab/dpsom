
import json
import gc
import argparse

import pandas as pd
import numpy as np

import functions.util_io as mlhc_io

def save_variable_quantiles(configs):
    ''' Saves the quantiles of all variables in the LAB/VITAL SIGN tables'''
    all_pids=mlhc_io.read_list_from_file(configs["included_pid_path"])
    vital_per_variables=mlhc_io.read_list_from_file(configs["list_per_variables"])
    vital_aper_variables=mlhc_io.read_list_from_file(configs["list_aper_variables"])
    var_quantiles={}

    print("Lab table...")
    df_lab=pd.read_hdf(configs["lab_table_path"],mode='r')
    df_lab=df_lab[df_lab.patientunitstayid.isin(all_pids)]
    print("Loaded lab table with {} rows".format(df_lab.shape[0]))
    all_lab_vars=df_lab.labname.unique()

    for lab_var in all_lab_vars:
        print("Lab variable: {}".format(lab_var))
        df_var= df_lab[df_lab.labname==lab_var]
        var_quantiles["lab_"+lab_var]=[]

        for quantile in np.arange(0.01,1.00,0.01):
            quant_val = df_var.labresult.quantile(quantile)
            var_quantiles["lab_"+lab_var].append(quant_val)

        print("List length: {}".format(len(var_quantiles["lab_"+lab_var])))
        
    gc.collect()
    print("Vital periodic table...")
    df_vital_per=pd.read_hdf(configs["vital_per_path"],mode='r')
    df_vital_per=df_vital_per[df_vital_per.patientunitstayid.isin(all_pids)]
    print("Loaded vital periodic table with {} rows".format(df_vital_per.shape[0]))

    for per_var in vital_per_variables:
        print("Periodic variable: {}".format(per_var))
        df_col=df_vital_per[per_var]
        var_quantiles["periodic_"+per_var]=[]

        for quantile in np.arange(0.01,1.00,0.01):
            quant_val=df_col.quantile(quantile)
            var_quantiles["periodic_"+per_var].append(quant_val)
                
        print("List length: {}".format(len(var_quantiles["periodic_"+per_var])))

    gc.collect()
    print("Vital aperiodic table...")
    df_vital_aper=pd.read_hdf(configs["vital_aper_path"],mode='r')
    df_vital_aper=df_vital_aper[df_vital_aper.patientunitstayid.isin(all_pids)]
    print("Loaded vital aperiodic table with {} rows".format(df_vital_aper.shape[0]))
    
    for aper_var in vital_aper_variables:
        print("Aperiodic variable: {}".format(aper_var))
        df_col=df_vital_aper[aper_var]
        var_quantiles["aperiodic_"+aper_var]=[]
        
        for quantile in np.arange(0.01,1.00,0.01):
            quant_val=df_col.quantile(quantile)
            var_quantiles["aperiodic_"+aper_var].append(quant_val)
                        
        print("List length: {}".format(len(var_quantiles["aperiodic_"+aper_var])))

    gc.collect()
    quantile_fp=open(configs["quantile_path"],mode='w')
    json.dump(var_quantiles,quantile_fp)
    quantile_fp.close()


if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--list_per_variables", default="../data/included_per_variables.txt", help="Periodic variables to process")
    parser.add_argument("--list_aper_variables", default="../data/included_aper_variables.txt", help="Aperiodic variables to process")
    parser.add_argument("--included_pid_path", default="../data/included_pid_stays.txt", help="Location where the included PIDs are saved")    
    parser.add_argument("--lab_table_path", default="../data/hdf/lab.h5", help="Lab table location")
    parser.add_argument("--vital_per_path", default="../data/hdf/vitalPeriodic.h5", help="Vital periodic table location")
    parser.add_argument("--vital_aper_path", default="../data/hdf/vitalAperiodic.h5", help="Vital aperiodic table location")

    # Output paths
    parser.add_argument("--quantile_path", default="../data/var_quantiles.json", help="JSON quantile dict")

    configs=vars(parser.parse_args())
    
    save_variable_quantiles(configs)
                                                                                                                                                                                                                                                                                                                                        
