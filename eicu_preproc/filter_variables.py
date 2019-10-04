"""
Generate lists of selected variables, and produce some reports
"""

import os
import os.path
import datetime
import gc
import argparse

import scipy.stats as sp_stats
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import functions.util_io as mlhc_io

def filter_variables(configs):

    vital_variables = ["temperature", "sao2", "heartrate", "respiration", "cvp", "etco2", "systemicsystolic", "systemicdiastolic",
                       "systemicmean", "pasystolic", "padiastolic", "pamean", "st1", "st2", "st3", "icp"]

    vital_aper_variables = ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean", "paop", "cardiacoutput", "cardiacinput", "svr",
                            "svri", "pvr", "pvri"]

    ptable_path = os.path.join(configs["hdf_dir"], "patient.h5")
    periodic_path = os.path.join(configs["hdf_dir"], "vitalPeriodic.h5")
    aperiodic_path = os.path.join(configs["hdf_dir"], "vitalAperiodic.h5")
    lab_path = os.path.join(configs["hdf_dir"], "lab.h5")
    input_files = [periodic_path, aperiodic_path, lab_path]
    output_files = [configs["output_selected_per_vars"], configs["output_selected_aper_vars"], configs["output_selected_lab_vars"]]
    all_pids = list(map(int, mlhc_io.read_list_from_file(configs["included_pid_path"])))

    var_obs_count_dict = {}
    aper_var_obs_count_dict = {}
    lab_var_obs_count_dict = {}

    if configs["debug_mode"]:
        base_size = 1000
    else:
        base_size = len(all_pids)

    for pidx, pat in enumerate(all_pids):
        if (pidx+1) % 1000 == 0:
            print("Patient {}/{}".format(pidx+1, len(all_pids)))
            if configs["debug_mode"]:
                break

        df_periodic = pd.read_hdf(periodic_path, mode='r', where="patientunitstayid={}".format(pat))
        df_aperiodic = pd.read_hdf(aperiodic_path, mode='r', where="patientunitstayid={}".format(pat))
        df_lab = pd.read_hdf(lab_path, mode='r', where="patientunitstayid={}".format(pat))[["labname", "labresult"]].dropna()
        unique_lab_vars = list(map(lambda elem: elem.strip(), list(df_lab.labname.unique())))

        for var in unique_lab_vars:
            if var not in lab_var_obs_count_dict:
                lab_var_obs_count_dict[var] = 0
            lab_var_obs_count_dict[var] += 1

        for var in vital_variables:
            df_var = df_periodic[var].dropna()
            if df_var.shape[0] > 0:
                if var not in var_obs_count_dict:
                    var_obs_count_dict[var] = 0
                var_obs_count_dict[var] += 1

        for var in vital_aper_variables:
            df_var = df_aperiodic[var].dropna()
            if df_var.shape[0] > 0:
                if var not in aper_var_obs_count_dict:
                    aper_var_obs_count_dict[var] = 0
                aper_var_obs_count_dict[var] += 1

    non_selected_vars = []
    per_selected_vars = []

    for var in sorted(var_obs_count_dict.keys()):
        percentage = var_obs_count_dict[var]/base_size
        if percentage >= configs["required_var_freq"]:
            per_selected_vars.append(var)
        else:
            non_selected_vars.append(var)

    non_selected_vars = []
    aper_selected_vars = []

    for var in sorted(aper_var_obs_count_dict.keys()):
        percentage = aper_var_obs_count_dict[var]/base_size
        if percentage >= configs["required_var_freq"]:
            aper_selected_vars.append(var)
        else:
            non_selected_vars.append(var)

    non_selected_vars = []
    lab_selected_vars = []

    for var in sorted(lab_var_obs_count_dict.keys()):
        percentage = lab_var_obs_count_dict[var]/base_size
        if percentage >= configs["required_var_freq"]:
            lab_selected_vars.append(var)
        else:
            non_selected_vars.append(var)

    mlhc_io.write_list_to_file(configs["output_selected_per_vars"], per_selected_vars)
    mlhc_io.write_list_to_file(configs["output_selected_aper_vars"], aper_selected_vars)
    mlhc_io.write_list_to_file(configs["output_selected_lab_vars"], lab_selected_vars)

    for var in per_selected_vars:
        print("Analyzing variable: {}".format(var))
        df_var = pd.read_hdf(periodic_path, mode='r', columns=[var, "patientunitstayid"]).dropna()
        df_var = df_var[df_var["patientunitstayid"].isin(all_pids)][var]
        f, axarr = plt.subplots(2)
        lower_cutoff = np.percentile(np.array(df_var), 0.1)
        upper_cutoff = np.percentile(np.array(df_var), 99.9)
        df_var.plot.hist(bins=100, ax=axarr[0], log=True, range=(lower_cutoff, upper_cutoff))
        df_var.plot.box(ax=axarr[1], sym="", vert=False)
        plt.clf()
        gc.collect()
        lower = np.percentile(np.array(df_var), 25)-5*sp_stats.iqr(np.array(df_var))
        upper = np.percentile(np.array(df_var), 75)+5*sp_stats.iqr(np.array(df_var))
        normal = np.median(np.array(df_var))
        if configs["debug_mode"]:
            break

    for var in aper_selected_vars:
        print("Analyzing variable: {}".format(var))
        df_var = pd.read_hdf(aperiodic_path, mode='r', columns=[var, "patientunitstayid"]).dropna()
        df_var = df_var[df_var["patientunitstayid"].isin(all_pids)][var]
        f, axarr = plt.subplots(2)
        lower_cutoff = np.percentile(np.array(df_var), 0.1)
        upper_cutoff = np.percentile(np.array(df_var), 99.9)
        df_var.plot.hist(bins=100, ax=axarr[0], log=True, range=(lower_cutoff, upper_cutoff))
        df_var.plot.box(ax=axarr[1], sym="", vert=False)
        plt.clf()
        gc.collect()
        lower = np.percentile(np.array(df_var), 25)-5*sp_stats.iqr(np.array(df_var))
        upper = np.percentile(np.array(df_var), 75)+5*sp_stats.iqr(np.array(df_var))
        normal = np.median(np.array(df_var))
        if configs["debug_mode"]:
            break

    df_all_vars = pd.read_hdf(lab_path, mode='r', columns=["labname", "labresult", "patientunitstayid"]).dropna()
    df_all_vars = df_all_vars[df_all_vars["patientunitstayid"].isin(all_pids)]

    for var in lab_selected_vars:
        print("Analyzing variable: {}".format(var))
        df_var = df_all_vars[df_all_vars["labname"] == var]["labresult"]
        f, axarr = plt.subplots(2)
        lower_cutoff = np.percentile(np.array(df_var), 0.1)
        upper_cutoff = np.percentile(np.array(df_var), 99.9)
        df_var.plot.hist(bins=100, ax=axarr[0], range=(lower_cutoff, upper_cutoff), log=True)
        df_var.plot.box(ax=axarr[1], sym="", vert=False)
        plt.clf()
        gc.collect()
        lower = np.percentile(np.array(df_var), 25)-5*sp_stats.iqr(np.array(df_var))
        upper = np.percentile(np.array(df_var), 75)+5*sp_stats.iqr(np.array(df_var))
        normal = np.median(np.array(df_var))
        if configs["debug_mode"]:
            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--included_pid_path", default="../data/included_pid_stays.txt", help="Location where the included PIDs are saved")
    parser.add_argument("--hdf_dir", default="../data/hdf", help="Directory where HDF input files are located")  
    
    # Output paths
    parser.add_argument("--output_selected_per_vars", default="../data/included_per_variables.txt", help="Location to save the list of selected periodic variables")
    parser.add_argument("--output_selected_aper_vars", default="../data/included_aper_variables.txt", help="Location to save the list of selected aperiodic variables")
    parser.add_argument("--output_selected_lab_vars", default="../data/included_lab_variables.txt", help="Location to save the list of selected lab variables")

    # Parameters
    parser.add_argument("--debug_mode", action="store_true", default=False, help="Should debug mode be enabled?")
    parser.add_argument("--required_var_freq", type=float, default=0.1, help="What proportion of PIDs need to have a variable to be included?")    
    parser.add_argument("--output_base", default="variable_selection", help="Basename of the LaTeX report to generate")

    args = parser.parse_args()
    configs = vars(args)
    
    filter_variables(configs)
