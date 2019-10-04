""" 
Imputation on eICU
"""

import json
import numpy as np
import pandas as pd

import functions.util_impute as eicu_impute
import functions.util_array as mlhc_array

class Timegridder():
    ''' Function transforming the input table from the eICU tables into imputed values'''

    def __init__(self, timegrid_step_mins=60.0):

        # List of selected vitalPeriodic variables
        self.sel_vs_vars = ["temperature", "sao2", "heartrate", "respiration", "cvp", "etco2", "systemicsystolic",
                            "systemicdiastolic", "systemicmean", "pasystolic", "padiastolic", "pamean",
                            "st1", "st2", "st3"]

        # List of selected vitalAperiodic variables
        self.sel_avs_vars = ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean", "paop", "cardiacoutput",
                             "cardiacinput", "svr", "svri", "pvr", "pvri"]

        # Time grid interval length in minutes
        self.timegrid_step_mins = timegrid_step_mins

        # Maximum forward filling time in seconds for a variable
        self.max_forward_fill_secs_vs = 3600
        self.max_forward_fill_secs_avs = 3600
        self.max_forward_fill_secs_lab = 24*3600

        self.create_pid_col = True

    def set_quantile_dict(self, quantile_dict):
        ''' Sets internal state with quantile dict'''
        self.var_quantile_dict=quantile_dict

    def set_selected_lab_vars(self, lab_vars):
        ''' Sets a list of selected lab variables'''
        self.lab_vars = lab_vars

    def transform(self, df_lab, df_vs, df_avs, pid=None):
        df_lab.sort_values(by="labresultoffset", inplace=True, kind="mergesort")
        df_vs.sort_values(by="observationoffset", inplace=True, kind="mergesort")
        hr_col = df_vs[["observationoffset", "heartrate"]].dropna()
        min_ts = hr_col.observationoffset.min()
        max_ts = hr_col.observationoffset.max()
        timegrid = np.arange(0.0, max_ts-min_ts, self.timegrid_step_mins)
        df_avs.sort_values(by="observationoffset", inplace=True, kind="mergesort")
        df_out_dict = {}
        df_out_dict["ts"] = timegrid

        if self.create_pid_col:
            df_out_dict["patientunitstayid"] = mlhc_array.value_empty(timegrid.size, int(pid))

        for var in self.sel_vs_vars:
            finite_df = df_vs[["observationoffset", var]].dropna()
            raw_ts = np.array(finite_df["observationoffset"])
            raw_values = np.array(finite_df[var])
            pred_values = eicu_impute.impute_variable(raw_ts, raw_values, timegrid, leave_nan_threshold_secs=self.max_forward_fill_secs_vs,
                                                      grid_period=self.timegrid_step_mins, normal_value=self.var_quantile_dict["periodic_"+var][49])
            df_out_dict["vs_{}".format(var)] = pred_values

        for var in self.sel_avs_vars:
            finite_df = df_avs[["observationoffset", var]].dropna()
            raw_ts = np.array(finite_df["observationoffset"])
            raw_values = np.array(finite_df[var])
            pred_values = eicu_impute.impute_variable(raw_ts, raw_values, timegrid, leave_nan_threshold_secs=self.max_forward_fill_secs_avs,
                                                      grid_period=self.timegrid_step_mins, normal_value=self.var_quantile_dict["aperiodic_"+var][49])
            df_out_dict["avs_{}".format(var)] = pred_values

        for var in self.lab_vars:
            normal_value = self.var_quantile_dict["lab_"+var][49]
            sel_df = df_lab[df_lab["labname"] == var]
            if sel_df.shape[0] == 0:
                pred_values = mlhc_array.value_empty(timegrid.size, normal_value)
            else:
                finite_df = sel_df[["labresultoffset", "labresult"]].dropna()
                raw_ts = np.array(finite_df["labresultoffset"])
                raw_values = np.array(finite_df["labresult"])
                pred_values = eicu_impute.impute_variable(raw_ts, raw_values, timegrid, leave_nan_threshold_secs=self.max_forward_fill_secs_lab,
                                                          grid_period=self.timegrid_step_mins, normal_value=normal_value)

            df_out_dict["lab_{}".format(var)] = pred_values

        df_out = pd.DataFrame(df_out_dict)
        return df_out
