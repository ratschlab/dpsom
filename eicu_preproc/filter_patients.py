"""
Generates the list of includes patient stays.
"""

import ipdb
import argparse

import numpy as np
import pandas as pd

import functions.util_io as mlhc_io

def filter_patients(configs):
    pid_list = mlhc_io.read_list_from_file(configs["all_pid_stay_path"])
    print("Number of PID stays in the database: {}".format(len(pid_list)))
    included_patients = []

    for pidx, pid in enumerate(pid_list):
        if (pidx+1) % 100 == 0:
            print("PID: {}/{}".format(pidx+1, len(pid_list)))
            print("Included patient stays: {}/{}".format(len(included_patients), pidx+1))
        df_vs = pd.read_hdf(configs["vital_per_table_path"], mode='r', where="patientunitstayid={}".format(pid))
        df_vs.sort_values(by="observationoffset", inplace=True, kind="mergesort")
        hr_col = df_vs[["observationoffset", "heartrate"]].dropna()
        diffed_hr_col = hr_col["observationoffset"].diff()
        min_ts = hr_col.observationoffset.min()
        max_ts = hr_col.observationoffset.max()
        segment_hours = (max_ts-min_ts)/60.0

        # Exclude stays longer than 30 days or shorter than 1 day
        if np.isnan(segment_hours) or segment_hours < 24*configs["min_length_days"] or segment_hours > configs["max_length_days"]*24:
            continue

        max_disconnect_mins = int(diffed_hr_col.max())

        # Exclude stays where the HR sensor is disconnected for more than 60 minutes.
        if max_disconnect_mins > configs["max_hr_disconnect_mins"]:
            continue

        included_patients.append(pid)

    sorted_inc_pids = list(sorted(included_patients))
    mlhc_io.write_list_to_file(configs["output_path"], sorted_inc_pids)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    

    # Input paths
    parser.add_argument("--all_pid_stay_path", default="../data/all_pid_stays.txt", help="Location of the file with all PIDs in the eICU database") 
    parser.add_argument("--vital_per_table_path", default="../data/hdf/vitalPeriodic.h5", help="Location of the vital periodic table") 

    # Output paths
    parser.add_argument("--output_path", default="../data/included_pid_stays.txt", help="Location of the list of included patients to output")

    # Parameters
    parser.add_argument("--min_length_days", type=int, default=5, help="Minimum length of a valid ICU stay in days")
    parser.add_argument("--max_length_days", type=int, default=30, help="Maximum length of a valid ICU stay in days")
    parser.add_argument("--max_hr_disconnect_mins", type=int, default=60, help="Maximum time in minutes that HR channel could be disconnected")

    args = parser.parse_args()
    configs = vars(args)
    
    filter_patients(configs)
