""" 
Save list of base PIDs contained in the eICU data-set
"""

import pandas as pd
import argparse

import functions.util_io as mlhc_io

def save_all_pids(configs):
    
    print("Patient static information")
    df_patient = pd.read_hdf(configs["patient_table_path"], configs["generic_dset_id"], mode='r')
    stay_ids = list(df_patient["patientunitstayid"].unique())
    mlhc_io.write_list_to_file(configs["pid_stay_list"], stay_ids)

    print("Async Vital Signs")
    df_async = pd.read_hdf(configs["vital_aper_path"], configs["generic_dset_id"], mode='r')
    for vs in configs["ASYNC_VITALS"]:
        orig = df_async[vs]
        finite = orig.dropna()
        print("{}: Number of entries: {}".format(vs, finite.size))

    print("Sync Vital Signs")
    df_sync = pd.read_hdf(configs["vital_per_path"], configs["generic_dset_id"], mode='r')

    for vs in configs["SYNC_VITALS"]:
        orig = df_sync[vs]
        finite = orig.dropna()
        print("{}: Number of entries: {}".format(vs, finite.size))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--patient_table_path", default="../data/hdf/patient.h5" , help="Location of patient table") 
    parser.add_argument("--vital_aper_path", default="../data/hdf/vitalAperiodic.h5", help="Path of vital aperiodic table") 
    parser.add_argument("--vital_per_path", default="../data/hdf/vitalPeriodic.h5" , help="Path of vital periodic table")

    # Output paths
    parser.add_argument("--pid_stay_list", default="../data/all_pid_stays.txt", help="PID stay list to write")

    # Parameters
    
    parser.add_argument("--generic_dset_id", default="data", help="HDF data-set ID")

    args = parser.parse_args()
    configs = vars(args)

    # Constants

    configs["ASYNC_VITALS"] = ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean", "paop", "cardiacoutput", "cardiacinput",
                               "svr", "svri", "pvr", "pvri"]

    configs["SYNC_VITALS"] = ["temperature", "sao2", "heartrate", "respiration", "cvp", "etco2", "systemicsystolic", "systemicdiastolic",
                              "systemicmean", "pasystolic", "padiastolic", "pamean", "st1", "st2", "st3", "icp"]

    save_all_pids(configs)
