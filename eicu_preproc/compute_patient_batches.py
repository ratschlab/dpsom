"""
Computes a partition of selected patient IDs for cluster processing
"""

import pickle
import os
import os.path
import argparse

import functions.util_io as mlhc_io

def compute_patient_batches(configs):
    pid_list = mlhc_io.read_list_from_file(configs["included_pid_path"])
    pid_list = list(map(int, pid_list))
    print("Number of PID stays in the database: {}".format(len(pid_list)))
    batch_to_list_dict = {}
    pid_to_batch_dict = {}

    for batch_run_idx, base_idx in enumerate(range(0, len(pid_list), configs["batch_size_patients_per_file"])):
        sub_pid_list = pid_list[base_idx:base_idx+configs["batch_size_patients_per_file"]]
        batch_to_list_dict[batch_run_idx] = pid_list[base_idx:base_idx+configs["batch_size_patients_per_file"]]
        for pid in sub_pid_list:
            pid_to_batch_dict[pid] = batch_run_idx

    pickle_obj = {"batch_to_lst": batch_to_list_dict, "pid_to_batch": pid_to_batch_dict}

    with open(configs["output_path"], 'wb') as fp:
        pickle.dump(pickle_obj, fp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    

    # Input paths
    parser.add_argument("--included_pid_path", default="../data/included_pid_stays.txt", help="What is the path of the included PIDs?")

    # Output paths
    parser.add_argument("--output_path", default="../data/patient_batches.pickle", help="What is the path to be used for the patient batch file")

    # Parameters
    parser.add_argument("--batch_size_patients_per_file", type=int, default=100,help="How many patients should be put into one batch file")

    args = parser.parse_args()
    configs = vars(args)
    
    compute_patient_batches(configs)
