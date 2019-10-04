""" 
Cluster dispatch script for endpoint computation
"""

import subprocess
import argparse
import pickle
import os.path
import os
import sys

import functions.util_filesystem as mlhc_fs

def label_all_patients(configs):
    job_index=0
    mem_in_mbytes=configs["mem_in_mbytes"]
    n_cpu_cores=1
    n_compute_hours=configs["nhours"]
    compute_script_path=configs["compute_script_path"]

    with open(configs["patient_batch_path"],'rb') as fp:
        obj=pickle.load(fp)
        batch_to_lst=obj["batch_to_lst"]
        batches=list(sorted(batch_to_lst.keys()))

    for batch_idx in batches:
        
        print("Dispatching labels for batch {}".format(batch_idx))
        job_name="labels_batch_{}".format(batch_idx)
        log_result_file=os.path.join(configs["log_base_dir"], "label_batch_{}_RESULT.txt".format(batch_idx))
        mlhc_fs.delete_if_exist(log_result_file)
        cmd_line=" ".join(["bsub", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours), 
                           "-J","{}".format(job_name), "-o", log_result_file, "python3", compute_script_path, "--run_mode CLUSTER", "--batch_id {}".format(batch_idx)])
        assert(" rm " not in cmd_line)
        job_index+=1

        if not configs["dry_run"]:
            subprocess.call([cmd_line], shell=True)
        else:
            print("Generated cmd line: [{}]".format(cmd_line))

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--patient_batch_path", default="../data/patient_batches.pickle",help="The path of the PID-Batch map") 
    parser.add_argument("--compute_script_path", default="./label_data_one_batch.py", help="Script to dispatch")

    # Output paths
    parser.add_argument("--log_base_dir", default="../data/logs", help="Log base directory")

    # Parameters
    parser.add_argument("--dry_run", action="store_true", default=False, help="Dry run, do not generate any jobs")
    parser.add_argument("--mem_in_mbytes", type=int, default=5000, help="Number of MB to request per script")
    parser.add_argument("--nhours", type=int, default=4, help="Number of hours to request")

    args=parser.parse_args()
    configs=vars(args)
    
    label_all_patients(configs)
