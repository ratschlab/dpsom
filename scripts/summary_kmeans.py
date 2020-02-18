''' Summary table of K-means results'''

import argparse
import csv
import glob
import os
import os.path
import ipdb
import math

import numpy as np

def execute(configs):

    km_result_dict={}
    hmm_result_dict=[]
    
    for fpath in sorted(glob.glob(os.path.join(configs["result_path"], "results_*.tsv"))):
        with open(fpath,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            
            for model,task,nmi in csv_fp:
                if "kmeans" not in model:
                    continue
                
                if task not in km_result_dict:
                    km_result_dict[task]=[]

                km_result_dict[task].append(float(nmi))

    for fpath in sorted(glob.glob(os.path.join(configs["result_path"], "hmm_results_*.tsv"))):
        with open(fpath,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for nmi in csv_fp:
                hmm_result_dict.append(float(nmi[0]))
                
    for task in sorted(km_result_dict.keys()):
        print("{}: Mean: {}, Stderr: {}".format(task, np.mean(km_result_dict[task]), np.std(km_result_dict[task])/math.sqrt(10)))

    print("HMM:  Mean: {:.5f}, Stderr: {:.5f}".format(np.mean(hmm_result_dict), np.std(hmm_result_dict)/math.sqrt(10)))

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()

    parser.add_argument("--result_path", default="../results")

    configs=vars(parser.parse_args())
    
    execute(configs)
