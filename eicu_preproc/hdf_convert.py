"""
Convert eICU tables into the HDF data format
"""

import os
import argparse
import csv
import os.path
import pandas as pd
import numpy as np
import glob as glob
import ipdb
import gc

TABLES=["admissionDrug", "admissionDx", "allergy", "apacheApsVar", "apachePatientResult", "apachePredVar", "carePlanCareProvider", 
        "carePlanEOL", "carePlanGeneral" , "carePlanGoal", "carePlanInfectiousDisease", "customLab", "diagnosis", "hospital", 
        "infusionDrug", "intakeOutput", "lab", "medication", "microLab", "note", "nurseAssessment", "nurseCare", "nurseCharting", 
        "pastHistory", "patient", "physicalExam", "respiratoryCare", "respiratoryCharting", "treatment", "vitalAperiodic",
        "vitalPeriodic"]

def hdf_convert(configs):
    
    for table in TABLES:

        print("Processing table {}".format(table))
        table_path=os.path.join(configs["source_data_dir"],"{}.csv.gz".format(table))

        if table=="lab":
            dtype_dict={"labresulttext": object}
        elif table=="infusionDrug":
            dtype_dict={"drugrate": object}
        elif table=="respiratoryCare":
            dtype_dict={"airwayposition": object, 
                        "airwaysize": object,
                        "apneaparms": object, 
                        "setapneafio2": object,
                        "setapneainsptime": object,
                        "setapneainterval": object,
                        "setapneaippeephigh": object,
                        "setapneapeakflow": object,
                        "setapneatv": object}
        elif table=="respiratoryCharting":
            dtype_dict={"respchartvalue": object}
        else:
            dtype_dict=None

        df=pd.read_csv(table_path,quoting=csv.QUOTE_ALL,dtype=dtype_dict, compression="gzip")
        df.to_hdf(os.path.join(configs["dest_dir"],"{}.h5".format(table)),key=configs["dset_key"],mode='w',format="table",
                  data_columns=["patientunitstayid"],complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"])

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--dset_key", default="data", help="Data set key to write")
    parser.add_argument("--hdf_comp_level", default=5, type=int, help="HDF compression level to use")
    parser.add_argument("--hdf_comp_alg", default="blosc:lz4", help="HDF compression algorithm to use")

    # Input paths
    parser.add_argument("--source_data_dir", default="../data/csv", help="Source data directory with CSV tables") 

    # Output paths
    parser.add_argument("--dest_dir", default="../data/hdf", help="Destination directory where the HDF tables should be saved into")

    args=parser.parse_args()
    configs=vars(args)

    hdf_convert(configs)
    
