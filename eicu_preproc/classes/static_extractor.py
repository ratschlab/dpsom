"""
Static data extraction from eICU
"""

import numpy as np
import pandas as pd

class StaticExtractor():
    
    def __init__(self):
        self.debug = False

        self.create_pid_col = True

        self.pat_table_cont_vars = ["age", "admissionheight", "admissionweight"]

        self.aav_table_cont_vars = ["intubated", "vent", "dialysis", "eyes", "motor", "verbal",
                                    "meds", "urine", "wbc", "temperature", "respiratoryrate",
                                    "sodium", "heartrate", "meanbp", "ph", "hematocrit", "creatinine",
                                    "albumin", "pao2", "pco2", "bun", "glucose", "bilirubin", "fio2"]

        self.apr_table_cont_vars = ["acutephysiologyscore", "apachescore", "predictedicumortality",
                                    "predictediculos", "predictedhospitalmortality", "predictedhospitallos",
                                    "preopmi", "preopcardiaccath", "ptcawithin24h", "predventdays"]

        self.apv_table_cont_vars = ["graftcount", "meds", "verbal", "motor", "eyes", "thrombolytics",
                                    "aids", "hepaticfailure", "lymphoma", "metastaticcancer",
                                    "leukemia", "immunosuppression", "cirrhosis", "electivesurgery",
                                    "activetx", "readmit", "ima", "midur", "ventday1", "oobventday1",
                                    "oobintubday1", "diabetes", "pao2", "fio2", "creatinine",
                                    "visitnumber", "day1meds", "day1verbal", "day1motor", "day1eyes",
                                    "day1pao2", "day1fio2"]

    def transform(self, df_pat, df_adm, df_aav, df_apr, df_apv, pid=None):
        df_out_dict = {}

        if self.create_pid_col:
            df_out_dict["patientunitstayid"] = np.array([int(pid)], dtype=np.int64)

        rel_row = df_pat.iloc[0]

        # GENDER
        gender = rel_row["gender"]

        if gender not in ["Male", "Female"]:
            df_out_dict["patient_gender"] = np.array([np.nan]).astype(np.float64)
        else:
            df_out_dict["patient_gender"] = np.array([1 if gender == "Male" else 0], dtype=np.float64)

        # ALL CONTINUOUS VARIABLES IN PATIENT TABLE
        for var in self.pat_table_cont_vars:
            try:
                var_val = float(rel_row[var])
            except:
                var_val = np.nan
            df_out_dict["patient_{}".format(var)] = np.array([var_val], dtype=np.float64)

        nrows = df_aav.shape[0]

        if nrows == 0:
            for var in self.aav_table_cont_vars:
                df_out_dict["apacheapsvar_{}".format(var)] = np.array([np.nan], dtype=np.float64)

        else:
            assert(nrows == 1)
            rel_row = df_aav.iloc[0]

            # ALL CONTINOUS VARIABLES IN AAV TABLE
            for var in self.aav_table_cont_vars:
                var_val = rel_row[var]

                # -1 ENCODING FOR MISSING VALUES IN THIS TABLE
                if var_val == -1:
                    var_val = np.nan

                df_out_dict["apacheapsvar_{}".format(var)] = np.array([float(var_val)], dtype=np.float64)

        nrows = df_apr.shape[0]

        if nrows == 0:
            for var in self.apr_table_cont_vars:
                df_out_dict["apachepatientresult_{}".format(var)] = np.array([np.nan], dtype=np.float64)

        else:
            rel_row = df_apr.iloc[0]

            # ALL CONTINUOUS VARIABLES IN APR TABLE
            for var in self.apr_table_cont_vars:
                var_val = rel_row[var]
                df_out_dict["apachepatientresult_{}".format(var)] = np.array([float(var_val)], dtype=np.float64)

        nrows = df_apv.shape[0]

        if nrows == 0:
            for var in self.apv_table_cont_vars:
                df_out_dict["apachepredvar_{}".format(var)] = np.array([np.nan], dtype=np.float64)

        else:
            assert(nrows == 1)
            rel_row = df_apv.iloc[0]

            # ALL CONTINUOUS VARIABLES IN APV TABLE
            for var in self.apv_table_cont_vars:
                var_val = rel_row[var]

                # -1 ENCODING FOR MISSING VALUES IN THIS TABLE
                if var_val == -1:
                    var_val = np.nan

                df_out_dict["apachepredvar_{}".format(var)] = np.array([float(var_val)], dtype=np.float64)

        df_out = pd.DataFrame(df_out_dict)

        if self.debug:
            n_cols = len(df_out.columns.values.tolist())
            assert(n_cols == 71)
            col_types = df_out.dtypes.values.tolist()
            float_cnt = 0
            int_cnt = 0

            for col_type in col_types:
                if col_type == np.dtype("int64"):
                    int_cnt += 1
                elif col_type == np.dtype("float64"):
                    float_cnt += 1

            assert(float_cnt == 70)
            assert(int_cnt == 1)

        return df_out
