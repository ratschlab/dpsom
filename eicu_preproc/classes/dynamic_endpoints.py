"""
Dynamic endpoints on eICU
"""

import numpy as np
import pandas as pd

import functions.util_array as mlhc_array

class DynamicEndpointExtractor():

    def __init__(self):

        self.create_pid_col = True

        # The horizons at the end which are marked as patient severity
        self.back_horizons = [1, 6, 12, 24]

        self.unit_discharge_categories = {"home": ["Home"],
                                          "telemetry": ["Telemetry"],
                                          "floor": ["Floor"],
                                          "step_down_unit": ["Step-Down Unit (SDU)"],
                                          "acute_care_floor": ["Acute Care/Floor"],
                                          "other_icu": ["Other ICU", "ICU", "Other ICU (CABG)"],
                                          "expired": ["Death"],
                                          "skilled_nursing_facility": ["Skilled Nursing Facility"],
                                          "other_hospital": ["Other Hospital"]}

        self.hospital_discharge_categories = {"home": ["Home"],
                                              "skilled_nursing_facility": ["Skilled Nursing Facility"],
                                              "expired": ["Death"],
                                              "rehabilitation": ["Rehabilitation"],
                                              "other_hospital": ["Other Hospital"],
                                              "nursing_home": ["Nursing Home"]}

        # The variables that are to be used as critical thresholds
        self.relevant_variables_vitals = ["temperature", "systemicmean", "respiration"]
        self.relevant_variables_lab = ["HCO3", "sodium", "potassium", "creatinine"]

    def transform(self, df_imputed, df_pat, pid=None):

        df_out_dict = {}

        if self.create_pid_col:
            df_out_dict["patientunitstayid"] = mlhc_array.value_empty(df_imputed.shape[0], pid, dtype=np.int64)

        df_out_dict["ts"] = df_imputed["ts"]

        rel_row = df_pat.iloc[0]

        hospital_discharge_location = str(rel_row["hospitaldischargelocation"]).strip()
        unit_discharge_location = str(rel_row["unitdischargelocation"]).strip()

        for var, vnames in self.unit_discharge_categories.items():

            if unit_discharge_location in vnames:

                for hor in self.back_horizons:
                    arr = np.zeros(df_imputed.shape[0], dtype=np.float64)
                    arr[-hor:] = 1.0
                    df_out_dict["unit_discharge_{}_{}".format(var, hor)] = arr

            else:

                for hor in self.back_horizons:
                    arr = np.zeros(df_imputed.shape[0], dtype=np.float64)
                    df_out_dict["unit_discharge_{}_{}".format(var, hor)] = arr

        for var, vnames in self.hospital_discharge_categories.items():

            if hospital_discharge_location in vnames:

                for hor in self.back_horizons:
                    arr = np.zeros(df_imputed.shape[0], dtype=np.float64)
                    arr[-hor:] = 1.0
                    df_out_dict["hospital_discharge_{}_{}".format(var, hor)] = arr

            else:

                for hor in self.back_horizons:
                    arr = np.zeros(df_imputed.shape[0], dtype=np.float64)
                    df_out_dict["hospital_discharge_{}_{}".format(var, hor)] = arr

        # Process the vital sign variables of interest
        temperature = np.array(df_imputed["vs_temperature"])
        abpm = np.array(df_imputed["vs_systemicmean"])
        rrate = np.array(df_imputed["vs_respiration"])
        hco3 = np.array(df_imputed["lab_HCO3"])
        sodium = np.array(df_imputed["lab_sodium"])
        potassium = np.array(df_imputed["lab_potassium"])
        creatinine = np.array(df_imputed["lab_creatinine"])*100  # Wrong unit in the input data

        for hor in self.back_horizons:

            full_score_out = np.zeros(df_imputed.shape[0])

            # TEMPERATURE

            set_indices = {}

            for config, thresholds in [("high4", [41, np.inf]), ("low4", [-np.inf, 30]), ("high3", [39, 41]), ("low3", [30, 32]),
                                       ("low2", [32, 34]), ("high1", [38.5, 39]), ("low1", [34, 36])]:

                temp_out = np.zeros(df_imputed.shape[0])

                for idx in np.arange(temperature.size):
                    forward_window = temperature[idx:min(temperature.size, idx+hor)]
                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        temp_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["vs_temperature_{}_{}".format(config, hor)] = temp_out

            # MEAN ARTERIAL PRESSURE

            set_indices = {}

            for config, thresholds in [("high4", [160, np.inf]), ("low4", [-np.inf, 50]), ("high3", [130, 160]), ("high2", [110, 130]),
                                       ("low2", [50, 70])]:

                abpm_out = np.zeros(df_imputed.shape[0])

                for idx in np.arange(abpm.size):
                    forward_window = abpm[idx:min(abpm.size, idx+hor)]
                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        abpm_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["vs_systemicmean_{}_{}".format(config, hor)] = abpm_out

            # RESPIRATION RATE

            set_indices = {}

            for config, thresholds in [("high4", [50, np.inf]), ("low4", [-np.inf, 6]), ("high3", [35, 50]), ("low2", [6, 10]),
                                       ("high1", [25, 35]), ("low1", [10, 12])]:

                rrate_out = np.zeros(df_imputed.shape[0])

                for idx in np.arange(rrate.size):
                    forward_window = rrate[idx:min(rrate.size, idx+hor)]
                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        rrate_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["vs_respiration_{}_{}".format(config, hor)] = rrate_out

            # HCO3 LAB VALUE

            set_indices = {}

            for config, thresholds in [("high4", [52, np.inf]), ("low4", [-np.inf, 15]), ("high3", [41, 52]), ("low3", [15, 18]), ("low2", [18, 23]), ("high1", [32, 41])]:

                hco3_out = np.zeros(df_imputed.shape[0])

                for idx in np.arange(hco3.size):
                    forward_window = hco3[idx:min(hco3.size, idx+hor)]

                    if not np.isfinite(forward_window).all():
                        ipdb.set_trace()

                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        hco3_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["lab_hc03_{}_{}".format(config, hor)] = hco3_out

            # SODIUM LAB VALUE

            set_indices = {}

            for config, thresholds in [("high4", [180, np.inf]), ("low4", [-np.inf, 111]), ("high3", [160, 180]), ("low3", [111, 120]), ("high2", [155, 160]), ("low2", [120, 130]), ("high1", [150, 155])]:

                sodium_out = np.zeros(df_imputed.shape[0])

                for idx in np.arange(sodium.size):
                    forward_window = sodium[idx:min(sodium.size, idx+hor)]
                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        sodium_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["lab_sodium_{}_{}".format(config, hor)] = sodium_out

            # POTASSIUM LAB VALUE

            set_indices = {}

            for config, thresholds in [("high4", [7, np.inf]), ("low4", [-np.inf, 2.5]), ("high3", [6, 7]), ("low2", [2.5, 3]), ("high1", [5.5, 6]), ("low1", [3, 3.5])]:

                potassium_out = np.zeros(df_imputed.shape[0])

                for idx in np.arange(potassium.size):
                    forward_window = potassium[idx:min(potassium.size, idx+hor)]
                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        potassium_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["lab_potassium_{}_{}".format(config, hor)] = potassium_out

            # CREATININE LAB VALUE

            set_indices = {}

            for config, thresholds in [("high4", [350, np.inf]), ("high3", [200, 350]), ("high2", [150, 200]), ("low2", [-np.inf, 60])]:

                creatinine_out = np.zeros(creatinine.size)

                for idx in np.arange(creatinine.size):
                    forward_window = creatinine[idx:min(creatinine.size, idx+hor)]
                    assert(np.isfinite(forward_window).all())
                    if ((forward_window >= thresholds[0]) & (forward_window < thresholds[1])).any():
                        creatinine_out[idx] = 1.0

                        if idx not in set_indices:
                            full_score_out[idx] += int(config[-1])
                            set_indices[idx] = True

                df_out_dict["lab_creatinine_{}_{}".format(config, hor)] = creatinine_out

            df_out_dict["full_score_{}".format(hor)] = full_score_out

        # Process the lab variables of interest

        df_out = pd.DataFrame(df_out_dict)

        return df_out
