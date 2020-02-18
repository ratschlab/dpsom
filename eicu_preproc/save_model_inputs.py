"""
Pre-processing functions for the T-DPSOM model, save data-set locally.
"""

import parmap
import numpy as np
from glob import glob
import pandas as pd
import h5py


def get_normalized_data(data, patientid, mins, scales):
    return ((data[data['patientunitstayid'] == patientid] - mins) /
            scales).drop(["patientunitstayid", "ts"], axis=1).fillna(0).values


def get_patient_last(patient, data_frame, data_frame_endpoint, max_n_step, mins_dynamic, scales_dynamic):
    time_series_all = []
    time_series_endpoint_all = []
    patient_data = get_normalized_data(data_frame, patient, mins_dynamic, scales_dynamic)
    patient_endpoint = data_frame_endpoint[data_frame_endpoint['patientunitstayid'] == patient].drop(
        ["patientunitstayid", "ts"], axis=1)
    patient_endpoint = patient_endpoint[['full_score_1', 'full_score_6', 'full_score_12', 'full_score_24',
                                         'hospital_discharge_expired_1', 'hospital_discharge_expired_6',
                                         'hospital_discharge_expired_12', 'hospital_discharge_expired_24',
                                         'unit_discharge_expired_1', 'unit_discharge_expired_6',
                                         'unit_discharge_expired_12', 'unit_discharge_expired_24']].fillna(0).values

    time_series = patient_data[len(patient_data) - max_n_step: len(patient_data)]
    time_series_endpoint = patient_endpoint[len(patient_data) - max_n_step: len(patient_data)]
    time_series_all.append(time_series)
    time_series_endpoint_all.append(time_series_endpoint)

    return np.array(time_series_all), np.array(time_series_endpoint_all)


def parmap_batch_generator(data_total, endpoints_total, mins_dynamic, scales_dynamic, max_n_step):
    time_series_all = []
    time_series_endpoint_all = []
    for p in range(len(data_total)):
        print(p)
        path = data_total[p]
        path_endpoint = endpoints_total[p]
        data_frame = pd.read_hdf(path).fillna(0)
        data_frame_endpoint = pd.read_hdf(path_endpoint).fillna(0)
        assert not data_frame.isnull().values.any(), "No NaNs allowed"
        assert not data_frame_endpoint.isnull().values.any(), "No NaNs allowed"
        patients = data_frame.patientunitstayid.unique()

        temp = parmap.map(get_patient_last, patients, data_frame, data_frame_endpoint, max_n_step, mins_dynamic,
                              scales_dynamic)

        data = []
        labels = []
        for a in range(len(temp)):
            for b in range(len(temp[a][1])):
                labels.append(temp[a][1][b])
                data.append(temp[a][0][b])
        data = np.array(data)
        labels = np.array(labels)
        time_series_all.extend(data)
        time_series_endpoint_all.extend(labels)

    return time_series_all, time_series_endpoint_all

# *******************************************************************************************************************

# path of the preprocessed data
data_total = glob("../data/time_grid/batch_*.h5")

# path of the labels of the preprocessed data
endpoints_total = glob("../data/labels/batch_*.h5")

# path of the labels of the mins
mins_dynamic = pd.read_hdf("../data/time_grid/normalization_values.h5","mins_dynamic")

# path of the labels of the scales
scales_dynamic = pd.read_hdf("../data/time_grid/normalization_values.h5", "scales_dynamic")

# *******************************************************************************************************************

# Create numpy arrays with the last 72 time-steps of each time-series.
data, labels = parmap_batch_generator(data_total, endpoints_total, mins_dynamic, scales_dynamic, max_n_step=72)
l = np.array(labels)
d = np.array(data)
hf = h5py.File('../data/eICU_data.csv', 'w')
hf.create_dataset('x', data=d)
hf.create_dataset('y', data=l)
hf.close()
