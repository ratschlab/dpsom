"""
Imputation / time gridding functions
"""

import numpy as np

def impute_variable(raw_ts, raw_values, timegrid, leave_nan_threshold_secs=None, grid_period=None, normal_value=None):
    pred_values = np.zeros_like(timegrid)
    input_ts = 0
    online_patient_mean = 0.0

    for idx, ts in np.ndenumerate(timegrid):

        while input_ts < raw_ts.size and raw_ts[input_ts] <= ts+grid_period:
            online_patient_mean = (input_ts*online_patient_mean+raw_values[input_ts])/(input_ts+1)
            input_ts += 1

        if input_ts == 0:
            pred_values[idx[0]] = normal_value
            continue

        ext_offset = ts-raw_ts[input_ts-1]

        # We do not fill in values using observed values of >1 hours ago, leave conservatively at NAN
        if ext_offset > leave_nan_threshold_secs:
            pred_values[idx[0]] = online_patient_mean
        else:
            pred_values[idx[0]] = raw_values[input_ts-1]

    return pred_values
