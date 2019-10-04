# Variational PSOM

## Reference
> Laura Manduchi, Matthias Hüser, Gunnar Rätsch, and Vincent Fortuin. Variational PSOM: Deep Probabilistic Clustering with Self-Organizing Maps. arXiv:1910.01590, Oct 2019

## Training and Evaluation

### Variational PSOM

The training script of VarPSOM model is `varpsom/VarPSOM.py`, the model is defined in `varpsom/VarPSOM_model.py`.
To train and test the VarPSOM model on the MNIST dataset using default parameters and feed-forward layers:

````python varpsom/VarPSOM.py````

This will train the model and then it will output the clustering performance on test set.

To use convolutional layers:

```python varpsom/VarPSOM.py with convolution=True```

Other possible configurations:
- `validation`: if True it will evaluate the model on validation set (default False).
- `val_epochs`: if True the clustering results are saved every 10 epochs on default output files (default False).
- `more_runs`: if True it will run the model 10 times and it will output the NMI and Purity means with standard errors (default False).

To train and test the VarPSOM model on the Fashion MNIST dataset using default parameters and feed-forward layers:

``` python varpsom/VarPSOM.py with data_set="fMNIST" beta=10. epochs_pretrain=20 ```

To use convolutional layers:

``` python varpsom/VarPSOM.py with data_set="fMNIST" beta=10. epochs_pretrain=20 convolution=True ```

To investigate the role of the weight of the SOM loss use

````python varpsom/VarPSOM.py with beta=<new_value>````

default is `beta=20`.

To reconstruct the centroids of the learned 2D SOM grid into the input space we refer to the Notebook `notebooks/centroids_rec.ipynb`.

### VarIDEC

The training script of VarIDEC model is `varpsom/VarIDEC.py`, the model is defined in `varpsom/VarIDEC_model.py`.
To train and test the model on the MNIST dataset using default parameters:

````python varpsom/VarIDEC.py````

To train and test the model on the Fashion MNIST dataset using default parameters:

````python varpsom/VarIDEC.py````

### Variational Temporal PSOM (VarTPSOM)

#### eICU preprocessing pipeline

The major preprocessing steps, which have to be performed sequentially, starting
from the raw eICU tables in CSV format, are listed below. The scripts expect
the tables to be stored in `data/csv`. Intermediate data is stored in various
sub-folders of `data`.

(a) Conversion of raw CSV tables, which can be downloaded from
    https://eicu-crd.mit.edu/ after access is granted, to HDF versions of the
    tables. (`eicu_preproc/hdf_convert.py`)

(b) Filtering of ICU stays based on inclusion criteria.
    (`eicu_preproc/save_all_pids.py`, `eicu_preproc/filter_patients.py`)

(c) Batching patient IDs for cluster processing
    (`eicu_preproc/compute_patient_batches.py`)

(d) Selection of variables to include in the multi-variate time series, from
    the vital signs and lab measurement tables.
    (`eicu_preproc/filter_variables.py`)

(e) Conversion of the eICU data to a regular time grid format using
    forward filling imputation, which can be processed by VarTPSOM.
    (`eicu_preproc/timegrid_all_patients.py`, `eicu_preproc/timegrid_one_batch.py`)

(f) Labeling of the time points in the time series with the current/future
    worse physiology scores as well as dynamic mortality, which
    are used in the enrichment analyses and data visualizations.
    (`eicu_preproc/label_all_patients.py`, `eicu_preproc/label_one_batch.py`)
 
#### Saving the eICU data-set

Insert the paths of the obtained preprocessed data into the script `eicu_preproc/save_model_inputs.py` and run it.

The script selects the last 72 time-step of each time-series and the following labels:

`'full_score_1', 'full_score_6', 'full_score_12', 'full_score_24',
                                         'hospital_discharge_expired_1', 'hospital_discharge_expired_6',
                                         'hospital_discharge_expired_12', 'hospital_discharge_expired_24',
                                         'unit_discharge_expired_1', 'unit_discharge_expired_6',
                                         'unit_discharge_expired_12', 'unit_discharge_expired_24'`
                                         
It then saves the dataset in a csv table in `data/eICU_data.csv`.

#### Training the model

Once the data is saved in `data/eICU_data.csv` the entire model can be trained using:

`python varpsom/VarTPSOM.py`

It will output NMI clustering results using APACHE scores as labels.

To train the model without prediction, use:

`python varpsom/VarTPSOM.py with eta=0`

To train the model without smoothness loss and without prediction, use:

`python varpsom/VarTPSOM.py with eta=0 kappa=0`

Other experiments as computing prediction, heatmaps and trajectories can be found in `notebook/eICU_experiments.ipynb`.

