# Classification sub-group

We have run an initial classification over the `total_dataset.pkl` and `acceptance_mc.pkl` dataset. The results are located in the `/classification/data_processed/` folder. We have stored the files in a compressed `bz2` format to save space. 

Each dataset contains only events which the neural net has identified as the signal. We include only the following columns: `q2`, `phi`, `costhetal`, and `costhetak`. 

To load the files, either use functions in `/classification/data_load.py` or use `pd.read_pickle(fp)`. 

The files of interest in data_processed are:
- `Classified total Dataset.bz2`: classified total dataset
- `Classified_acceptance_mc.bz2`: classified acceptance dataset 