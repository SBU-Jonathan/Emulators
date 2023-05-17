# Halofit Neural Network emulator
This project aims to study Halofit emulation to investigate some questions regarding COLA emulation:
- How many data points do we need to have a sub-percent emulation error?
- Do we need BAO smearing? What's the best method to smear BAO?
The files in this repository are:
- `Halofit_Emulator.ipynb`: notebook that reads power spectrum data, preprocess it (i.e. BAO smearing, PCA, min-max normalization), trains a NN model on the data and evaluates the model on test data.
- `train_utils.py`: library that contains some useful functions and constants called in the notebook.
- `models/`: folder where models are stored.
- `plots/`: folder where plots are stored.
