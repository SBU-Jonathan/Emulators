# Halofit Neural Network emulator
This project aims to study Halofit emulation to investigate some questions regarding COLA emulation:
- How many data points do we need to have a sub-percent emulation error?
- Do we need BAO smearing? What's the best method to smear BAO?

The files in this repository are:
- `Halofit_Emulator.ipynb`: notebook that reads power spectrum data, preprocess it (i.e. BAO smearing, PCA, min-max normalization), trains a NN model on the data and evaluates the model on test data.
- `train_utils.py`: library that contains some useful functions and constants called in the notebook. In this library, the model definitions are stated.
- `models/`: folder where models are stored.
- `plots/`: folder where plots are stored.

The model: ResNet (ANN with two hidden layers and one skipped connection from the first hidden layer to the output layer). It is defined in `train_utils.py` in the function `generate_resnet`. I also use `l1_l2` regularization with magnitude `alpha = 1e-4` and `l1_ratio = 0.1`. For training, I use 2.5k epochs, and starting from epoch 1500, the learning rate decays by a factor of 2 every 200 epochs.
