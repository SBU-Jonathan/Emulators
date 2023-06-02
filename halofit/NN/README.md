# Halofit Neural Network emulator
This project aims to study Halofit emulation to investigate some questions regarding COLA emulation:
- How many data points do we need to have a sub-percent emulation error?
- Do we need BAO smearing? What's the best method to smear BAO?
- What is the best architecture?

The files in this repository are:
- `Halofit_Emulator.ipynb`: notebook that reads power spectrum data, preprocess it (i.e. BAO smearing, PCA, min-max normalization), trains a NN model on the data and evaluates the model on test data.
- `train_utils.py`: library that contains some useful functions and constants called in the notebook. In this library, the model definitions are stated.
- `models/`: folder where models are stored.
- `plots/`: folder where plots are stored.

Conclusions:
- From all the things we tried (e.g. number of layers, number of neurons, number of PCs, residual connections...), the thing that caused the most impact on the emulation errors is the activation function. We have used an activation defined in e.g. [CosmoPower](https://arxiv.org/pdf/2106.03846.pdf) and references therein. This activation has greatly outperformed ReLU.
- With this activation, an MLP with 3 hidden layers and 512 neurons is sufficient to achieve great results with emulation errors around 0.2%.
