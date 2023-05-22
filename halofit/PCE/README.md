## Table of Contents
1. [Polynomial Chaos Expansion](#polynomial-chaos-expansion)
 
2. [Installation](#installation)   
   1. [Packages](#usage-subsection-1)
3. [Usage](#usage)



 



# Polynomial Chaos Expansion

Emulation, in its essence, is about generating data that's both fast and precise for a wide range of potential inputs all at once. One tool that's really handy for this is something called Polynomial Chaos Expansion (PCE), which people have been studying and using in a bunch of different areas, including cosmology, it's a go-to method for quantifying uncertainty and creating surrogate models, along with Gaussian Process (GP) and Neural Network (NN) regression.

At its core, PCE is about breaking down the output of a model or simulation into a set of orthogonal polynomials. These polynomials are functions of the input variables themselves. The magic of PCE lies in its ability to approximate the relationship between the inputs and outputs using a polynomial expansion. This approximation lets us determine the expansion coefficients with only a few evaluations of the model or simulation. It's especially handy when the input variables are uncertain or change a lot, as it allows us to analyze how sensitive the output is to changes in the inputs.

The basis for the expansion in PCE depends on the probability distribution of the input variables. We choose the basis functions so they're orthogonal with respect to the joint probability distribution of these variables and then we determine the expansion coefficients by fitting the output data to the polynomial expansion. Different types of orthogonal polynomials can be used to deal with different types of probability distributions. By choosing the right set of orthogonal polynomials, the PCE method can accurately capture the relationship between the input variables and the output, and quantify the uncertainty in the output due to changes in the input variables. In cases where the input variables don't follow a particular probability distribution, PCE simplifies to just a polynomial expansion that can approximate any function of the input variables. When the input variables don't conform to a specific probability distribution, the process becomes somewhat simpler. Polynomial Chaos Expansion (PCE) transforms into a basic polynomial expansion that is capable of approximating any function of the input variables.

As for determining each coefficient of the polynomial, one popular technique we turn to is called the Elastic Net. The Elastic Net is a clever method that combines two techniques - ridge regression and lasso regression. Ridge regression prevents overfitting by adding a penalty equivalent to square of the magnitude of coefficients. Lasso regression, on the other hand, performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the model. Elastic Net combines these approaches to handle situations where there are highly correlated variables. This involves finding the best set of coefficients that will minimize the difference between what we predict and what we actually observe. We have a few tools at our disposal to achieve this, like gradient descent. The ultimate goal is to identify the polynomial that best fits the given data points. And how do we decide the degree of this polynomial? Well, that's a bit of a balancing act between accuracy and complexity, we want a model that's accurate while ensuring it doesn't become overly complex to manage.

The purpose of this part of the project is to investigate Halofit emulation, placing a particular emphasis on a few critical questions around COLA emulation: Firstly, what is the minimum number of data points needed to limit the emulation error to less than one percent? Secondly, is there a necessity to include Baryon Acoustic Oscillations (BAO) smearing in the process? If so, what would be the optimal technique to implement BAO smearing?


# Installation
...

# Packages 

To run the training notebook you will need to have the following packages installed:

- chaospy
- numpoly
- scikit-learn
- euclidemu2
- regfortran___2  
- scipy
- numpy
- tqdm

To install chaospy and numpoly use:

    pip install numpoly && pip install chaospy

**DO NOT USE ''CONDA INSTALL''** \-> both packages will be installed incorrectly using the conda install command due to a dependency mismatch.

To install scikit-learn follow [this](https://scikit-learn.org/stable/install.html).

To install euclidemu2 follow [this](https://github.com/miknab/EuclidEmulator2).

To install scipy follow [this](https://scipy.org/install/).

To install numpy follow [this]( https://numpy.org/install/).

To install tqdm follow [this]( https://github.com/tqdm/tqdm#installation).

regfortran___2 


# Usage
...






