# DH17830 Individual Project Code

## Packages

Install dependencies using 'conda env create' while inside the directory
or 'conda env create --file path/to/file/PDS_ALGORITHM/PDS.yml'

## Running

Run using 'python PDS.py [compas|german]' with either the dataset you would like. Compas is the default option if left blank.
Parameters P2N, N2P and target are defined in PDS.py, or can be input at the start by running the program in interactive mode using 'python PDS.py interactive'.
By default, the target parameter will be set to the dataset base rate, but can be set manually using interactive mode.

## Files

compas.py and german.py are used for loading the datasets and generating predictions.
network.py is the neural network implemented in PyTorch.
graphing.py is the data visualisation functions using PyPlot.
