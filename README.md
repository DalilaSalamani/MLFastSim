

This repository contains the set of scripts used to train, generate and validate the generative model used in [Par04](https://gitlab.cern.ch/geant4/geant4/-/tree/master/examples/extended/parameterisations/Par04) Geant4 example.

- configure: defines the set of common variables.
- model: defines the VAE model class.
- instantiate_model: instantiate a VAE model and define all the parameters.
- preprocess: defines the data loading and preprocessing functions. 
- train: performs model training.
- generate: generate showers using a saved VAE model. 
- observables: defines a set of shower observables. 
- validate: creates validation plots using shower observables. 
- convert: defines the conversion function to and ONNX file.

## Getting Started

The setup script creates necessary folders used to save model checkpoints, generate showers and validation plots.

```
python3 setup.py
``` 

## Full simulation dataset

The full simulation dataset can be downloaded/linked to from [Zenodo](https://zenodo.org/record/6082201#.Ypo5UeDRaL4).

## Training

In order to launch the training:

```
python3 train.py
``` 

## ML shower generation (MLFastSim)

In order to generate showers using the ML model, use the generate script and specify information of geometry, energy and angle of the particle and the epoch of the saved checkpoint model. The number of events to generate can also be specified (by default is set to 10.000):

```
python3 generate.py --geometry SiW --energyParticle 64 --angleParticle 90 --epoch 1000
``` 

## Validation

In order to validate the MLFastSim and the full simulation, use the validate script and specify information of geometry, energy and angle of the particle: 

```
python3 validate.py --geometry SiW --energyParticle 64 --angleParticle 90 
``` 

## Conversion

After training and validation, the model can be converted into a format that can be used in C++, such as ONNX, use the convert script:

```
python3 convert.py --epoch 1000
``` 

 
