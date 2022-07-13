This repository contains the set of scripts used to train, generate and validate the generative model used
in [Par04](https://gitlab.cern.ch/geant4/geant4/-/tree/master/examples/extended/parameterisations/Par04) Geant4 example.

- constants: defines the set of common variables.
- model: defines the VAE model class.
- instantiate_model: instantiate a VAE model and define all the parameters.
- preprocess: defines the data loading and preprocessing functions.
- train: performs model training.
- generate: generate showers using a saved VAE model.
- observables: defines a set of shower observables.
- validate: creates validation plots using shower observables.
- convert: defines the conversion function to and ONNX file.
- optimizer: defines the Optimizer class.
- optimize: performs hyperparameters optimization.

## Getting Started

`setup.py` script creates necessary folders used to save model checkpoints, generate showers and validation plots.

```
python3 setup.py
``` 

## Full simulation dataset

The full simulation dataset can be downloaded from/linked to [Zenodo](https://zenodo.org/record/6082201#.Ypo5UeDRaL4).

## Training

In order to launch the training:

```
python3 train.py
``` 

## Model optimization

If you want to tune hyperparameters, specify in `optimize.py` parameters to be tuned. There are three types of
parameters: discrete, continuous and categorical. Discrete and continuous require range specification (low, high), but
the categorical parameter requires a list of possible values to be chosen. Then run it with:

```
python3 optimize.py
```

If you want to parallelize tuning process you need to specify a common storage (preferable MySQL database) by
setting `--storage="URL_TO_MYSQL_DATABASE"`. Then you can run multiple processes with the same command:

```
python3 optimize.py --storage="URL_TO_MYSQL_DATABASE"
```

## ML shower generation (MLFastSim)

In order to generate showers using the ML model, use `generate.py` script and specify information of geometry, energy
and angle of the particle and the epoch of the saved checkpoint model. The number of events to generate can also be
specified (by default is set to 10.000):

```
python3 generate.py --geometry SiW --energyParticle 64 --angleParticle 90 --epoch 1000
``` 

## Validation

In order to validate the MLFastSim and the full simulation, use `validate.py` script and specify information of
geometry, energy and angle of the particle:

```
python3 validate.py --geometry SiW --energyParticle 64 --angleParticle 90 
``` 

## Conversion

After training and validation, the model can be converted into a format that can be used in C++, such as ONNX,
use `convert.py` script:

```
python3 convert.py --epoch 1000
```

 
