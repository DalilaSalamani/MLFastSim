"""
** convert **
defines the conversion function to and ONNX file
"""

# Imports
import sys
import argparse
from configure import Configure
from instantiate_model import *


# parse_args function
"""
    epoch: epoch of the saved checkpoint model
"""
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default="")
    args = p.parse_args()
    return args

# main function
def main(argv):
    # 1. Setup the model to convert
    # Parse commandline arguments
    args = parse_args(argv)
    epoch = args.epoch
    # Get list of common variables
    variables = Configure()
    # Instantiate and load a saved model 
    vae = instantiate()
    # Load the saved weights
    vae.vae.load_weights('%sVAE-%s.h5'%(variables.checkpoint_dir,epoch) )

    # 2. Convert the model to ONNX format
    import keras2onnx
    # Create the Keras model and convert itinto an ONNX model
    kerasModel = vae.decoder
    onnxModel = keras2onnx.convert_keras(kerasModel,"name")
    # Save the ONNX model. Generator.onnx can then be used to perform the inference in the example
    keras2onnx.save_model(onnxModel,"%sGenerator.onnx"%variables.conversion)

if __name__ == '__main__':
    exit(main(sys.argv[1:]))