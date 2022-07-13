"""
** convert **
defines the conversion function to and ONNX file
"""

import argparse
import sys

import keras2onnx

from core.constants import CHECKPOINT_DIR
from core.model import VAE

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
    # 1. Set up the model to convert
    # Parse commandline arguments
    args = parse_args(argv)
    epoch = args.epoch
    # Instantiate and load a saved model
    vae = VAE()
    # Load the saved weights
    vae.vae.load_weights(f"{CHECKPOINT_DIR}VAE-{epoch}.h5")

    # 2. Convert the model to ONNX format
    # Create the Keras model and convert it into an ONNX model
    keras_model = vae.decoder
    onnx_model = keras2onnx.convert_keras(keras_model, "name")
    # Save the ONNX model. Generator.onnx can then be used to perform the inference in the example
    keras2onnx.save_model(onnx_model, "Generator.onnx")


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
