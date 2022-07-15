import os
from argparse import ArgumentParser

import tensorflow as tf

from core.constants import MAX_GPU_MEMORY_ALLOCATION, GPU_ID
from utils.hyperparameter_tuner import HyperparameterTuner

# Hyperparemeters to be optimized.
discrete_parameters = {"nb_hidden_layers": (1, 6)}
continuous_parameters = {"learning_rate": (0.0001, 0.01)}
categorical_parameters = {}


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--study-name", type=str, default="default_study_name")
    argument_parser.add_argument("--storage", type=str, default=None)
    argument_parser.add_argument("--max-gpu-memory-allocation", type=int, default=MAX_GPU_MEMORY_ALLOCATION)
    argument_parser.add_argument("--gpu-id", type=int, default=GPU_ID)
    args = argument_parser.parse_args()
    return args


def main():
    args = parse_args()
    study_name = args.study_name
    storage = args.storage
    max_gpu_memory_allocation = args.max_gpu_memory_allocation
    gpu_id = args.gpu_id

    if storage is None:
        hyperparameter_tuner = HyperparameterTuner(discrete_parameters, continuous_parameters, categorical_parameters)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate max_gpu_memory_allocation*1024 MB of memory on one of the GPUs
            try:
                tf.config.set_logical_device_configuration(gpus[0], [
                    tf.config.LogicalDeviceConfiguration(memory_limit=1024 * max_gpu_memory_allocation)])  # in MB
                logical_gpus = tf.config.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        hyperparameter_tuner = HyperparameterTuner(discrete_parameters, continuous_parameters, categorical_parameters,
                                                   storage, study_name)

    # Run main tuning function.
    hyperparameter_tuner.tune()

    # Watch out! This script neither deletes the study in DB nor deletes the database itself. If you are using
    # parallelized optimization, then you should care about deleting study in the database by yourself.
    # TODO(@mdragula): Implement cleaning when all processes are done.


if __name__ == "__main__":
    exit(main())
