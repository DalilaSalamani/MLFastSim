from argparse import ArgumentParser

from core.constants import GPU_IDS, MAX_GPU_MEMORY_ALLOCATION
from utils.gpu_limiter import GPULimiter
from utils.preprocess import preprocess


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--max-gpu-memory-allocation", type=int, default=MAX_GPU_MEMORY_ALLOCATION)
    argument_parser.add_argument("--gpu-ids", type=str, default=GPU_IDS)
    args = argument_parser.parse_args()
    return args


def main():
    # 0. Parse arguments.
    args = parse_args()
    max_gpu_memory_allocation = args.max_gpu_memory_allocation
    gpu_ids = args.gpu_ids

    # 1. Set GPU memory limits.
    GPULimiter(_gpu_ids=gpu_ids, _max_gpu_memory_allocation=max_gpu_memory_allocation)()

    # 2. Data loading/preprocessing

    # The preprocess function reads the data and performs preprocessing and encoding for the values of energy,
    # angle and geometry
    energies_train, cond_e_train, cond_angle_train, cond_geo_train = preprocess()

    # 3. Manufacture model handler.

    # This import must be local because otherwise it is impossible to call GPULimiter.
    from core.model import VAEHandler
    vae = VAEHandler(_number_of_k_fold_splits=1, _epochs=50, _learning_rate=0.01, _batch_size_per_replica=128,
                     _early_stop=False)

    # 4. Train model.
    histories = vae.train(energies_train,
                          cond_e_train,
                          cond_angle_train,
                          cond_geo_train
                          )

    # Note : One history object can be used to plot the loss evaluation as function of the epochs. Remember that the
    # function returns a list of those objects. Each of them represents a different fold of cross validation.


if __name__ == "__main__":
    exit(main())
