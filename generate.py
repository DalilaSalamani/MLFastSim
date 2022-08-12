"""
** generate **
generate showers using a saved VAE model 
"""
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

from core.constants import GLOBAL_CHECKPOINT_DIR, GEN_DIR, BATCH_SIZE_PER_REPLICA
from core.model import VAEHandler
from utils.preprocess import get_condition_arrays


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="")
    p.add_argument("--energy", type=int, default="")
    p.add_argument("--angle", type=int, default="")
    p.add_argument("--events", type=int, default=10000)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--study-name", type=str, default="default_study_name")
    args = p.parse_args()
    return args


# main function
def main():
    # 0. Parse arguments.
    args = parse_args()
    energy = args.energy
    angle = args.angle
    geometry = args.geometry
    events = args.events
    epoch = args.epoch
    study_name = args.study_name

    # 1. Load a saved model.

    # Create a handler and build model.
    vae = VAEHandler()

    # Load the saved weights
    weights_dir = f"VAE_epoch_{epoch:03}" if epoch is not None else "VAE_best"
    vae.model.load_weights(f"{GLOBAL_CHECKPOINT_DIR}/{study_name}/{weights_dir}/model_weights").expect_partial()

    # The generator is defined as the decoder part only
    generator = vae.model.decoder

    # 2. Prepare data. Get condition values. Sample from the prior (normal distribution) in d dimension (d=latent_dim,
    # latent space dimension). Gather them into tuples. Wrap data in Dataset objects. The batch size must now be set
    # on the Dataset objects. Disable AutoShard.
    e_cond, angle_cond, geo_cond = get_condition_arrays(geometry, energy, events)

    z_r = np.random.normal(loc=0, scale=1, size=(events, vae.latent_dim))

    data = ((z_r, e_cond, angle_cond, geo_cond),)

    data = Dataset.from_tensor_slices(data)

    data = data.batch(BATCH_SIZE_PER_REPLICA)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    data = data.with_options(options)

    # 3. Generate showers using the VAE model.
    generated_events = generator.predict(data) * (energy * 1000)

    # 4. Save the generated showers.
    np.save(f"{GEN_DIR}/VAE_Generated_Geo_{geometry}_E_{energy}_Angle_{angle}.npy", generated_events)


if __name__ == "__main__":
    exit(main())
