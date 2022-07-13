"""
** generate **
generate showers using a saved VAE model 
"""
import argparse

import numpy as np

from utils.preprocess import get_condition_arrays
from core.constants import CHECKPOINT_DIR, GEN_DIR
from core.model import VAE

"""
    - geometry : name of the calorimeter geometry (eg: SiW, SciPb)
    - energyParticle : energy of the primary particle in GeV units
    - angleParticle : angle of the primary particle in degrees
    - nbEvents : number of events to generate
    - epoch: epoch of the saved checkpoint model
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="")
    p.add_argument("--energyParticle", type=int, default="")
    p.add_argument("--angleParticle", type=int, default="")
    p.add_argument("--nbEvents", type=int, default=10000)
    p.add_argument("--epoch", type=int, default="")
    args = p.parse_args()
    return args


# main function
def main():
    # Parse commandline arguments
    args = parse_args()
    energy_particle = args.energyParticle
    angle_particle = args.angleParticle
    geometry = args.geometry
    nb_events = args.nbEvents
    epoch = args.epoch
    # 1. Get condition values
    cond_e, cond_angle, cond_geo = get_condition_arrays(geometry, energy_particle, nb_events)
    # 2. Load a saved model
    vae = VAE()
    # Load the saved weights
    vae.vae.load_weights(f"{CHECKPOINT_DIR}VAE-{epoch}.h5")
    # The generator is defined as the decoder part only
    generator = vae.decoder
    # 3. Generate showers using the VAE model by sampling from the prior (normal distribution) in d dimension
    # (d=latent_dim, latent space dimension)
    z_r = np.random.normal(loc=0, scale=1, size=(nb_events, vae.latent_dim))
    z = np.column_stack((z_r, cond_e[:nb_events], cond_angle[:nb_events], cond_geo[:nb_events]))
    generated_events = (generator.predict(z)) * (energy_particle * 1000)
    # 4. Save the generated showers
    np.savetxt(
        f"{GEN_DIR}VAE_Generated_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.txt",
        generated_events)


if __name__ == "__main__":
    exit(main())
