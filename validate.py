"""
** validate **
creates validation plots using shower observables 
"""

import argparse

from observables import *
from preprocess import load_showers

# parse_args function
"""
    - geometry : name of the calorimeter geometry (eg: SiW, SciPb)
    - energyParticle : energy of the primary particle in GeV units
    - angleParticle : angle of the primary particle in degrees
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="")
    p.add_argument("--energyParticle", type=int, default="")
    p.add_argument("--angleParticle", type=int, default="")
    args = p.parse_args()
    return args


# main function
def main():
    # Parse commandline arguments
    args = parse_args()
    energy_particle = args.energyParticle
    angle_particle = args.angleParticle
    geometry = args.geometry
    # 1. Full simulation data loading
    # Load energy of showers from a single geometry, energy and angle
    e_layer_g4 = load_showers(variables.init_dir, geometry, energy_particle, angle_particle)
    valid_dir = variables.valid_dir
    # 2. Fast simulation data loading, scaling to original energy range & reshaping
    vae_energies = np.loadtxt(
        f"{variables.gen_dir}VAE_Generated_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.txt") * (
                           energy_particle * 1000)
    # Reshape the events into 3D
    e_layer_vae = vae_energies.reshape(len(vae_energies), variables.nCells_r, variables.nCells_phi, variables.nCells_z)
    # 3. Plot observables
    lp_g4 = []
    lp_vae = []
    tp_g4 = []
    tp_vae = []
    for i in range(variables.nCells_z):
        lp_g4.append(np.mean(np.array([np.sum(i) for i in e_layer_g4[:, :, :, i]])))
        lp_vae.append(np.mean(np.array([np.sum(i) for i in e_layer_vae[:, :, :, i]])))
    for i in range(variables.nCells_r):
        tp_g4.append(np.mean(np.array([np.sum(i) for i in e_layer_g4[:, i, :, :]])))
        tp_vae.append(np.mean(np.array([np.sum(i) for i in e_layer_vae[:, i, :, :]])))
    longitudinal_profile(lp_g4, lp_vae, energy_particle, angle_particle, geometry, valid_dir)
    lateral_profile(tp_g4, tp_vae, energy_particle, angle_particle, geometry, valid_dir)
    g4 = e_layer_g4.reshape(len(e_layer_g4), 40500)
    vae = e_layer_vae.reshape(len(e_layer_vae), 40500)
    sum_g4 = np.array([np.sum(i) for i in g4])
    sum_vae = np.array([np.sum(i) for i in vae])
    e_tot(sum_g4, sum_vae, energy_particle, angle_particle, geometry, valid_dir)
    cell_energy(g4, vae, energy_particle, angle_particle, geometry, valid_dir)
    energy_layer(e_layer_g4.reshape(len(e_layer_g4), 18, 50, 5, 9), e_layer_vae.reshape(len(e_layer_vae), 18, 50, 5, 9),
                 energy_particle, angle_particle, geometry, valid_dir)


if __name__ == "__main__":
    exit(main())
