"""
** validate **
creates validation plots using shower observables 
"""

import argparse

from core.constants import VALID_DIR, INIT_DIR, GEN_DIR, N_CELLS_PHI
from utils.observables import *
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
    e_layer_g4 = load_showers(INIT_DIR, geometry, energy_particle, angle_particle)
    valid_dir = VALID_DIR
    # 2. Fast simulation data loading, scaling to original energy range & reshaping
    vae_energies = np.loadtxt(
        f"{GEN_DIR}VAE_Generated_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.txt") * (
                           energy_particle * 1000)
    # Reshape the events into 3D
    e_layer_vae = vae_energies.reshape(len(vae_energies), N_CELLS_R, N_CELLS_PHI, N_CELLS_Z)
    # 3. Plot observables
    lp_g4 = []
    lp_vae = []
    tp_g4 = []
    tp_vae = []
    for i in range(N_CELLS_Z):
        lp_g4.append(np.sum(np.array([np.sum(i) for i in e_layer_g4[:, :, :, i]])))
        lp_vae.append(np.sum(np.array([np.sum(i) for i in e_layer_vae[:, :, :, i]])))
    for i in range(N_CELLS_R):
        tp_g4.append(np.sum(np.array([np.sum(i) for i in e_layer_g4[:, i, :, :]])))
        tp_vae.append(np.sum(np.array([np.sum(i) for i in e_layer_vae[:, i, :, :]])))
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
    z_ids = np.arange(N_CELLS_Z)
    r_ids = np.arange(N_CELLS_R)
    fml_g4 = []
    fml_vae = []
    fmt_g4 = []
    fmt_vae = []
    for s_id in range(len(e_layer_g4)):
        e_g4 = [np.sum(e_layer_g4[s_id, :, :, i]) for i in range(N_CELLS_Z)]
        fml_g4.append(np.sum([z_ids[i] * SIZE_Z * e_g4[i] for i in range(N_CELLS_Z)]) / sum_g4[s_id])
        e_vae = [np.sum(e_layer_vae[s_id, :, :, i]) for i in range(N_CELLS_Z)]
        fml_vae.append(np.sum([z_ids[i] * SIZE_Z * e_vae[i] for i in range(N_CELLS_Z)]) / sum_vae[s_id])
        e_g4 = [np.sum(e_layer_g4[s_id, i, :, :]) for i in range(N_CELLS_R)]
        fmt_g4.append(np.sum([r_ids[i] * SIZE_R * e_g4[i] for i in range(N_CELLS_R)]) / sum_g4[s_id])
        e_vae = [np.sum(e_layer_vae[s_id, i, :, :]) for i in range(N_CELLS_R)]
        fmt_vae.append(np.sum([r_ids[i] * SIZE_R * e_vae[i] for i in range(N_CELLS_R)]) / sum_vae[s_id])
    longitudinal_first_moment(fml_g4, fml_vae, energy_particle, angle_particle, geometry, valid_dir)
    lateral_first_moment(fmt_g4, fmt_vae, energy_particle, angle_particle, geometry, valid_dir)


if __name__ == "__main__":
    exit(main())
