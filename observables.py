"""
** observables **
defines a set of shower observables 
"""

import matplotlib.pyplot as plt
import numpy as np

from configure import Configure

plt.rcParams.update({"font.size": 22})

# Get list of common variables
variables = Configure()


def prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry, y_log_scale):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), clear=True, sharex="all")
    axes[0].scatter(np.arange(variables.nCells_z), g4, label="FullSim", alpha=0.4)
    axes[0].scatter(np.arange(variables.nCells_z), vae, label="MLSim", alpha=0.4)
    if y_log_scale:
        axes[0].set_yscale("log")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Mean energy [Mev]")
    axes[0].set_title(f" $e^-$ , {energy_particle} [GeV], {angle_particle}$^{{\circ}}$, {geometry} ")
    axes[1].plot(np.array(vae) / np.array(g4), "-o")
    axes[1].set_ylabel("MLSim/FullSim")
    axes[1].axhline(y=1, color="black")
    return fig, axes


# longitudinal_profile function plots the longitudinal profile comparing full and fast simulation data of a single
# geometry, energy and angle of primary particles
def longitudinal_profile(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True):
    fig, axes = prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry, y_log_scale)
    axes[0].set_xlabel("Layer index")
    axes[1].set_xlabel("Layer index")
    plt.savefig(f"{save_dir}LongProf_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
    plt.show()


# lateral_profile function plots the lateral profile comparing full and fast simulation data of a single geometry,
# energy and angle of primary particles
def lateral_profile(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True):
    fig, axes = prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry, y_log_scale)
    axes[0].set_xlabel("r index")
    axes[1].set_xlabel("r index")
    plt.savefig(f"{save_dir}LatProf_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
    plt.show()


# Total energy distribution comparing full and fast simulation data of a single geometry, energy and angle of primary
# particles
def e_tot(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True):
    plt.figure(figsize=(12, 8))
    bins = np.linspace(np.min(g4), np.max(vae), 50)
    plt.hist(g4, histtype="step", label="FullSim", bins=bins, color="black")
    plt.hist(vae, histtype="step", label="MLSim", bins=bins, color="red")
    plt.legend()
    if y_log_scale:
        plt.yscale("log")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("# events")
    plt.savefig(f"{save_dir}Etot_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
    plt.show()


# Energy per layer distribution comparing full and fast simulation data of a single geometry, energy and angle of
# primary particles
def energy_layer(g4, vae, energy_particle, angle_particle, geometry, save_dir):
    fig, ax = plt.subplots(5, 9, figsize=(20, 20))
    cpt = 0
    for i in range(5):
        for j in range(9):
            g4_l = np.array([np.sum(i) for i in g4[:, :, :, i, j]])
            vae_l = np.array([np.sum(i) for i in vae[:, :, :, i, j]])
            bins = np.linspace(0, np.max(g4_l), 15)
            n_g4, bins_g4, _ = ax[i][j].hist(g4_l, histtype="step", label="FullSim", bins=bins, color="black")
            n_vae, bins_vae, _ = ax[i][j].hist(vae_l, histtype="step", label="FastSim", bins=bins, color="red")
            ax[i][j].set_title("Layer %s" % cpt, fontsize=12)
            cpt += 1
    plt.savefig("%sELayer_Geo_%s_E_%s_Angle_%s.png" % (save_dir, geometry, energy_particle, angle_particle))
    plt.show()


# Cell energy distribution comparing full and fast simulation data of a single geometry, energy and angle of primary
# particles
def cell_energy(g4, vae, energy_particle, angle_particle, geometry, save_dir):
    def log_energy(events, colour, label):
        all_log_en = []
        for ev in range(len(events)):
            energies = events[ev]
            for en in energies:
                if en > 0:
                    all_log_en.append(np.log10(en))
                else:
                    all_log_en.append(0)
        return plt.hist(all_log_en, bins=np.linspace(-10, 1, 1000), facecolor=colour, histtype="step", label=label)

    plt.figure(figsize=(12, 8))
    log_energy(g4, "b", "FullSim")
    log_energy(vae, "r", "FastSim")
    plt.xlabel("log10(E//MeV)")
    plt.ylim(bottom=1)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.ylabel("# entries")
    plt.grid(True)
    plt.legend()
    plt.savefig("%sCellEDist_Log_Geo_%s_E_%s_Angle_%s.png" % (save_dir, geometry, energy_particle, angle_particle))
    plt.show()
