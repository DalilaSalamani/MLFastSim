"""
** observables **
defines a set of shower observables 
"""

import matplotlib.pyplot as plt
import numpy as np

from configure import Configure

plt.rcParams.update({'font.size': 22})

# Get list of common variables
variables = Configure()


# longitudinal_profile function plots the longitudinal profile comparing full and fast simulation data of a single
# geometry, energy and angle of primary particles
def longitudinal_profile(G4, VAE, energyParticle, angleParticle, geometry, saveDir, yLogScale=True):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), clear=True, sharex='all')
    axes[0].scatter(np.arange(variables.nCells_z), G4, label='FullSim', alpha=0.4)
    axes[0].scatter(np.arange(variables.nCells_z), VAE, label='MLSim', alpha=0.4)
    if (yLogScale):
        axes[0].set_yscale('log')
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Mean energy [Mev]')
    axes[0].set_xlabel('Layer index')
    axes[0].set_title(' $e^-$ , %s [GeV], %s$^{\circ}$, %s ' % (energyParticle, angleParticle, geometry))
    axes[1].plot(np.array(VAE) / np.array(G4), '-o')
    axes[1].set_ylabel('MLSim/FullSim')
    axes[1].set_xlabel('Layer index')
    axes[1].axhline(y=1, color='black')
    plt.savefig("%sLongProf_Geo_%s_E_%s_Angle_%s.png" % (saveDir, geometry, energyParticle, angleParticle))
    plt.show()


# lateral_profile function plots the lateral profile comparing full and fast simulation data of a single geometry,
# energy and angle of primary particles
def lateral_profile(G4, VAE, energyParticle, angleParticle, geometry, saveDir, yLogScale=True):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), clear=True, sharex=True)
    axes[0].scatter(np.arange(variables.nCells_r), G4, label='FullSim', alpha=0.4)
    axes[0].scatter(np.arange(variables.nCells_r), VAE, label='MLSim', alpha=0.4)
    if (yLogScale):
        axes[0].set_yscale('log')
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Mean energy [Mev]')
    axes[0].set_xlabel('r index')
    axes[0].set_title(' $e^-$ , %s [GeV], %s$^{\circ}$, %s ' % (energyParticle, angleParticle, geometry))
    axes[1].plot(np.array(VAE) / np.array(G4), '-o')
    axes[1].set_ylabel('MLSim/FullSim')
    axes[1].set_xlabel('r index')
    axes[1].axhline(y=1, color='black')
    plt.savefig("%sLatProf_Geo_%s_E_%s_Angle_%s.png" % (saveDir, geometry, energyParticle, angleParticle))
    plt.show()


# Total energy distribution comparing full and fast simulation data of a single geometry, energy and angle of primary
# particles
def Etot(G4, VAE, energyParticle, angleParticle, geometry, saveDir, yLogScale=True):
    plt.figure(figsize=(12, 8))
    bins = np.linspace(np.min(G4), np.max(VAE), 50)
    plt.hist(G4, histtype='step', label='FullSim', bins=bins, color='black')
    plt.hist(VAE, histtype='step', label='MLSim', bins=bins, color='red')
    plt.legend()
    if (yLogScale):
        plt.yscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('# events')
    plt.savefig("%sEtot_Geo_%s_E_%s_Angle_%s.png" % (saveDir, geometry, energyParticle, angleParticle))
    plt.show()


# Energy per layer distribution comparing full and fast simulation data of a single geometry, energy and angle of
# primary particles
def energy_layer(G4, VAE, energyParticle, angleParticle, geometry, saveDir, yLogScale=True):
    fig, ax = plt.subplots(5, 9, figsize=(20, 20))
    cpt = 0
    for i in range(5):
        for j in range(9):
            g4_l = np.array([np.sum(i) for i in G4[:, :, :, i, j]])
            vae_l = np.array([np.sum(i) for i in VAE[:, :, :, i, j]])
            bins = np.linspace(0, np.max(g4_l), 15)
            n_g4, bins_g4, _ = ax[i][j].hist(g4_l, histtype='step', label='FullSim', bins=bins, color='black')
            n_vae, bins_vae, _ = ax[i][j].hist(vae_l, histtype='step', label='FastSim', bins=bins, color='red')
            bin_width_g4 = bins_g4[1] - bins_g4[0]
            bin_width_vae = bins_vae[1] - bins_vae[0]
            ax[i][j].set_title("Layer %s" % cpt, fontsize=12)
            cpt += 1
    plt.savefig("%sELayer_Geo_%s_E_%s_Angle_%s.png" % (saveDir, geometry, energyParticle, angleParticle))
    plt.show()


# Cell energy distribution comparing full and fast simulation data of a single geometry, energy and angle of primary
# particles
def cell_energy(G4, VAE, energyParticle, angleParticle, geometry, saveDir, yLogScale=True):
    def log_energy(events, colour, label):
        all_logEn = []
        for ev in range(len(events)):
            energies = events[ev]
            for en in energies:
                if (en > 0):
                    all_logEn.append(np.log10(en))
                else:
                    all_logEn.append(0)
        return plt.hist(all_logEn, bins=np.linspace(-10, 1, 1000), facecolor=colour, histtype='step', label=label)

    plt.figure(figsize=(12, 8))
    log_energy(G4, 'b', 'FullSim')
    log_energy(VAE, 'r', 'FastSim')
    plt.xlabel('log10(E//MeV)')
    plt.ylim(bottom=1)
    plt.yscale('log')
    plt.ylim(bottom=1)
    plt.ylabel('# entries')
    plt.grid(True)
    plt.legend()
    plt.savefig("%sCellEDist_Log_Geo_%s_E_%s_Angle_%s.png" % (saveDir, geometry, energyParticle, angleParticle))
    plt.show()
