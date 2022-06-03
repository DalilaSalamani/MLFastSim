"""
** observables **
defines a set of shower observables 
"""

# Imports
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22}) 
import numpy as np
from configure import Configure
# Get list of common variables
variables = Configure()


# longitudinal_profile function plots the longitudinal profile comparing full and fast simulation data of a single geometry, energy and angle of primary particles
def longitudinal_profile(G4,VAE,energyParticle,angleParticle,geometry,saveDir,yLogScale=True):
	fig, axes = plt.subplots(2, 1, figsize=(15, 10), clear=True, sharex=True)
	axes[0].scatter(np.arange(variables.nCells_z), G4, label='FullSim', alpha=0.4)
	axes[0].scatter(np.arange(variables.nCells_z), VAE, label='MLSim', alpha=0.4)
	if(yLogScale):
	    axes[0].set_yscale('log')
	axes[0].legend(loc='upper right')
	axes[0].set_ylabel('Mean energy [Mev]')
	axes[0].set_xlabel('Layer index')
	axes[0].set_title(' $e^-$ , %s [GeV], %s$^{\circ}$, %s ' % (energyParticle, angleParticle, geometry))
	axes[1].plot(np.array(VAE)/np.array(G4), '-o')
	axes[1].set_ylabel('MLSim/FullSim')
	axes[1].set_xlabel('Layer index')
	axes[1].axhline(y=1, color='black')
	plt.savefig("%sLongProf_Geo_%s_E_%s_Angle_%s.png" %(saveDir, geometry, energyParticle, angleParticle))
	plt.show()

# lateral_profile function plots the lateral profile comparing full and fast simulation data of a single geometry, energy and angle of primary particles
def lateral_profile(G4,VAE,energyParticle,angleParticle,geometry,saveDir,yLogScale=True):
	fig, axes = plt.subplots(2, 1, figsize=(15, 10), clear=True, sharex=True)
	axes[0].scatter(np.arange(variables.nCells_r), G4, label='FullSim', alpha=0.4)
	axes[0].scatter(np.arange(variables.nCells_r), VAE, label='MLSim', alpha=0.4)
	if(yLogScale):
	    axes[0].set_yscale('log')
	axes[0].legend(loc='upper right')
	axes[0].set_ylabel('Mean energy [Mev]')
	axes[0].set_xlabel('r index')
	axes[0].set_title(' $e^-$ , %s [GeV], %s$^{\circ}$, %s ' % (energyParticle, angleParticle, geometry))
	axes[1].plot(np.array(VAE)/np.array(G4), '-o')
	axes[1].set_ylabel('MLSim/FullSim')
	axes[1].set_xlabel('r index')
	axes[1].axhline(y=1, color='black')
	plt.savefig("%sLatProf_Geo_%s_E_%s_Angle_%s.png" %(saveDir, geometry, energyParticle, angleParticle))
	plt.show()

