"""
** validate **
creates validation plots using shower observables 
"""

# Imports
from configure import Configure
from preprocess import load1E1A1Geo
from observables import *
import sys
import argparse
import numpy as np


# parse_args function
"""
    - geometry : name of the calorimter geometry (eg: SiW, SciPb)
    - energyParticle : energy of the primary particle in GeV units
    - angleParticle : angle of the primary particle in degrees
"""
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="")
    p.add_argument("--energyParticle", type=int, default="")
    p.add_argument("--angleParticle", type=int, default="")
    args = p.parse_args()
    return args

# main function 
def main(argv):
	# Parse commandline arguments
	args = parse_args(argv)
	energyParticle = args.energyParticle
	angleParticle = args.angleParticle
	geometry = args.geometry
	# Get list of common variables
	variables = Configure()
	# 1. Full simulation data loading
	# Load energy of showers from a single geometry, energy and angle
	ELayer_G4 = load1E1A1Geo(variables.init_dir,geometry,energyParticle,angleParticle)
	valid_dir =  variables.valid_dir
	# 2. Fast simulation data loading, scaling to original energy range & reshaping
	VAE_energies = np.loadtxt("%sVAE_Generated_Geo_%s_E_%s_Angle_%s.txt"%(variables.gen_dir,geometry,energyParticle,angleParticle))*(energyParticle*1000) 
	# Reshape the events into 3D
	ELayer_VAE = (VAE_energies).reshape(len(VAE_energies), variables.nCells_r, variables.nCells_phi, variables.nCells_z)  
	# 3. Plot observables 
	LP_G4 = []
	LP_VAE = []
	TP_G4 = []
	TP_VAE = []
	for i in range(variables.nCells_z):
	    LP_G4.append(np.mean(np.array([np.sum(i) for i in ELayer_G4[:, :, :, i]])))
	    LP_VAE.append(np.mean(np.array([np.sum(i) for i in ELayer_VAE[:, :, :, i]])))
	for i in range(variables.nCells_r):
	    TP_G4.append(np.mean(np.array([np.sum(i) for i in ELayer_G4[:, i, :, :]])))
	    TP_VAE.append(np.mean(np.array([np.sum(i) for i in ELayer_VAE[:, i, :, :]])))
	longitudinal_profile(LP_G4,LP_VAE,energyParticle,angleParticle,geometry,valid_dir)
	lateral_profile(TP_G4,TP_VAE,energyParticle,angleParticle,geometry,valid_dir)

if __name__ == '__main__':
    exit(main(sys.argv[1:]))

















