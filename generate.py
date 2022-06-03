"""
** generate **
generate showers using a saved VAE model 
"""

# Imports
from configure import Configure
from preprocess import getConditionArrays
from instantiate_model import *
import sys
import argparse
import numpy as np

# parse_args function
"""
    - geometry : name of the calorimter geometry (eg: SiW, SciPb)
    - energyParticle : energy of the primary particle in GeV units
    - angleParticle : angle of the primary particle in degrees
    - nbEvents : number of events to generate
    - epoch: epoch of the saved checkpoint model
"""
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="")
    p.add_argument("--energyParticle", type=int, default="")
    p.add_argument("--angleParticle", type=int, default="")
    p.add_argument("--nbEvents", type=int, default=10000)
    p.add_argument("--epoch", type=int, default="")
    args = p.parse_args()
    return args

# main function 
def main(argv):
	# Parse commandline arguments
	args = parse_args(argv)
	energyParticle = args.energyParticle
	angleParticle = args.angleParticle
	geometry = args.geometry
	nbEvents = args.nbEvents
	epoch = args.epoch
	# Get list of common variables
	variables = Configure()
	# 1. Get condition values
	condE,condAngle,condGeo = getConditionArrays(geometry,energyParticle,angleParticle,nbEvents)
	# 2. Load a saved model 
	vae = instantiate()
	# Load the saved weights
	vae.vae.load_weights('%sVAE-%s.h5'%(variables.checkpoint_dir,epoch) )
	# The generator is defined as the decoder part only
	generator = vae.decoder
	# 3. Generate showers using the VAE model by samplign from the prior (normal distribution) in d dimension (d=latent_dim, latent space dimension)
	zR = np.random.normal(loc=0, scale=1, size=( nbEvents , vae.latent_dim))
	z = np.column_stack((zR, condE[:nbEvents], condAngle[:nbEvents], condGeo[:nbEvents]))
	generatedEvents = (generator.predict(z))*(energyParticle*1000)
	# 4. Save the generated showers
	np.savetxt("%sVAE_Generated_Geo_%s_E_%s_Angle_%s.txt"%(variables.gen_dir,geometry,energyParticle,angleParticle),generatedEvents)

if __name__ == '__main__':
    exit(main(sys.argv[1:]))












