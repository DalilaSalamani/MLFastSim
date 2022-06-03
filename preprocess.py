"""
** preprocess **
defines the data loading and preprocessing functions 
"""

# Imports
from configure import Configure
variables = Configure()
import h5py
import numpy as np

# preprocess function loads the data and returns the array of the shower energies and the condition arrays 
def preprocess(): 
    energies_Train = []
    condE_Train = []
    condAngle_Train = []
    condGeo_Train = []
    # This example is trained using 2 detector geometries
    for geo in [ 'SiW' , 'SciPb'  ]: 
        dirGeo = variables.init_dir + geo + '/'
        # loop over the angles in a step of 10
        for angleParticle in range(variables.min_angle,variables.max_angle+10,10): 
            fName = '%s_angle_%s.h5' %(geo,angleParticle)
            fName = dirGeo + fName
            # read the HDF5 file
            h5 = h5py.File(fName,'r')
            # loop over energies from min_energy to max_energy
            energyParticle = variables.min_energy
            while(energyParticle<=variables.max_energy):
                # scale the energy of each cell to the energy of the primary particle (in MeV units) 
                events = np.array(h5['%s'%energyParticle])/(energyParticle*1000)
                energies_Train.append( events.reshape(len(events),variables.original_dim)  )
                # build the energy and angle condition vectors
                condE_Train.append( [energyParticle/variables.max_energy]*len(events) )
                condAngle_Train.append( [angleParticle/variables.max_angle]*len(events) )
                # build the geometry condition vector (1 hot encoding vector)
                if( geo == 'SiW' ):
                    condGeo_Train.append( [[0,1]]*len(events) )
                if( geo == 'SciPb' ):
                    condGeo_Train.append( [[1,0]]*len(events) )
                energyParticle *=2
    # return numpy arrays 
    energies_Train = np.concatenate(energies_Train)
    condE_Train = np.concatenate(condE_Train)
    condAngle_Train = np.concatenate(condAngle_Train)
    condGeo_Train = np.concatenate(condGeo_Train)
    return energies_Train,condE_Train,condAngle_Train,condGeo_Train 


# getConditionArrays function returns condition values from a single geometry, a single energy and angle of primary particles 
"""
    - geo : name of the calorimter geometry (eg: SiW, SciPb)
    - energyParticle : energy of the primary particle in GeV units
    - angleParticle : angle of the primary particle in degrees
    - nbEvents : number of events
"""
def getConditionArrays(geo,energyParticle,angleParticle,nbEvents):
    condE = [energyParticle/variables.max_energy]*nbEvents
    condAngle = [energyParticle/variables.max_energy]*nbEvents
    if(geo == 'SiW'):
        condGeo = [[0,1]]*nbEvents
    if(geo == 'SciPb'):
        condGeo = [[1,0]]*nbEvents
    return condE,condAngle,condGeo 


# load1E1A1Geo function loads events from a single geometry, a single energy and angle of primary particles 
"""
    - init_dir: the name of the directory which contains the HDF5 files 
    - geo : name of the calorimter geometry (eg: SiW, SciPb)
    - energyParticle : energy of the primary particle in GeV units
    - angleParticle : angle of the primary particle in degrees

"""
def load1E1A1Geo(init_dir,geo,energyParticle,angleParticle):
    dirGeo = init_dir + geo + '/'
    fName = '%s_angle_%s.h5' %(geo,angleParticle)
    fName = dirGeo + fName
    # read the HDF5 file
    h5 = h5py.File(fName,'r')
    energies = np.array(h5['%s'%energyParticle])
    return energies

