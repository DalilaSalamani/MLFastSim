"""
** preprocess **
defines the data loading and preprocessing functions 
"""

import h5py
import numpy as np

from configure import Configure

variables = Configure()


# preprocess function loads the data and returns the array of the shower energies and the condition arrays
def preprocess():
    energies_train = []
    cond_e_train = []
    cond_angle_train = []
    cond_geo_train = []
    # This example is trained using 2 detector geometries
    for geo in ["SiW", "SciPb"]:
        dir_geo = variables.init_dir + geo + "/"
        # loop over the angles in a step of 10
        for angleParticle in range(variables.min_angle, variables.max_angle + 10, 10):
            f_name = f"{geo}_angle_{angleParticle}.h5"
            f_name = dir_geo + f_name
            # read the HDF5 file
            h5 = h5py.File(f_name, "r")
            # loop over energies from min_energy to max_energy
            energy_particle = variables.min_energy
            while energy_particle <= variables.max_energy:
                # scale the energy of each cell to the energy of the primary particle (in MeV units) 
                events = np.array(h5[f"{energy_particle}"]) / (energy_particle * 1000)
                energies_train.append(events.reshape(len(events), variables.original_dim))
                # build the energy and angle condition vectors
                cond_e_train.append([energy_particle / variables.max_energy] * len(events))
                cond_angle_train.append([angleParticle / variables.max_angle] * len(events))
                # build the geometry condition vector (1 hot encoding vector)
                if geo == "SiW":
                    cond_geo_train.append([[0, 1]] * len(events))
                if geo == "SciPb":
                    cond_geo_train.append([[1, 0]] * len(events))
                energy_particle *= 2
    # return numpy arrays 
    energies_train = np.concatenate(energies_train)
    cond_e_train = np.concatenate(cond_e_train)
    cond_angle_train = np.concatenate(cond_angle_train)
    cond_geo_train = np.concatenate(cond_geo_train)
    return energies_train, cond_e_train, cond_angle_train, cond_geo_train


# get_condition_arrays function returns condition values from a single geometry, a single energy and angle of primary
# particles
"""
    - geo : name of the calorimeter geometry (eg: SiW, SciPb)
    - energy_particle : energy of the primary particle in GeV units
    - nb_events : number of events
"""


def get_condition_arrays(geo, energy_particle, nb_events):
    cond_e = [energy_particle / variables.max_energy] * nb_events
    cond_angle = [energy_particle / variables.max_energy] * nb_events
    if geo == "SiW":
        cond_geo = [[0, 1]] * nb_events
    else:  # geo == "SciPb"
        cond_geo = [[1, 0]] * nb_events
    return cond_e, cond_angle, cond_geo


# load_showers function loads events from a single geometry, a single energy and angle of primary particles
"""
    - init_dir: the name of the directory which contains the HDF5 files 
    - geo : name of the calorimeter geometry (eg: SiW, SciPb)
    - energy_particle : energy of the primary particle in GeV units
    - angle_particle : angle of the primary particle in degrees

"""


def load_showers(init_dir, geo, energy_particle, angle_particle):
    dir_geo = init_dir + geo + "/"
    f_name = f"{geo}_angle_{angle_particle}.h5"
    f_name = dir_geo + f_name
    # read the HDF5 file
    h5 = h5py.File(f_name, "r")
    energies = np.array(h5[f"{energy_particle}"])
    return energies
