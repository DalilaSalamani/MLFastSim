"""
** configure **
defines the set of commun variables  
"""

# Configure class
class Configure:
	def __init__(self):
		# Number of calorimeter layers (z-axis segmentation)
		self.nCells_z = 45
		# Segmentation in the r,phi direction
		self.nCells_r = 18
		self.nCells_phi = 50
		# Total number of readout cells (represents the number of nodes in the input/output layers of the model)
		self.original_dim = self.nCells_z*self.nCells_r*self.nCells_phi
		# Minimum and maximum primary particle energy to consider for training in GeV units 
		self.min_energy = 1
		self.max_energy = 1024
		# Minimum and maximum primary particle angle to consider for training in degrees units 
		self.min_angle = 50
		self.max_angle = 90
		# Directory to load the full simulation dataset
		self.init_dir = './dataset/'
		# Directory to save VAE checkpoints
		self.checkpoint_dir =  './checkpoint/'
		# Directory to save model after conversion to a format that can be used in C++
		self.conv_dir = './conversion/'
		# Directory to save validation plots
		self.valid_dir = './validation/'
		# Directory to save VAE generated showers
		self.gen_dir = './generation/'



	
