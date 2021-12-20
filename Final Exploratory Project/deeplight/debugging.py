import torch
from torch import tensor
from torch import linalg as LA
from system_torch import photocore as new_photocore
#from deeplight.system import photocore
import pickle
import sys
import random
import matplotlib.pyplot as plt
from typing import NamedTuple

torch.autograd.set_detect_anomaly(True) # Very useful for debugging and stack-tracing gradients

class photocore_Struct(NamedTuple):
    wavelengths: torch.tensor
    heater_volts: torch.tensor
    index_dicts: torch.nn.ParameterDict
    # dev_structures: as of right now, dev_structure is only ever accessed and 
    # never altered per Heater_FPMod, so we just need one representaiton for it (done in __init__)

vecs = photocore_Struct(wavelengths=tensor([1.3000e-06, 1.3000e-06]), heater_volts=tensor([0., 0.]), index_dicts=tensor([[2.8500-0.5100j, 2.8500-0.5100j],
        [3.5000+0.0000j, 3.5000+0.0000j],
        [1.5000+0.0000j, 1.5000+0.0000j],
        [1.5000+0.0000j, 1.5000+0.0000j],
        [1.3201-12.5880j, 1.3201-12.5880j],
        [1.0000+0.0000j, 1.0000+0.0000j],
        [0.4440-8.2240j, 0.4440-8.2240j]]))

material_index_values = tensor([[1.3201-12.5880j, 1.3201-12.5880j],
        [1.5000+0.0000j, 1.5000+0.0000j],
        [1.5000+0.0000j, 1.5000+0.0000j],
        [1.3201-12.5880j, 1.3201-12.5880j],
        [1.3201-12.5880j, 1.3201-12.5880j],
        [1.0000+0.0000j, 1.0000+0.0000j],
        [0.4440-8.2240j, 0.4440-8.2240j],
        [1.3201-12.5880j, 1.3201-12.5880j],
        [2.8500-0.5100j, 2.8500-0.5100j],
        [1.5000+0.0000j, 1.5000+0.0000j]])

dev_list_thicknesses = tensor([3.0000e-09, 1.1216e-07, 1.5397e-07, 1.0373e-07, 1.0578e-07, 1.0782e-07,
        2.2518e-04, 2.5462e-07, 1.2259e-04, 1.5000e-08])

nm = 1e-9
dev_structure = [('air',0.0*nm), ('al',20*nm), ('sio',30*nm), 
                                ('ito',30*nm), ('sio',15*nm), ('vo2',3*nm),
                                ('sio',15*nm), ('si',40*nm), ('ito',30*nm),
                                ('sio',70*nm), ('al',20*nm), ('air', 0.0*nm)]
n_nodes = 2
cali_num = 2
phcore = new_photocore(dev_structure = dev_structure, n_nodes=n_nodes, materials_oneHot=torch.empty((0)), thicknesses=torch.empty((0))) #add oneHot arg --> if oneHot exists, dont use dev_structure.  
#print("calibration")
phcore.cali_num = cali_num
phcore.calibration()

prop = phcore.Heater_FPMOD_propagation(vecs, material_index_values[6], dev_list_thicknesses[6])
print(prop)
print(phcore.smatrix_to_tmatrix(prop))