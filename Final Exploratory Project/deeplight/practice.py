## The practice.py file is used to try different examples using Pytorch 
## in a simpler execution. It is used to test concepts that are applied to
## the system-torch.py file without having to run the entire system.

import numpy as np 
import torch
#from .device import Heater_FPMod, Det
import timeit
import multiprocessing
import multiprocessing.pool
from torch.multiprocessing import Pool
import math
from scipy import interpolate
from typing import NamedTuple
import torch.nn.functional as F


# t = torch.tensor([[1, 1], [2, 2]])
# t2 = torch.tensor([[3, 3]])
# b = torch.cat((t, t2))
# print(t, t2, b)

## Initializing variables
# x = []
# y = []
# for node_idx in range(4):
#     x.append(1.5)
#     y.append(4.3)
# x = torch.tensor(x)
# y = torch.tensor(y)
# print(x)

## Small struct example
# class MyStruct(NamedTuple):
#     a: torch.tensor
#     b: torch.tensor

# myitem = MyStruct(x, y)
# print("b", myitem.b)



## Struct example as it pertains to system-torch.py
# class vecs(NamedTuple):
#     wavelengths: torch.tensor
#     heater_volts: torch.tensor
#     index_dicts: torch.nn.ParameterDict

# index_dict = torch.nn.ParameterDict({'si': torch.nn.Parameter(torch.tensor([3.5, 0.0])), 
#                                                   'sio': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
#                                                   'ito': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
#                                                   'al': torch.nn.Parameter(torch.tensor([1.3201, -12.588])), 
#                                                   'air': torch.nn.Parameter(torch.tensor([1.0, 0.0])), 
#                                                   'gold': torch.nn.Parameter(torch.tensor([0.444, -8.224])),
#                                                   'vo2': torch.nn.Parameter(torch.tensor([2.85, -0.51]))})


# ## New way of initiallizing index_dict (not using ParameterDict)
# materialKeys = ['si', 'sio', 'ito', 'al', 'air', 'gold', 'vo2']
# materialValues = torch.tensor([[3.5 + 0.0j], [1.5 + 0.0j], [1.5 + 0.0j],
#                     [1.3201 - 12.588j], [1.0 + 0.0j], [0.444 - 8.224j],
#                     [2.85 - 0.51j]])

# tiled_materialValues = materialValues.repeat(1, 4) #np.tile(materialValues, 4) 
# # print(tiled_materialValues)
# # tiled_materialValues[0][0] = 2.0-5j
# # print(tiled_materialValues[0])

# index_dict_new = {}
# for i in range(len(materialKeys)):
#     index_dict_new[materialKeys[i]] = tiled_materialValues[i]
# #print("new\n", index_dict_new)



# wavelength = 1.0
# heater_volt = 2.0
# responsivity = 3.0
# core_wavelengths = []
# core_heater_volts = []
# core_dets = []
# core_weights_index_dicts = []

# for node_idx in range(4):
#     core_wavelengths.append(wavelength)
#     core_heater_volts.append(heater_volt)
#     core_dets.append(responsivity)

#     # core_index_dicts = []
#     # for weights_idx in range(4):
#     #     core_index_dicts.append(index_dict)

#     # core_weights_index_dicts.append(core_index_dicts)
#     core_weights_index_dicts.append(index_dict_new)

    

# #print("ID", core_index_dicts)
# core_weights_wavelengths = np.tile(core_wavelengths, (4, 1))
# core_weights_heater_volts = np.tile(core_heater_volts, (4,1))
# #core_weights_index_dicts = np.repeat(core_index_dicts, 4, axis=0) #repeat doesn't work for parameter dict. 
# #print("tiled", len(core_weights_index_dicts), len(core_weights_index_dicts[0]), core_weights_index_dicts)
# core_wavelengths = torch.tensor(core_wavelengths)
# core_heater_volts = torch.tensor(core_heater_volts)

# # The photocore's vectors, weights, and detectors
# core_vectors = vecs(core_wavelengths, core_heater_volts, index_dict_new) #core_index_dicts)
# core_weights = vecs(core_weights_wavelengths, core_weights_heater_volts, core_weights_index_dicts)
# core_detectors = torch.tensor(core_dets)

# #print("\nvecs", core_vectors)
# #print("\nweights", core_weights.index_dicts[0]['vo2'][2]) # access vo2 of heater_fpmod @ index [0, 2]
# #print("\ndets", core_detectors)

# # Try setting a wavelength
# core_weights.wavelengths[2,1] = 4.0
# #print("\nweights", core_weights.index_dicts[1][1]["vo2"])


# ## Example of how to update functions (Using original loop structures)
# def Heater_FPMOD_set_heater(vecs, vec_idx, heater_volt):
#     vecs.heater_volts[vec_idx] = heater_volt
#     vo2New = 80+9j #self.Heater_FPMOD_refra_volt(self.heater_volt)
#     #vecs.index_dicts[vec_idx]['vo2'] = vo2New #torch.nn.Parameter(torch.tensor([vo2New.real, vo2New.imag]))
#     vecs.index_dicts['vo2'][vec_idx] = vo2New

# print("Before set heater with indexing:\n", core_vectors.heater_volts, core_vectors.index_dicts['vo2'], "\n")
# Heater_FPMOD_set_heater(core_vectors, 2, 6.0)
# print("set heater with indexing:\n", core_vectors.heater_volts, core_vectors.index_dicts['vo2'], "\n")
# #print(core_vectors.index_dicts)


# ## Example of how to update functions (Using assignments, without loops)
# def Heater_FPMOD_set_heater_Parallel(vecs, heater_volts):
#     vecs.heater_volts[:] = heater_volts[:]
#     # If Heater_FPMOD_refra_volt function returned a list of torch paramenters instead...
#     # vo2New = [torch.nn.Parameter(torch.tensor([80.0, 9.0])),
#     #             torch.nn.Parameter(torch.tensor([80.0, 10.0])),
#     #             torch.nn.Parameter(torch.tensor([80.0, 11.0])),
#     #             torch.nn.Parameter(torch.tensor([80.0, 12.0]))]
#     vo2New = torch.tensor([80+9j, 80+10j, 80+11j, 80+12j]) #self.Heater_FPMOD_refra_volt(self.heater_volt)
#     vecs.index_dicts['vo2'] = vo2New
    
#     #vecs.index_dicts[:]['vo2'] = vo2New
#     #vecs.index_dicts[0]['vo2'], vecs.index_dicts[1]['vo2'] = vo2New[0], vo2New[1]
#     #torch.nn.ParameterDict.update(vecs.index_dicts[:]['vo2'], vo2New)
#     #vecs.index_dicts[:]['vo2'] = vo2New[:] #torch.nn.Parameter(torch.tensor([vo2New.real, vo2New.imag]))

# print("Before set heater with assignment:\n", core_vectors.heater_volts, "\n vo2s: ", core_vectors.index_dicts['vo2'])
# heater_volts = torch.tensor([10.0, 11.0, 12.0, 13.0])
# Heater_FPMOD_set_heater_Parallel(core_vectors, heater_volts)
# print("set heater with assignment:\n", core_vectors.heater_volts, "\n vo2s: ", core_vectors.index_dicts['vo2'])


##
print("autograd on parameter dict")
index_dict = torch.nn.ParameterDict(parameters={('si', torch.nn.Parameter(torch.tensor([3.5, 0.0]))), 
                                    ('sio', torch.nn.Parameter(torch.tensor([1.5, 0.0]))),
                                    ('ito', torch.nn.Parameter(torch.tensor([1.5, 0.0]))),
                                    # 'al': torch.nn.Parameter(torch.tensor([1.3201, -12.588])), 
                                    # 'air': torch.nn.Parameter(torch.tensor([1.0, 0.0])), 
                                    # 'gold': torch.nn.Parameter(torch.tensor([0.444, -8.224])),
                                    ('vo2', torch.nn.Parameter(torch.tensor([2.85, -0.51])))})

print("dict :\n", index_dict)
grad = torch.rand(2)
grads = [torch.rand(2), torch.rand(2), grad, torch.rand(2), grad, grad, grad]
#print(grads)
print("keys :\n", list(index_dict.values()))
print("vals :\n", list(index_dict.keys()))

print("Grad ", torch.autograd.backward(index_dict.values(), grads))


## 
print("complex number matrix multiplication")
index_dict_keys = list(index_dict.keys())
index_dict_vals = list(index_dict.values())
mat1 = torch.tensor([[1.0+5j, 1.5+10j], [2.0+5j, 2.5+10j]])
mat2 = torch.tensor([[1+1j, 2+1j], [3+1j, 4+1j]])
ans = np.matmul(mat1, mat2)
print("numpy complex mult: \n", ans)

#torch doesn't work on complex numbers ON MY LATPTOP cuz of CPU restraints (should work on others)
# ans2 = torch.matmul(mat1, mat2)
# print(ans2)

# # split up complex numbers and try torch.mm? 
# real = torch.matmul(mat1.real, mat2.real)
# img = torch.matmul(mat1.imag, mat2.imag)
# print("torch complex mult: \n", torch.complex(real, img))
                

# accessing
nm = 10^-9
dev_structure = [('air',0.0*nm), ('al',20*nm), ('sio',30*nm), 
                              ('ito',30*nm), ('sio',15*nm), ('vo2',3*nm),
                              ('sio',15*nm), ('si',40*nm), ('ito',30*nm),
                              ('sio',70*nm), ('al',20*nm), ('air', 0.0*nm)]
keys = [x[0] for x in dev_structure]
print("KEYS: ", keys)
labels = {'air':0, 'al':1, 'sio':2, 'ito':3, 'vo2':4, 'si':5}
keys_ints = torch.tensor(list(map(labels.get, keys)))
print(keys_ints)
oneHot = F.one_hot(keys_ints)
print(oneHot)

#print(F.gumbel_softmax(oneHot, tau=1, hard=True))




### gumbel soft max
logits = torch.randn(6, 3)
print(logits)
# Sample soft categorical using reparametrization trick:
print(F.gumbel_softmax(logits, tau=1, hard=False))
# Sample hard categorical using "Straight-through" trick:
print(F.gumbel_softmax(logits, tau=1, hard=True))

a = [[0, 1, 1], 
     [1, 0, 0],
     [0, 0, 0]]
b = ['a', 'b']
c = [1, 2, 3]
#print(a*b)
# print("t", torch.tensor(a)[torch.tensor([1,0,1])])
# print(torch.tensor(a)*torch.tensor(c))
# print(torch.matmul(torch.tensor(a),torch.tensor(c)))

m = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 1., 1., 1., 0., 0., 1., 0., 0., 0.],
                [1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
r = torch.arange(0,6, dtype=torch.float)
#print(torch.matmul(r,m))
#print(r, r[1:], r[2:], torch.tile(r, [2,1]), m.shape[0])

mat = torch.tensor([[1, 1, 1], 
                [2, 2, 2],
                [3, 3, 3]])

#print(torch.gather(t, 0, torch.tensor([[1],[0],[1]])))
#print(torch.nonzero(m[:, 0]).item())

v = torch.tensor([[2, 2],
                  [3, 3]]) #2 materials vecs
w = torch.tensor([[[4, 4],
                   [5, 5]],
                  [[6, 6],
                   [7, 7]]]) #2 materials weights

enc = torch.tensor([[0, 1, 1],
                    [1, 0, 0]]) #2 materials, 3 layers encoding example

# print(enc[:,0].repeat(2, 1).transpose(0,1))
# # even in expanding... still not sure how I'd get rid of indexing since its in a for loop? -> don't use for loop? parallelize it?
# index_expand = torch.index_select(enc, 1, torch.tensor([0])).repeat_interleave(2, dim=1)
# print("single idx expand: \n", index_expand)
# print(torch.mm(index_expand,v))
# # attempy full expand for parallelization
# full_expand = enc.repeat_interleave(2, dim=1) # 2 is n_nodes - corresponds to dimensionality of v (index_dicts)
# print("full expand: \n", full_expand, "\n", full_expand.reshape(3, 2, 2))

# print(torch.matmul(v.T, enc).T)
# print(torch.matmul(w.T, enc).T)

print(w.reshape([2, 1, 4]))
print("W", w.transpose(0, 1), w.T, torch.transpose(w, 0,1))










