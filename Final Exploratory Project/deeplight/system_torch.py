## system_torch.py combines the system.py and device.py files.
## In doing so, we could reformat the photocore data into a representation
## that is more efficient for accessing and altering data as needed.
## The new data representation is as follows:
## photocore
##   --> core_vectors struct
##          --> wavelengths : 1 x n_nodes tensor
##          --> heater_volts : 1 x n_nodes tensor
##          --> index_dicts : dictionary with 7 keys (different materials) and
##                            1 x n_nodes tensor values (index_dict value for each node)
##   --> core_weights struct
##          --> wavelengths : n_nodes x n_nodes tensor 
##          --> heater_volts : n_nodes x n_nodes tensor
##          --> index_dicts : dictionary with 7 keys (different materials) and 
##                            n_nodes x n_nodes tensor values (index_dict value for each node)
##   --> core_detectors : 1 x n_nodes tensor
##
## To maintain organization, the overall system_torch.py file is seperated into two portions: 
## 1) Photocore Functions : functions from the original system.py file
## 2) Heater_FPMod Device Functions : functions from the original device.py file. All of these function
##                                    names start with "Heater_FPMOD_" for further clarity. 
## I'd also like to note that for most functions, I've tried to maintain the original code commented out above
## the updated code to serve as a reference and help with code clarity. This can be removed in the future. 

import numpy as np 
import torch
import timeit
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool
import math
from scipy import interpolate
from typing import NamedTuple
import torch.nn.functional as funct
import torch.nn as nn

## Unit constants
um = 1e-6
nm = 1e-9
mW = 1e-3

## Torch interpolate implementation
i_k = 40 # interpolate step; larger number means for fine-grained interpolate function 
t_n_sampling = torch.tensor([2.85, 1.39]).view(1,1,2)
t_k_sampling = torch.tensor([0.51, 2.08]).view(1,1,2)
t_f1 = torch.nn.functional.interpolate(t_n_sampling, size=i_k, mode='linear', align_corners=True).view(-1)
t_f2 = torch.nn.functional.interpolate(t_k_sampling, size=i_k, mode='linear', align_corners=True).view(-1)

# No longer need multiprocessing.pool.Pool due to the parallelism the new data representation provides.
"""
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
"""

# Struct used to represent a series of Heater_FPMod devices together.
# Used to create core_vectors and core_weights in __init__.
# If NamedTuple ever becomes a bottleneck, maybe explore normal class declaration for speedup.
class photocore_Struct(NamedTuple):
    wavelengths: torch.tensor
    heater_volts: torch.tensor
    index_dicts: torch.nn.ParameterDict
    # dev_structures: as of right now, dev_structure is only ever accessed and 
    # never altered per Heater_FPMod, so we just need one representaiton for it (done in __init__)


class photocore:
    ## Photocore Functions ------------------------------------------------------------
    def __init__(self, dev_structure=None, n_nodes=3, wavelength=1.3*um, materials_oneHot=None, thicknesses=None):
        # index @ 1.3um
        self.n_nodes = n_nodes
        self.cali_num = 1000 
        #self.materials = ['vo2', 'si', 'sio', 'ito', 'al', 'air', 'gold'] # still include gold?
        responsivity = 1

        # Device Structure ---------------------------------------------------
        if dev_structure==None: 
            # Original dev_structure (for reference)
            # self.dev_structure = [('air',0.0*nm), ('al',20*nm), ('sio',30*nm), 
            #                   ('ito',30*nm), ('sio',15*nm), ('vo2',3*nm),
            #                   ('sio',15*nm), ('si',40*nm), ('ito',30*nm),
            #                   ('sio',70*nm), ('al',20*nm), ('air', 0.0*nm)]

            # dev_structure is seperated into two components (materials and thicknesses) to allow for automatic torch gradients
            structure_materials = ['air', 'al', 'sio', 'ito', 'sio', 'vo2', 'sio', 'si', 'ito', 'sio', 'al', 'air'] 
            structure_thickness = torch.tensor([0.0*nm, 20*nm, 0.0*nm, 30*nm, 30*nm, 15*nm, 3*nm, 15*nm, 40*nm, 30*nm, 70*nm, 20*nm, 0.0*nm], requires_grad=True)
            
        else:
            structure_materials = [x[0] for x in dev_structure]
            structure_thickness = torch.tensor([x[1] for x in dev_structure], requires_grad=True)

        self.layerNum = len(structure_materials)

        if (not torch.numel(materials_oneHot)):
            #self.dev_structure_materials_encoding = self.init_materials_encoding(structure_materials) # first attempt: delete this. 
            # Use torch.functional one-hot encoding for materials to allow torch gradients for changing material ordering in device structure
            labels = {'vo2':0, 'si':1, 'sio':2, 'ito':3, 'al':4, 'air':5, 'gold':6}
            materials_asInts = torch.tensor(list(map(labels.get, structure_materials)))
            self.oneHot = funct.one_hot(materials_asInts, num_classes=7).T.float() # num_class corresponds with the number of different materials available
            self.oneHot.requires_grad=True
            dev_structure_materials_encoding = self.oneHot.type(torch.double)
            print("No OneHot Provided")
        else:
            self.oneHot = materials_oneHot
            # self.oneHot.retain_grad()
            dev_structure_materials_encoding = self.oneHot
            print("oneHot Provided")

    
        if (not torch.numel(thicknesses)):
            self.dev_structure_thickness = structure_thickness
            print("No Thicknesses Provided")
        else:
            # thicknesses.requires_grad == True
            self.dev_structure_thickness = thicknesses
            # self.dev_structure_thickness.retain_grad()
            print("Thicknesses Provided")

        # Insitantiate one-hot using gumbel softmax. 
        # TODO: Ask Cunxi. Do we want it to be randomized in the case that a structure isn't provided by the user?
        #self.randMat = torch.rand(len(self.materials), len(structure_materials))
        #self.randMat.requires_grad = True
        #print(randMat)
        #gs = funct.gumbel_softmax(self.randMat, tau=1, hard=True, dim=0) 
        gs = funct.gumbel_softmax(dev_structure_materials_encoding*20, tau=10, hard=True, dim=0) # 25 is an arbitrary large number that allows the gumbel softmax to produce the desired hot encoding based on the input structure.
        #print(gs == dev_structure_materials_encoding)
        self.dev_structure_materials = gs
        #self.dev_structure_materials.requires_grad = True
        #self.dev_structure_materials.register_hook(lambda x: print("0 dvm grad: ", x)) 

        # print("inner dsm", self.dev_structure_materials)

        #self.a = nn.Parameter(self.dev_structure_materials)
        #self.a = nn.Module.register_parameter(self, "materials", self.dev_structure_materials)# Parameter(self.dev_structure_materials)
        #nn.Module.register_parameter(self, "materials", self.dev_structure_materials)
        #self.dev_structure_materials.retain_grad()
        #torch.Tensor.retain_grad(self.dev_structure_materials)

        # May be deleted.
        # index assignments: 'air' = 0, 'al' = 1, 'sio' = 2, 'ito' = 3, 'vo2' = 4, 'si' = 5
        #self.materials_list = ['air', 'al', 'sio', 'ito', 'vo2', 'si']
        #print(self.materials_list*self.dev_structure_materials_encoding)
        #self.dev_structure_material_indices = torch.argmax(self.dev_structure_materials_encoding, axis=0)
        #self.dev_structure_material_indices = self.dev_structure_material_indices.float()
        #print(self.dev_structure_material_indices)
        #self.dev_structure_material_indices.requires_grad=True 

        ## Create photocore_Structs's base data structures needed for each attribute of a 
        ## Heater_FPMod (wavelengths, heater_volts, index_dicts)
        # Index_Dict -------------------------------------------------------
        # Original index_dict per Heater_FPMOD (for reference)
        # self.index_dict = {'si': 3.5 + 0.0j, 'sio': 1.5 + 0.0j, 'ito': 1.5 + 0.0j,
        #                    'al': 1.3201 - 12.588j, 'air': 1.0 + 0.0j, 'gold': 0.444 - 8.224j,
        #                    'vo2': 2.85 - 0.51j}

        # New way of initiallizing index_dict: use a normal python dictionary where each value is 
        # a tensor corresponding to approriate size (for vectors --> 1 x n_nodes ; for weights --> n_nodes x n_nodes)
        # Creating keys and values 
        # self.materialKeys = ['vo2', 'si', 'sio', 'ito', 'al', 'air', 'gold']
        materialValues = torch.tensor([[2.85 - 0.51j], [3.5 + 0.0j], [1.5 + 0.0j], [1.5 + 0.0j],
                            [1.3201 - 12.588j], [1.0 + 0.0j], [0.444 - 8.224j]])   
        vectors_tiled_materialValues = materialValues.repeat(1, n_nodes)
        weights_tiled_materialValues = vectors_tiled_materialValues.repeat(1, n_nodes)
        vectors_index_dict = vectors_tiled_materialValues
        weights_index_dict = weights_tiled_materialValues.reshape(7, n_nodes, n_nodes)
        #print(weights_index_dict[0])

        # Combining keys and values into a dictionary
        # self.index_dict = torch.nn.ParameterDict({'si': torch.nn.Parameter(torch.tensor([3.5, 0.0])), 
        #                                           'sio': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
        #                                           'ito': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
        #                                           'al': torch.nn.Parameter(torch.tensor([1.3201, -12.588])), 
        #                                           'air': torch.nn.Parameter(torch.tensor([1.0, 0.0])), 
        #                                           'gold': torch.nn.Parameter(torch.tensor([0.444, -8.224])),
        #                                           'vo2': torch.nn.Parameter(torch.tensor([2.85, -0.51]))})
        # vectors_index_dict = {} #torch.nn.ParameterDict() # dont use parameter dict yet --> later will probably use one-hot selection process instead of dictionaries all together. 
        # weights_index_dict = {}
        # for i in range(len(materialKeys)):
        #     vectors_index_dict[materialKeys[i]] = vectors_tiled_materialValues[i] #torch.nn.Parameter(vectors_tiled_materialValues[i])
        #     weights_index_dict[materialKeys[i]] = weights_tiled_materialValues[i].reshape(n_nodes, n_nodes)
        # print(weights_index_dict)

        # Initialize vo2
        init_heater_volt = torch.tensor(1.5)
        vo2_new = self.Heater_FPMOD_refra_volt(init_heater_volt)
        # 'vo2' is always at index 0.
        vectors_index_dict[0] = vo2_new.repeat(n_nodes) #torch.nn.Parameter(vo2_new.repeat(n_nodes))
        weights_index_dict[0] = vo2_new.repeat(n_nodes, n_nodes)
        # vectors_index_dict['vo2'] = vo2_new.repeat(n_nodes) #torch.nn.Parameter(vo2_new.repeat(n_nodes))
        # weights_index_dict['vo2'] = vo2_new.repeat(n_nodes, n_nodes)
        #print(vectors_index_dict, gs)#, vectors_index_dict*gs)

        # Wavelengths and Heater_Volts -------------------------------------
        # For Vectors dimensions (1 x n_nodes)
        core_vectors_wavelengths = torch.tensor(wavelength).repeat(n_nodes)
        core_vectors_heater_volts = init_heater_volt.repeat(n_nodes)
        
        # For Weights dimensions (n_nodes x n_nodes)
        core_weights_wavelengths = torch.tensor(wavelength).repeat(n_nodes, n_nodes)
        core_weights_heater_volts = init_heater_volt.repeat(n_nodes, n_nodes)  

        ## Initiallize the photocore's vectors and weights as structs to organize data
        self.core_vectors = photocore_Struct(core_vectors_wavelengths, core_vectors_heater_volts, vectors_index_dict) 
        self.core_weights = photocore_Struct(core_weights_wavelengths, core_weights_heater_volts, weights_index_dict) 
        self.core_detectors = torch.tensor(responsivity).repeat(n_nodes)
        

    def simulate(self, in_pwr=100*mW):
        self.input_pwr = in_pwr 
        start = timeit.default_timer()
        v_out_chan = torch.zeros(self.n_nodes) 
        
        """
        # Method 1 : Multiprocessing with Pooling
        p = Pool(20) 
        v_trans_pool = p.map(self.Heater_FPMOD_composition, self.core_vectors) #[self.core_vectors[i] for i in range(self.n_nodes)])
        p.close()
        p.join() 

        v_out_chan = torch.tensor(v_trans_pool)*in_pwr 
        split_out_chans = v_out_chan/self.n_nodes
        """

        """
        # Method 2 : For loop (no multiprocessing)
        for vec_ind in range(self.n_nodes):
            v_trans = Heater_FPMOD_composition(self.core_vectors) #self.vectors[vec_ind].composition()
            v_out_chan[vec_ind] = in_pwr*v_trans
        
        v_out_chan = torch.tensor(v_trans)*in_pwr 
        split_out_chans = v_out_chan/self.n_nodes
        """

        # Method 3 : Parallel direct assignment using new data represenation
        v_trans = self.Heater_FPMOD_composition(self.core_vectors)
        v_out_chan = v_trans*in_pwr 
        split_out_chans = v_out_chan/self.n_nodes
        # For comparison: 
        #print("v_trans_pool \n", v_trans_pool, "\nv_trans \n", v_trans)

        start = timeit.default_timer()
        w_out_mat = torch.zeros(self.n_nodes, self.n_nodes)

        """
        # Method 1 : Multiprocessing with Pooling
        p = Pool(20)
        w_row_line_pool = p.map(Heater_FPMod_composition, [self.weights[i][j] for i in range(self.n_nodes) for j in range(self.n_nodes)])
        p.close()
        p.join()

        split_out_chans_flat = split_out_chans.repeat((self.n_nodes,)) 
        w_out_mat_pool = split_out_chans_flat * w_row_line_pool
        w_out_mat_pool.reshape(-1,self.n_nodes) 
        w_out_mat_pool = torch.reshape(w_out_mat_pool, (-1, self.n_nodes))
        """

        """
        # Method 2 : For loop (no multiprocessing)
        for row_ind in range(self.n_nodes):
            w_row_line = np.zeros(self.n_nodes)
            for col_ind in range(self.n_nodes):
                w_row_line[col_ind] = self.weights[row_ind][col_ind].composition()
            w_out_mat[row_ind, :] = split_out_chans*w_row_line

        # w_out_mat.reshape(-1,self.n_nodes) 
        # w_out_mat = torch.reshape(w_out_mat, (-1, self.n_nodes))

        stop = timeit.default_timer()
        """

        # Method 3 : Parallel direct assignment using new data represenation
        w_row_line_pool = self.Heater_FPMOD_composition(self.core_weights)
        w_out_mat_pool = split_out_chans * w_row_line_pool
        w_out_mat_pool.reshape(-1,self.n_nodes) 
        w_out_mat_pool = torch.reshape(w_out_mat_pool, (-1, self.n_nodes))
        #print("w_out_mat_pool ", w_out_mat_pool)

        sum_out_chans = torch.sum(w_out_mat_pool.T, 0)
        #assert(w_row_line_test == w_row_line)
        #assert(np.equal(sum_out_chans,sum_out_chans_test).all())  ## assertion passed. 08/20/2020 10:20 am [ycunxi]
    
        det_reading = self.core_detectors*sum_out_chans
        #det_reading.register_hook(lambda x: print("det_reading grad: ", x)) 
        return det_reading


    def calibration(self, in_pwr=100*mW):
        """ For simplicity, every modulator is the same """

        self.volt_sweep = torch.linspace(start=0.0, end=1.5, steps=self.cali_num) 
        self.v_curves_w_max = torch.zeros(len(self.volt_sweep)) 
        self.v_curves_w_min = torch.zeros(len(self.volt_sweep))
        self.w_curves_v_max = torch.zeros(len(self.volt_sweep))

        ## 1. set weight at max and sweep vector
        # for row_ind in range(self.n_nodes):
        #     for col_ind in range(self.n_nodes):
        #         self.weights[row_ind][col_ind].set_heater(0.0)
        self.Heater_FPMOD_set_heater(self.core_weights, torch.zeros(self.n_nodes, self.n_nodes))

        for volt_ind in range(len(self.volt_sweep)):
            # self.vectors[0].set_heater(self.volt_sweep[volt_ind])
            # Since the heater_volt is only being set for 1 device, the full set_heater method is not 
            # necessary. Direct assignment can be used.
            self.core_vectors.heater_volts[0] = self.volt_sweep[volt_ind] 
            self.core_vectors.index_dicts[0][0] = self.Heater_FPMOD_refra_volt(self.volt_sweep[volt_ind]) # 'vo2' is always in position 0. could use self.materials.indexOf('vo2') for clarity if desired.
            #self.core_vectors.index_dicts['vo2'][0] = self.Heater_FPMOD_refra_volt(self.volt_sweep[volt_ind])
            self.v_curves_w_max[volt_ind] = self.simulate(in_pwr)[0] 

        #self.vectors[0].set_heater(1.5)
        self.core_vectors.heater_volts[0] = 1.5
        self.core_vectors.index_dicts[0][0] = self.Heater_FPMOD_refra_volt(torch.tensor(1.5))
        #self.core_vectors.index_dicts['vo2'][0] = self.Heater_FPMOD_refra_volt(torch.tensor(1.5))
        
        self.reset()
        
        ## 2. set vector at max and sweep weight
        # for vec_ind in range(self.n_nodes):
        #     self.vectors[vec_ind].set_heater(0.0)
        self.Heater_FPMOD_set_heater(self.core_vectors, torch.zeros(self.n_nodes))

        for volt_ind in range(len(self.volt_sweep)):
            #self.weights[0][0].set_heater(self.volt_sweep[volt_ind])
            self.core_weights.heater_volts[0][0] = self.volt_sweep[volt_ind]
            self.core_weights.index_dicts[0][0][0] = self.Heater_FPMOD_refra_volt(self.volt_sweep[volt_ind])
            #self.core_weights.index_dicts['vo2'][0][0] = self.Heater_FPMOD_refra_volt(self.volt_sweep[volt_ind])
            self.w_curves_v_max[volt_ind] = self.simulate(in_pwr)[0]

        #self.weights[0][0].set_heater(1.5)
        self.core_weights.heater_volts[0][0] = 1.5
        self.core_weights.index_dicts[0][0][0] = self.Heater_FPMOD_refra_volt(torch.tensor(1.5))
        #self.core_weights.index_dicts['vo2'][0][0] = self.Heater_FPMOD_refra_volt(torch.tensor(1.5))

        self.reset()

        ## 3. set weight at min and sweep vector
        for volt_ind in range(len(self.volt_sweep)):
            #self.vectors[0].set_heater(self.volt_sweep[volt_ind])
            self.core_vectors.heater_volts[0] = self.volt_sweep[volt_ind]
            self.core_vectors.index_dicts[0][0] = self.Heater_FPMOD_refra_volt(self.volt_sweep[volt_ind])
            #self.core_vectors.index_dicts['vo2'][0] = self.Heater_FPMOD_refra_volt(self.volt_sweep[volt_ind])
            self.v_curves_w_min[volt_ind] = self.simulate(in_pwr)[0]

        #self.vectors[0].set_heater(1.5)
        self.core_vectors.heater_volts[0] = 1.5
        self.core_vectors.index_dicts[0][0] = self.Heater_FPMOD_refra_volt(torch.tensor(1.5))
        #self.core_vectors.index_dicts['vo2'][0] = self.Heater_FPMOD_refra_volt(torch.tensor(1.5))
        
        ##### Find the correct tuning range ######
        self.v_max_w_max = torch.max(self.v_curves_w_max)
        self.v_min_w_max = torch.min(self.v_curves_w_max)
        self.v_max_w_min = torch.max(self.v_curves_w_min)
        self.v_min_w_min = torch.min(self.v_curves_w_min)
        self.full_range = self.v_max_w_max + self.v_min_w_min - self.v_max_w_min - self.v_min_w_max 


    def reset(self):
        # for vec_ind in range(self.n_nodes):
        #     self.vectors[vec_ind].set_heater(1.5)
        self.Heater_FPMOD_set_heater(self.core_vectors, 1.5*torch.ones(self.n_nodes))

        # for row_ind in range(self.n_nodes):
        #     for col_ind in range(self.n_nodes):
        #         self.weights[row_ind][col_ind].set_heater(1.5) 
        self.Heater_FPMOD_set_heater(self.core_weights, 1.5*torch.ones(self.n_nodes, self.n_nodes))
        

    ## Old Implementation
    # def set_vector_voltage(self,vector,vector_chan=0):
    #     if vector > 0:
    #         target_output = (self.v_max_w_max - self.v_min_w_max) * vector + self.v_min_w_max
    #         set_volt_vec_zero = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max - self.v_min_w_max))]
    #         set_volt_vec = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max - target_output))]
    #     else:
    #         target_output = (self.v_min_w_max - self.v_max_w_max) * np.abs(vector) + self.v_max_w_max
    #         set_volt_vec_zero = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max - self.v_max_w_max))]
    #         set_volt_vec = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max- target_output))]
    #     return set_volt_vec, set_volt_vec_zero
    
    ## Optimized Implementation
    def set_vector_voltages(self,vector,vector_chan=0):
        # Do both if and else in parallel and choose correct one. 
        set_volt_vec_zero = torch.zeros(vector.size())
        set_volt_vec = torch.zeros(vector.size())

        target_output1 = (self.v_max_w_max - self.v_min_w_max) * vector + self.v_min_w_max
        target_output2 = (self.v_min_w_max - self.v_max_w_max) * torch.abs(vector) + self.v_max_w_max
        difference1 = self.v_curves_w_max - self.v_min_w_max
        difference2 = self.v_curves_w_max - self.v_max_w_max
        #print(torch.argmin(torch.abs(self.v_curves_w_max.repeat(self.n_nodes,1) - target_output1.reshape(self.n_nodes,1)), dim=1))
        
        # Shape matrixes appropriately for the parallel subtraction to work correctly in set_volt_vec assignment
        v_curves_v_max_reshaped = self.v_curves_w_max.repeat(self.n_nodes, 1)
        target_output1_reshaped = target_output1.reshape(self.n_nodes, 1)
        target_output2_reshaped = target_output2.reshape(self.n_nodes, 1)

        set_volt_vec[vector > 0] = self.volt_sweep[torch.argmin(torch.abs(v_curves_v_max_reshaped - target_output1_reshaped), dim=1)][vector > 0]
        set_volt_vec[vector <= 0] = self.volt_sweep[torch.argmin(torch.abs(v_curves_v_max_reshaped - target_output2_reshaped), dim=1)][vector <= 0]
        set_volt_vec_zero[vector > 0] = self.volt_sweep[torch.argmin(torch.abs(difference1))]
        set_volt_vec_zero[vector <= 0] = self.volt_sweep[torch.argmin(torch.abs(difference2))]
        return set_volt_vec, set_volt_vec_zero

    ## Old Implementation
    # def set_weight_voltage(self,weight,weight_chan=(0,0)):
    #     w_max = torch.max(self.w_curves_v_max)
    #     w_min = torch.min(self.w_curves_v_max)
    #     if weight > 0:
    #         target_output = (w_max - w_min) * weight + w_min
    #         set_volt_weight_zero = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - self.v_max_w_min))]
    #         set_volt_weight = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - target_output))]
    #     else:
    #         target_output = (w_min - w_max) * np.abs(weight) + w_max
    #         set_volt_weight_zero = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - self.v_max_w_max))]
    #         set_volt_weight = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - target_output))]
    #     return set_volt_weight, set_volt_weight_zero

    ## Optimized Implementation
    def set_weight_voltages(self,weight,weight_chan=(0,0)):
        # Do both if and else in parallel and choose correct one. 
        w_max = torch.max(self.w_curves_v_max)
        w_min = torch.min(self.w_curves_v_max)
        set_volt_weight_zero = torch.zeros(weight.size())
        set_volt_weight = torch.zeros(weight.size())

        target_output1 = (w_max - w_min) * weight + w_min
        target_output2 = (w_min - w_max) * torch.abs(weight) + w_max
        difference1 = self.w_curves_v_max - self.v_max_w_min
        difference2 = self.w_curves_v_max - self.v_max_w_max
        #print(self.volt_sweep[torch.argmin(torch.abs(self.w_curves_v_max.repeat(self.n_nodes,self.n_nodes,1) - target_output1.reshape(self.n_nodes, self.n_nodes, 1)), dim=2)])
        
        # Shape matrixes appropriately for the parallel subtraction to work correctly in set_volt_vec assignment
        w_curves_v_max_reshaped = self.w_curves_v_max.repeat(self.n_nodes,self.n_nodes,1)
        target_output1_reshaped = target_output1.reshape(self.n_nodes, self.n_nodes, 1)
        target_output2_reshaped = target_output2.reshape(self.n_nodes, self.n_nodes, 1)

        set_volt_weight[weight > 0] = self.volt_sweep[torch.argmin(torch.abs(w_curves_v_max_reshaped - target_output1_reshaped), dim=2)][weight > 0]
        set_volt_weight[weight <= 0] = self.volt_sweep[torch.argmin(torch.abs(w_curves_v_max_reshaped - target_output2_reshaped), dim=2)][weight <= 0]
        set_volt_weight_zero[weight > 0] = self.volt_sweep[torch.argmin(torch.abs(difference1))]
        set_volt_weight_zero[weight <= 0] = self.volt_sweep[torch.argmin(torch.abs(difference2))]

        return set_volt_weight, set_volt_weight_zero


    def multiply(self, matrix, vector):
        
        num_size = len(vector)
        set_volt_vec = torch.zeros(self.n_nodes)
        set_volt_vec_zero = torch.zeros(self.n_nodes)
        set_volt_w = torch.zeros([self.n_nodes, self.n_nodes])
        set_volt_w_zero = torch.zeros([self.n_nodes, self.n_nodes])

        # for vec_ind in range(self.n_nodes):
        #     vec_val = vector[vec_ind]
        #     set_volt_vec[vec_ind], set_volt_vec_zero[vec_ind] = self.set_vector_voltage(vec_val)
        set_volt_vec, set_volt_vec_zero = self.set_vector_voltages(vector)

        # Old vs. New Verification:
        # set_volt_vec2 = torch.zeros([self.n_nodes, self.n_nodes])
        # set_volt_vec2_zero = torch.zeros([self.n_nodes, self.n_nodes])
        # set_volt_vec2, set_volt_vec2_zero = self.set_vector_voltages(vector)
        # if (not torch.equal(set_volt_vec, set_volt_vec2)) or (not torch.equal(set_volt_vec_zero, set_volt_vec2_zero)):
        #     print("error in set_weight_voltage")

        # for row_ind in range(self.n_nodes):
        #     for col_ind in range(self.n_nodes):
        #         w_val = matrix[row_ind,col_ind]
        #         set_volt_w[row_ind, col_ind], set_volt_w_zero[row_ind, col_ind] = self.set_weight_voltage(w_val)
        set_volt_w, set_volt_w_zero = self.set_weight_voltages(matrix)
        
        # Old vs. New Verification:
        # set_volt_w2 = torch.zeros([self.n_nodes, self.n_nodes])
        # set_volt_w2_zero = torch.zeros([self.n_nodes, self.n_nodes])
        # set_volt_w2, set_volt_w2_zero = self.set_weight_voltages(matrix)
        # if (not torch.equal(set_volt_w, set_volt_w2)) or (not torch.equal(set_volt_w_zero, set_volt_w2_zero)):
        #     print("error in set_weight_voltage")

        ## o1
        # for row_ind in range(self.n_nodes):
        #     self.vectors[row_ind].set_heater(set_volt_vec[row_ind])
        #     for col_ind in range(self.n_nodes):
        #         self.weights[row_ind][col_ind].set_heater(set_volt_w[row_ind,col_ind])
        self.Heater_FPMOD_set_heater(self.core_vectors, set_volt_vec)
        self.Heater_FPMOD_set_heater(self.core_weights, set_volt_w)
        o1 = self.simulate(self.input_pwr)
        #o1.register_hook(lambda x: print("o1 grad: ", x)) 
        
        ## o2
        # for row_ind in range(self.n_nodes):
        #     self.vectors[row_ind].set_heater(set_volt_vec_zero[row_ind])
        #     for col_ind in range(self.n_nodes):
        #         self.weights[row_ind][col_ind].set_heater(set_volt_w[row_ind,col_ind])
        self.Heater_FPMOD_set_heater(self.core_vectors, set_volt_vec_zero)
        self.Heater_FPMOD_set_heater(self.core_weights, set_volt_w)
        o2 = self.simulate(self.input_pwr)

        ## o3
        # for row_ind in range(self.n_nodes):
        #     self.vectors[row_ind].set_heater(set_volt_vec[row_ind])
        #     for col_ind in range(self.n_nodes):
        #         self.weights[row_ind][col_ind].set_heater(set_volt_w_zero[row_ind,col_ind])
        self.Heater_FPMOD_set_heater(self.core_vectors, set_volt_vec)
        self.Heater_FPMOD_set_heater(self.core_weights, set_volt_w_zero)
        o3 = self.simulate(self.input_pwr)

        ## o4
        # for row_ind in range(self.n_nodes):
        #     self.vectors[row_ind].set_heater(set_volt_vec_zero[row_ind])
        #     for col_ind in range(self.n_nodes):
        #         self.weights[row_ind][col_ind].set_heater(set_volt_w_zero[row_ind,col_ind])
        self.Heater_FPMOD_set_heater(self.core_vectors, set_volt_vec_zero)
        self.Heater_FPMOD_set_heater(self.core_weights, set_volt_w_zero)
        o4 = self.simulate(self.input_pwr)

        #print("o1-o4:", o1,o2,o3,o4, self.full_range)
        output = (o1-o2-o3+o4)/self.full_range
        #output.register_hook(lambda x: print("multiply output grad: ", x)) 
        return output


    def matmul(self, mat1, mat2):
        """
        This is a wrapper around the hardware multiply function
        to run general matrix-matrix multiplication.

        Arguments:
            mat1: N x M numpy matrix
            mat2: M x P numpy matrix
        """
        #print(mat1.ndim, mat2.ndim)
        # matmul assumes that if batched multiplication occurs, it will be done in mat1.
        # improvements? add asserts.
        isBatchedMult = (mat1.ndim == 3)
        if (isBatchedMult): # batched matrix
            batchSize, n1, m1 = mat1.shape
        else:
            batchSize = 1
            n1, m1 = mat1.shape # regular matrix

        m2, p2 = mat2.shape

        assert (m1 == m2), ValueError('Incompatible dimension of matrices')
        
        block_N = int(np.ceil(np.max([n1, m1])/self.n_nodes))
        N_mat = block_N*self.n_nodes

        if (isBatchedMult): # batched matrix
            output_expand = torch.zeros([batchSize, int(N_mat), int(p2)]) 
            mat1_expand = torch.zeros([batchSize, int(N_mat), int(N_mat)])
            mat1_expand[:, 0:n1, 0:m1] = mat1
        else: # regular matrix
            output_expand = torch.zeros([1, int(N_mat), int(p2)]) 
            mat1_expand = torch.zeros([1, int(N_mat), int(N_mat)])
            mat1_expand[0:n1, 0:m1] = mat1

        mat2_expand = torch.zeros([int(N_mat), int(p2)])
        mat2_expand[0:m2,:] = mat2

        for batches in range(batchSize): # loop through each sample in batch. --> this may be further optimized to be done in parallel.
            for col2_ind in range(int(p2)):
                for block_row_ind in range(block_N):
                    for block_col_ind in range(block_N):
                        output_expand[batches, block_row_ind*self.n_nodes:(block_row_ind+1)*self.n_nodes, col2_ind] += \
                                    self.multiply(mat1_expand[batches, block_row_ind*self.n_nodes:(block_row_ind+1)*self.n_nodes, 
                                    block_col_ind*self.n_nodes:(block_col_ind+1)*self.n_nodes], mat2_expand[block_col_ind*self.n_nodes:(block_col_ind+1)*self.n_nodes,col2_ind])    
        
        output = output_expand[0:batchSize, 0:n1, 0:p2] if isBatchedMult else output_expand[0, 0:n1, 0:p2]
        #output.register_hook(lambda x: print("matmul output grad: ", x)) 
        return output 


    ## Heater_FPMod Device Functions --------------------------------------------------
    def Heater_FPMOD_set_heater(self, mat, heater_volts):
        mat.heater_volts[:] = heater_volts[:]
        vo2New = self.Heater_FPMOD_refra_volt(heater_volts)
        mat.index_dicts[0] = vo2New #'vo2' is always at index 0.
        #mat.index_dicts['vo2'] = vo2New

   
    def Heater_FPMOD_refra_volt(self, heater_volts):
        heater_volt_sampling = [0.0, 2.0]
        assert(heater_volt_sampling[0]==0)
    
        idx = (i_k * heater_volts/heater_volt_sampling[1]).long() ## in the torch interpolate, we define interpolate steps as i_k. so it up/down sample in the linear mode 
                        ## with i_k steps (evenly split the range heater_volt_sampling). and output of the torch.interpolate is 
                        ## a list of vectors sampled (in this case, len() = i_k = 40). The result will be the corresponding index
                        ## -- which is the position of heater_vol over heater_volt_sampling range, i.e., heater_volt/heater_volt_sampling[1]
                        ## ;and we multiply with the number of steps i_k to get the final index 
        
        return t_f1[idx] - 1j*t_f2[idx]  # function may get replaced with gumbel soft-max


    def Heater_FPMOD_interface(self, v, left, right, isFirst):
        #if left.shape == torch.tensor(1).shape : # old code
        # n_l = v.index_dicts[left]
        # n_r = v.index_dicts[right]
        n_l = left
        n_r = right
        #print("inter ", n_l)
        # n_l = v.index_dicts[self.materials_list[left]]
        # n_r = v.index_dicts[self.materials_list[right]]
        
        # Slight dimensionality changes need to be addressed based on which vector is passed in (core_vectors or core_weights)
        reshape_dim = self.n_nodes if n_l.size() == torch.Size([self.n_nodes]) else self.n_nodes*self.n_nodes
        #print("s11 b4: \n", (2*n_l/(n_l + n_r)))
        s11 = (2*n_l/(n_l + n_r)).reshape(reshape_dim)
        #print("s11 after: \n", s11)
        s12 = ((n_r - n_l)/(n_l + n_r)).reshape(reshape_dim)
        s21 = ((n_l - n_r)/(n_l + n_r)).reshape(reshape_dim)
        s22 = (2*n_r/(n_l + n_r)).reshape(reshape_dim)
        
        smatrix = torch.stack((s11, s12, s21, s22), 0)
        #print("norm inter ", smatrix.size(), smatrix)
        #smatrix.register_hook(lambda x: print("inter smat grad: ", x)) 

        # else: # new parallelized code
        #     n_l = left
        #     n_r = right
        #     #print("inter ", n_l)

        #     # n_l = v.index_dicts[self.materials_list[left]]
        #     # n_r = v.index_dicts[self.materials_list[right]]
        #     # if n_l.size() == torch.Size([self.layerNum-1, self.n_nodes]):
        #     #     reshape_dim = (self.layerNum-1, self.n_nodes)
        #     # elif n_l.dim() == 1: #torch.Size([self.layerNum-1]):
        #     #     reshape_dim = len(n_l) #(self.layerNum-1) # no reshape
        #     # else:
        #     #     reshape_dim = (self.layerNum-1, self.n_nodes, self.n_nodes)
        #     layerNum = n_l.size()[0]
        #     if isFirst: # processing 0th element
        #         if n_l.dim() == 1: # vectors
        #             reshape_dim = n_l.size()
        #         else: # weights
        #             reshape_dim = (1, self.n_nodes*self.n_nodes)
        #     elif n_l.size() == torch.Size([layerNum, self.n_nodes]): # processing vectors
        #         reshape_dim = (layerNum, self.n_nodes)
        #     else: # processing weights
        #         reshape_dim = (layerNum, self.n_nodes, self.n_nodes)

            
        #     # Slight dimensionality changes need to be addressed based on which vector is passed in (core_vectors or core_weights)
        #     #reshape_dim = (self.layerNum-1, self.n_nodes) if n_l.size() == torch.Size([self.layerNum-1, self.n_nodes]) or n_l.size() == torch.Size([self.layerNum-1]) else (self.layerNum-1, self.n_nodes*self.n_nodes)
        #     #print("s11 parallel b4: \n", (2*n_l/(n_l + n_r)))
        #     s11 = (2*n_l/(n_l + n_r)).reshape(reshape_dim)
        #     #print("s11 parallel after: \n", s11)
        #     s12 = ((n_r - n_l)/(n_l + n_r)).reshape(reshape_dim)
        #     s21 = ((n_l - n_r)/(n_l + n_r)).reshape(reshape_dim)
        #     s22 = (2*n_r/(n_l + n_r)).reshape(reshape_dim)
            
        #     smatrix = torch.stack((s11, s12, s21, s22), 0)
        #     #print("new inter ", smatrix.size(), smatrix)
        
        #print("interface: smatrix representation for all devices: \n", smatrix)
        return smatrix

    def Heater_FPMOD_propagation(self, v, medium, distance):
        #if distance.shape == torch.tensor(1).shape : # old code
        # n_prop = v.index_dicts[medium]
        n_prop = medium
        #print("prop ", n_prop)
        #n_prop = v.index_dicts[self.materials_list[medium]]
        #print(distance.shape, torch.tensor(1).shape)
        phase_prop = n_prop*2*math.pi / (v.wavelengths*distance)
        #print("NORM MATH: \nnumerator = ", n_prop*2*math.pi, "\ndenominator = ", v.wavelengths*distance)
        #print("division = ", phase_prop) 

        # Slight dimensionality changes need to be addressed based on which vector is passed in (core_vectors or core_weights)
        reshape_dim = self.n_nodes if n_prop.size() == torch.Size([self.n_nodes]) else self.n_nodes*self.n_nodes
        s11 = torch.exp(-1j*phase_prop).reshape(reshape_dim)
        s12 = torch.zeros(self.n_nodes) if s11.size() == torch.Size([self.n_nodes]) else torch.zeros(self.n_nodes*self.n_nodes) #(16) 
        s21 = torch.zeros(self.n_nodes) if s11.size() == torch.Size([self.n_nodes]) else torch.zeros(self.n_nodes*self.n_nodes) #(16) #torch.zeros(self.n_nodes)
        s22 = torch.exp(-1j*phase_prop).reshape(reshape_dim)
        
        smatrix = torch.stack((s11, s12, s21, s22), 0)
        #smatrix.register_hook(lambda x: print("prop smat grad: ", x)) 
        #print("norm prop: ", smatrix.size(), smatrix)

        # else: # new parallelized code
        #     n_prop = medium
        #     #print("prop ", n_prop)
        #     layerNum = n_prop.size()[0] #self.layerNum-1

        #     if n_prop.size() == torch.Size([layerNum, self.n_nodes]):
        #         reshape_dim = (layerNum, self.n_nodes)
        #         tileDim = [layerNum, 1]
        #     else:
        #         reshape_dim = (layerNum, self.n_nodes*self.n_nodes)
        #         tileDim = [layerNum, 1, 1]

        #     #print(n_prop.shape, torch.tile(v.wavelengths,tileDim).shape, distance.reshape(tileDim).shape)
        #     phase_prop = n_prop*2*math.pi / torch.tile(v.wavelengths,tileDim)*distance.reshape(tileDim)
        #     #print("NEW MATH: \n numerator = ", n_prop*2*math.pi, "\n denominator = ", torch.tile(v.wavelengths,tileDim)*distance.reshape(tileDim))
        #     #print("division = ", phase_prop)

        #     # Slight dimensionality changes need to be addressed based on which vector is passed in (core_vectors or core_weights)
        #     #reshape_dim = (layerNum, self.n_nodes) if n_prop.size() == torch.Size([layerNum, self.n_nodes]) else (layerNum, self.n_nodes*self.n_nodes)
        #     s11 = torch.exp(-1j*phase_prop).reshape(reshape_dim)
        #     s12 = torch.zeros(layerNum, self.n_nodes) if s11.size() == torch.Size([layerNum, self.n_nodes]) else torch.zeros(layerNum, self.n_nodes*self.n_nodes) #(16) 
        #     s21 = torch.zeros(layerNum, self.n_nodes) if s11.size() == torch.Size([layerNum, self.n_nodes]) else torch.zeros(layerNum, self.n_nodes*self.n_nodes) #(16) #torch.zeros(self.n_nodes)
        #     s22 = torch.exp(-1j*phase_prop).reshape(reshape_dim)
            
        #     smatrix = torch.stack((s11, s12, s21, s22), 0)
        #     #print("new prop: ", smatrix.size(), smatrix)
        #print("propagation: smatrix representation for all devices: \n", smatrix)
        return smatrix


    def Heater_FPMOD_composition(self, vecs):
        ######## using one-hot encoding directly.
        dev_list_materials = self.dev_structure_materials 
        dev_list_thickness = self.dev_structure_thickness
        #print("Device Thickness: \n", dev_list_thickness, "\nDevice Materials Encoding: \n", dev_list_materials)
        
        # Obtain indexDict values for the required materials using matrix multiplication. 
        dev_list_materials_complex = dev_list_materials.clone().type(torch.complex64) # This is necessary to avoid in-place operations in material_indxDict_Values calculation and maintain gradient --> allows for a new memory location to be allocated instead of overwriting variable which breaks gradient chain.
        idxDict_transpose = vecs.index_dicts.clone().T # This is necessary to avoid in-place operations in material_indxDict_Values calculation and maintain gradient
        material_indxDict_Values = torch.matmul(idxDict_transpose, dev_list_materials_complex).T

        #tmatrix = np.diag([1.0+0.0j,1.0+0.0j]) # torch.diag does not work for complex numbers --> made my own diagonal matrix
        tmatrix = torch.tensor([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]], requires_grad=True) 
        
        for i in range(len(dev_list_thickness) - 1):
            if  dev_list_thickness[i] != 0:
                if i == 0:
                    #tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, torch.argmax(dev_list_materials[:,i]), torch.argmax(dev_list_materials[:,i+1]), False))
                    tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, material_indxDict_Values[i], material_indxDict_Values[i+1], False))
                    #print("0 iteration \n", tmatrix.size(), "\n", tmatrix) # for debugging
                else:
                    #print(i, " iteration \n\n") # for debugging
                    #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, torch.argmax(dev_list_materials[:,i]), dev_list_thickness[i])), tmatrix)
                    tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, material_indxDict_Values[i], dev_list_thickness[i])), tmatrix)
                    #print("1st mult \n",i," ", tmatrix) # for debugging
                    # print("mult: \n", self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, torch.argmax(dev_list_materials[:,i]), torch.argmax(dev_list_materials[:,i+1]), False)), tmatrix)

                    #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, torch.argmax(dev_list_materials[:,i]), torch.argmax(dev_list_materials[:,i+1]), False)), tmatrix)
                    tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, material_indxDict_Values[i], material_indxDict_Values[i+1], False)), tmatrix)
                    #print("2nd mult \n",i, " ", tmatrix) # for debugging

        ##### Using encoding for parallelization
        # #print("\n-------------------PARALLEL IMPLEMENTATION --------------------")
        # material_indxDict_Values = torch.matmul(vecs.index_dicts.T, dev_list_materials.type(torch.complex64)).T
        # #print("mat vals: \n", material_indxDict_Values)

        # # for i == 0:
        # i = material_indxDict_Values[0,:] 
        # j = material_indxDict_Values[1,:] # i+1
        # tmatrix0 = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, i, j, True))
        # #print("T0: \n", tmatrix0.size(), "\n", tmatrix0)

        # # for the rest of the indices:
        # i = material_indxDict_Values[1:self.layerNum-1,:] #[:self.layerNum-1,1:]
        # j = material_indxDict_Values[2:,:] # i+1

        # ### Can't be completely parallelized because tmatrix1 should get reassigned in every loop before the matmul..
        # # tmatrix1 = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, i, dev_list_thickness[1:self.layerNum-1])), tmatrix0)
        # # print("T1: \n", tmatrix1)
        # # tmatrix2 = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, i, j, False)), tmatrix1)
        # # print("T2: \n", tmatrix2)

        # ### Parallelize internal functions --> then do loop for matmul
        # T1_prop = self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, i, dev_list_thickness[1:self.layerNum-1]))
        # T2_inter = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, i, j, False))
        # tmatrix1 = tmatrix0
        # for n in range(len(dev_list_thickness) - 2):
        #     # print(tmatrix1)
        #     tmatrix1 = torch.matmul(T1_prop[n], tmatrix1)
        #     # print("T1 looped: \n", tmatrix1)
        #     # print("mult: \n", T2_inter[n], tmatrix1)
        #     tmatrix1 = torch.matmul(T2_inter[n], tmatrix1)
        #     # print("T2 looped: \n", tmatrix1)

        # # New code output
        # # print("new tmatrix: \n", tmatrix1)
        # smatrix_new = self.tmatrix_to_smatrix(tmatrix1)
        # #print("new smatrix: \n", smatrix_new)
        # transmittance_new = torch.abs(smatrix_new[0])**2 
        # transmittance_new = transmittance_new.reshape(self.n_nodes, self.n_nodes) if transmittance_new.size() == torch.Size([self.n_nodes*self.n_nodes]) else transmittance_new
            
        # Original code output
        # print("old tmatrix: \n", tmatrix)
        smatrix = self.tmatrix_to_smatrix(tmatrix)
        #print("old smatrix: \n", smatrix)
        transmittance = torch.abs(smatrix[0])**2 
        transmittance = transmittance.reshape(self.n_nodes, self.n_nodes) if transmittance.size() == torch.Size([self.n_nodes*self.n_nodes]) else transmittance
        #print("transmittance", transmittance)

        #print("Transmittance Comparison: \n", transmittance_new, transmittance)
        #transmittance.register_hook(lambda x: print("transmittance grad: ", x)) 
        #transmittance_new.register_hook(lambda x: print("transmittance new grad: ", x))
        return transmittance



        ######### using one hot as indexes
        # dev_list_materials = self.dev_structure_materials 
        # # print(dev_list_materials)
        # #print(torch.argmax(dev_list_materials[:,6]))
        # print("vecs", vecs)
        # dev_list_thickness = self.dev_structure_thickness
        # #print(dev_list_thickness, dev_list_materials)
        # #tmatrix = np.diag([1.0+0.0j,1.0+0.0j])
        # # torch.diag does not work for complex numbers --> made my own diagonal matrix
        # tmatrix = torch.tensor([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]) 
        # for i in range(len(dev_list_thickness) - 1):
        #     if i == 0:
        #         tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, torch.argmax(dev_list_materials[:,i]), torch.argmax(dev_list_materials[:,i+1])))
        #         #tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list_materials[i], dev_list_materials[i+1]))
        #         #print("0 iteration \n", tmatrix) # for debugging
        #     else:
        #         #print(i, " iteration \n\n") # for debugging
        #         tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, torch.argmax(dev_list_materials[:,i]), dev_list_thickness[i])), tmatrix)
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, dev_list_materials[i], dev_list_thickness[i])), tmatrix)
        #         #print("1st mult \n", tmatrix) # for debugging
                
        #         tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, torch.argmax(dev_list_materials[:,i]), torch.argmax(dev_list_materials[:,i+1]))), tmatrix)
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list_materials[i], dev_list_materials[i+1])), tmatrix)
        #         #print("2nd mult \n", tmatrix) # for debugging

        # smatrix = self.tmatrix_to_smatrix(tmatrix)
        # transmittance = torch.abs(smatrix[0])**2 
        # transmittance = transmittance.reshape(self.n_nodes, self.n_nodes) if transmittance.size() == torch.Size([self.n_nodes*self.n_nodes]) else transmittance
        # #print("transmittance", transmittance)
        # return transmittance

        ######### original
        # #dev_list_materials = self.dev_structure_materials 
        # dev_list_materials = self.materials_decode(self.dev_structure_materials_encoding)
        # #print(self.dev_structure_materials)
        # #print(dev_list_materials)
        # #dev_list_materials_indices = self.dev_structure_material_indices
        # dev_list_thickness = self.dev_structure_thickness
        # #dev_list = self.dev_structure
        
        # #tmatrix = np.diag([1.0+0.0j,1.0+0.0j])
        # # torch.diag does not work for complex numbers --> made my own diagonal matrix
        # tmatrix = torch.tensor([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]) 
        # for i in range(len(self.dev_structure_materials)-1):
        #     #material = self.materials_list[torch.nonzero(self.dev_structure_materials[:, i]).item()]
        #     #materialPlusOne = self.materials_list[torch.nonzero(self.dev_structure_materials[:, i+1]).item()]

        #     if i == 0:
        #         #tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, material, materialPlusOne))
        #         #tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list[i][0], dev_list[i+1][0]))
        #         tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list_materials[i], dev_list_materials[i+1]))
        #         #tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list_materials_indices[i], dev_list_materials_indices[i+1]))
        #         #print("0 iteration \n", tmatrix) # for debugging
        #     else:
        #         #print(i, " iteration \n\n") # for debugging
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, material, dev_list_thickness[i])), tmatrix)
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, dev_list[i][0], dev_list[i][1])), tmatrix)
        #         tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, dev_list_materials[i], dev_list_thickness[i])), tmatrix)
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, dev_list_materials_indices[i], dev_list_thickness[i])), tmatrix)
        #         #print("1st mult \n", tmatrix) # for debugging
                
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, material, materialPlusOne)), tmatrix)
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list[i][0], dev_list[i+1][0])), tmatrix)
        #         tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list_materials[i], dev_list_materials[i+1])), tmatrix)
        #         #tmatrix = torch.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list_materials_indices[i], dev_list_materials_indices[i+1])), tmatrix)
        #         #print("2nd mult \n", tmatrix) # for debugging

        # smatrix = self.tmatrix_to_smatrix(tmatrix)
        # transmittance = torch.abs(smatrix[0])**2 
        # transmittance = transmittance.reshape(self.n_nodes, self.n_nodes) if transmittance.size() == torch.Size([self.n_nodes*self.n_nodes]) else transmittance
        # #print("transmittance", transmittance)
        # return transmittance


    ## Transformation Functions --------------------------------------------------------
    def smatrix_to_tmatrix(self, smatrix):
        #print("smat ", smatrix)
        s11 = smatrix[0] 
        s12 = smatrix[1] 
        s21 = smatrix[2] 
        s22 = smatrix[3] 

        detS = s11*s22 - s12*s21
        #print("detS ", detS)
        
        # Determines whether the vectors or weights are being processed. 
        # Slight variable changes are necessary to accomodate for differeing dimensions
        if s11.dim() > 1: 
            #print(s11)
            layerNum = s11.size()[0]
            # # Vectors
            # if s11.size() == torch.Size([layerNum, self.n_nodes]):
            #     ones_mat = torch.ones(layerNum, self.n_nodes)
            #     reshape_rows = self.n_nodes
            # elif s11.size() == torch.Size([layerNum]):
            #     ones_mat = torch.ones(layerNum)
            #     reshape_rows = 1
            # # Weights
            # else: 
            #     ones_mat = torch.ones(layerNum, self.n_nodes*self.n_nodes)
            #     reshape_rows = self.n_nodes*self.n_nodes

            if s11.size() == torch.Size([layerNum, self.n_nodes]): # processing vectors
                reshape_rows = self.n_nodes
                ones_mat = torch.ones(layerNum, self.n_nodes)
            elif s11.size() == torch.Size([layerNum, self.n_nodes*self.n_nodes]): #processing first weights
                reshape_rows = self.n_nodes*self.n_nodes
                ones_mat = torch.ones(layerNum, self.n_nodes*self.n_nodes)
            else: # processing other weights
                reshape_rows = self.n_nodes*self.n_nodes
                ones_mat = torch.ones(layerNum, self.n_nodes, self.n_nodes)
            
            numerator =  torch.stack((detS, s12, -s21, ones_mat), 0)
            tmatrix = torch.div(numerator, s22)
            #print("tmathere0 ", tmatrix)
            tmatrix = tmatrix.transpose(0, 1) #.reshape([self.layerNum-1, 4, 2])
            #print("reshape ", tmatrix)
            tmatrix = torch.transpose(tmatrix, 1, 2) if s11.size() == torch.Size([s11.size()[0], self.n_nodes]) else torch.transpose(tmatrix, -3, -1)#.transpose(1,2)
            #print("tmathere1 ", tmatrix)
            tmatrix = tmatrix.reshape(layerNum, reshape_rows, 2, 2)

        else: # Old Code
            # Vectors
            if s11.size() == torch.Size([self.n_nodes]):
                ones_mat = torch.ones(self.n_nodes)
                reshape_rows = self.n_nodes
            # Weights
            else: 
                ones_mat = torch.ones(self.n_nodes*self.n_nodes)
                reshape_rows = self.n_nodes*self.n_nodes

            numerator =  torch.stack((detS, s12, -s21, ones_mat), 0)
            if s22[0] == torch.tensor(0.+0.j):
                tmatrix = numerator
            else:
                tmatrix = torch.div(numerator, s22)
            #print("tmathere0 ", tmatrix)
            tmatrix = torch.transpose(tmatrix, 0, 1)
            #print("tmathere1 ", tmatrix)
            tmatrix = tmatrix.reshape(reshape_rows, 2, 2)
        
        #print("tmat ", tmatrix)
        return tmatrix 


    def tmatrix_to_smatrix(self, tmatrix):
        #print("tmat ", tmatrix)
        tmatrix = tmatrix.reshape(self.n_nodes, 4) if tmatrix.size()[0] == self.n_nodes else tmatrix.reshape(self.n_nodes*self.n_nodes, 4)
        tmatrix = torch.transpose(tmatrix, 0, 1)
        
        t11 = tmatrix[0] 
        t12 = tmatrix[1] 
        t21 = tmatrix[2]
        t22 = tmatrix[3] 

        detT = t11*t22 - t12*t21
        #print("detT ", detT)
        
        # Determines whether vectors or weights are being processed. 
        # Slight variable changes are necessary to accomodate for differeing dimensions
        # Vectors
        if t11.size() == torch.Size([self.n_nodes]): 
            ones_mat = torch.zeros(self.n_nodes)
        # Weights
        else: 
            ones_mat = torch.zeros(self.n_nodes*self.n_nodes)
    
        numerator =  torch.stack((detT, t12, -t21, ones_mat), 0)
        if t22[0] == torch.tensor(0.+0.j):
            smatrix = numerator
        else:
            smatrix = torch.div(numerator, t22)
        # smatrix = torch.div(numerator, t22)

        #print("smat ", smatrix)
        return smatrix


    def init_materials_encoding(self, materials=None):
        # index assignments: 'air' = 0, 'al' = 1, 'sio' = 2, 'ito' = 3, 'vo2' = 4, 'si' = 5
        materials_list = self.materials #['air', 'al', 'sio', 'ito', 'vo2', 'si']
        #print(materials)

        # Number of materials depends on the material_list provided above.
        materials_encoding = torch.zeros([len(materials_list), len(materials)]) # number of material types x number of layers
        for i in range(len(materials)):
            position = materials_list.index(materials[i])
            #print(position)
            materials_encoding[position, i] = 1

        #print(materials_encoding)
        #materials_encoding.requires_grad = True
        return materials_encoding

    def materials_decode(self, materials=None):
        materials_list = self.materials 

        indices = torch.argmax(materials, axis=0)
        materials_decoded = []
        for i in range(materials.shape[1]):
            materials_decoded.append(materials_list[indices[i]])

        #print(materials_decoded)
        return materials_decoded




