import numpy as np 
import torch
#from .device import Heater_FPMod, Det
import timeit
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool
import math
from scipy import interpolate
from typing import NamedTuple

## unit constant
um = 1e-6
nm = 1e-9
mW = 1e-3

# turn into function?
# torch interpolate impl
i_k = 40 #interpolate step; larger number means for fine-grained interpolate function 
t_n_sampling = torch.tensor([2.85, 1.39]).view(1,1,2)
t_k_sampling = torch.tensor([0.51, 2.08]).view(1,1,2)
t_f1 = torch.nn.functional.interpolate(t_n_sampling, size=i_k, mode='linear', align_corners=True).view(-1)
t_f2 = torch.nn.functional.interpolate(t_k_sampling, size=i_k, mode='linear', align_corners=True).view(-1)

# How to group data together for organization?
# Could use NamedTuple, but normal class declaration may be faster 
class photocore_Struct(NamedTuple):
    wavelengths: torch.tensor
    heater_volts: torch.tensor
    index_dicts: torch.nn.ParameterDict
    #dev_structures: i don't think this is needed for every Heater_FPMod since 
    # dev_structure is only ever accessed and never altered per Heater_FPMod.


class photocore:
    
    ## Photocore Functions ------------------------------------------------------------
    def __init__(self, n_nodes=3, wavelength=1.3*um):

        ## Device specific variables (per Heater_FPMod) -------------------------------
        # I don't think this data needs to be "self" anymore (except for dev_structure).
        self.wavelength = wavelength
        # index @ 1.3um

        # Original index_dict
        # self.index_dict = {'si': 3.5 + 0.0j, 'sio': 1.5 + 0.0j, 'ito': 1.5 + 0.0j,
        #                    'al': 1.3201 - 12.588j, 'air': 1.0 + 0.0j, 'gold': 0.444 - 8.224j,
        #                    'vo2': 2.85 - 0.51j}

        # Index_dict with torch dictionary
        self.index_dict_ew = torch.nn.ParameterDict({'si': torch.nn.Parameter(torch.tensor([3.5, 0.0])), 
                                                  'sio': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
                                                  'ito': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
                                                  'al': torch.nn.Parameter(torch.tensor([1.3201, -12.588])), 
                                                  'air': torch.nn.Parameter(torch.tensor([1.0, 0.0])), 
                                                  'gold': torch.nn.Parameter(torch.tensor([0.444, -8.224])),
                                                  'vo2': torch.nn.Parameter(torch.tensor([2.85, -0.51]))})
        

        # New way of initiallizing index_dict (without using ParameterDict)
        # normal python dictionary with tensory values. 
        materialKeys = ['si', 'sio', 'ito', 'al', 'air', 'gold', 'vo2']
        materialValues = torch.tensor([[3.5 + 0.0j], [1.5 + 0.0j], [1.5 + 0.0j],
                            [1.3201 - 12.588j], [1.0 + 0.0j], [0.444 - 8.224j],
                            [2.85 - 0.51j]])
        vectors_tiled_materialValues = materialValues.repeat(1, n_nodes)
        weights_tiled_materialValues = vectors_tiled_materialValues.repeat(1, n_nodes)

        vectors_index_dict = {}
        weights_index_dict = {}
        for i in range(len(materialKeys)):
            vectors_index_dict[materialKeys[i]] = vectors_tiled_materialValues[i]
            weights_index_dict[materialKeys[i]] = weights_tiled_materialValues[i].reshape(n_nodes, n_nodes)

        # Since dev_structure is only ever used for array accesses to get the string names of different materials,
        # I didn't think updating it to a tensor would make much of a difference. 
        # Might update to tensor later on if needed
        self.dev_structure = [('air',0.0*nm), ('al',20*nm), ('sio',30*nm), 
                              ('ito',30*nm), ('sio',15*nm), ('vo2',3*nm),
                              ('sio',15*nm), ('si',40*nm), ('ito',30*nm),
                              ('sio',70*nm), ('al',20*nm), ('air', 0.0*nm)]

        self.heater_volt = 1.5

        vo2_new = self.Heater_FPMOD_refra_volt(self.heater_volt)
        vectors_index_dict['vo2'] = torch.tensor(vo2_new).repeat(n_nodes)
        weights_index_dict['vo2'] = torch.tensor(vo2_new).repeat(n_nodes, n_nodes)
        self.index_dict_ew['vo2'] = torch.nn.Parameter(torch.tensor([vo2_new.real, vo2_new.imag]))

        # Dets variable
        self.responsivity = 1

        ## Photocore specific variables -----------------------------------------------
        self.n_nodes = n_nodes
        self.cali_num = 1000 

        # Create photocore's base data structures needed for each 
        # attribute of a Heater_FPMod (wavelengths, heater_volts, index_dicts)
        # (Note: Dev structure is not included in this for now since it is only used for accessing and 
        # is not altered per Heater_FPMod)
        core_wavelengths = []
        core_heater_volts = []
        core_dets = []
        #core_weights_index_dicts = []
        for node_idx in range(n_nodes):
            # can be updated to use torch instead, but okay for now. 
            core_wavelengths.append(self.wavelength)
            core_heater_volts.append(self.heater_volt)
            core_dets.append(self.responsivity)

            # The tiling (done below for weight's wavelengths and heater_volts) does not work 
            # for torch's ParameterDict --> manual repetition
            # core_index_dicts = []
            # for weights_idx in range(4):
            #     core_index_dicts.append(self.index_dict)

            # core_weights_index_dicts.append(core_index_dicts)

            # may be able to use tile again since using python dictionary.
            #core_weights_index_dicts.append(index_dict)
            
        # Tile elements for weights array since it is n_nodes x n_nodes dimensionality
        #core_weights_wavelengths = torch.repeat(tensor.torch(core_wavelengths))
        core_weights_wavelengths = np.tile(core_wavelengths, (n_nodes, 1))
        core_weights_heater_volts = np.tile(core_heater_volts, (n_nodes,1))

        # Convert elements to tensors
        # (Note: index_dicts is not converted to a tensor since it is already a torch.nn.ParameterDict)
        core_wavelengths = torch.tensor(core_wavelengths)
        core_heater_volts = torch.tensor(core_heater_volts)

        # Initiallize the photocore's vectors, weights, and detectors as structs to organize data
        self.core_vectors = photocore_Struct(core_wavelengths, core_heater_volts, vectors_index_dict) #core_index_dicts)
        self.core_weights = photocore_Struct(core_weights_wavelengths, core_weights_heater_volts, weights_index_dict) #core_weights_index_dicts)
        self.core_detectors = torch.tensor(core_dets)

    
    # def simulate(self, in_pwr=100*mW):

    #     self.input_pwr = in_pwr 
    #     start = timeit.default_timer()

    #     v_out_chan = torch.zeros(self.n_nodes) 
    #     p = Pool(20)
    #     v_trans_pool = p.map(Heater_FPMod_composition, [self.vectors[i] for i in range(self.n_nodes)])
    #     p.close()
    #     #p.join()

    #     """
    #     for vec_ind in range(self.n_nodes):
    #         v_trans = Heater_FPMOD_composition(self.core_vectors) #self.vectors[vec_ind].composition()
    #         v_out_chan[vec_ind] = in_pwr*v_trans
    #     """
    #     v_out_chan = torch.tensor(v_trans_pool)*in_pwr 
    #     split_out_chans = torch.div(v_out_chan, self.n_nodes) #v_out_chan/self.n_nodes

    #     ### check equivalence between v_trans_pool and v_trans. 


    #     start = timeit.default_timer()
    #     w_out_mat = torch.zeros(self.n_nodes, self.n_nodes)
    #     p = Pool(20)
    #     w_row_line_pool = p.map(Heater_FPMod_composition, [self.weights[i][j] for i in range(self.n_nodes) for j in range(self.n_nodes)])
    #     p.close()
    #     #p.join()
    #     split_out_chans_flat = split_out_chans.repeat((self.n_nodes,)) #np.tile(split_out_chans,(self.n_nodes,))

    #     """
    #     for row_ind in range(self.n_nodes):
    #         w_row_line = np.zeros(self.n_nodes)
    #         for col_ind in range(self.n_nodes):
    #             w_row_line[col_ind] = self.weights[row_ind][col_ind].composition()
    #         w_out_mat[row_ind, :] = split_out_chans*w_row_line
    #     stop = timeit.default_timer()
    #     """
    #     # Do we still have to do the following underneath? not after switching to for loop
    #     w_out_mat_pool = split_out_chans_flat * torch.tensor(w_row_line_pool)
    #     #print("r ", w_out_mat_pool)
    #     w_out_mat_pool.reshape(-1,self.n_nodes) 
    #     #print("r1 ", w_out_mat_pool)
    #     w_out_mat_pool = torch.reshape(w_out_mat_pool, (-1, self.n_nodes))
    #     #print("r2 ", w_out_mat_pool)

    #     #May I delete the commented out code?
    #     #sum_out_chans = np.zeros(self.n_nodes)
    #     #for row_ind in range(self.n_nodes):
    #     #    sum_out_chans[row_ind] = np.sum(w_out_mat[row_ind,:])
    #     sum_out_chans = torch.sum(w_out_mat_pool.T, 0) #np.add.reduce(w_out_mat_pool.T) 
    #     #print(sum_out_chans, np.add.reduce(w_out_mat_test.reshape(self.n_nodes,self.n_nodes).T))
    #     #assert(w_row_line_test == w_row_line)
    #     #assert(np.equal(sum_out_chans,sum_out_chans_test).all())  ## assertion passed. 08/20/2020 10:20 am [ycunxi]
       


    #     det_reading = torch.zeros(self.n_nodes)
    #     for det_ind in range(self.n_nodes):
    #         det_reading[det_ind] = self.detectors[det_ind].output(sum_out_chans[det_ind])
    #     #print(det_reading)

    #     # Issues with dtype again
    #     # responsivities = map(Det.responsivity, self.detectors)
    #     # det_reading2 = torch.tensor(responsivities) * sum_out_chans
    #     # print(det_reading2)

    #     return det_reading


    def Heater_FPMOD_refra_volt(self, heater_volt):
        heater_volt_sampling = [0.0, 2.0]
        assert(heater_volt_sampling[0]==0)
        
        idx = int(i_k * (heater_volt/heater_volt_sampling[1])) ## in the torch interpolate, we define interpolate steps as i_k. so it up/down sample in the linear mode 
								## with i_k steps (evenly split the range heater_volt_sampling). and output of the torch.interpolate is 
								## a list of vectors sampled (in this case, len() = i_k = 40). The result will be the corresponding index
								## -- which is the position of heater_vol over heater_volt_sampling range, i.e., heater_volt/heater_volt_sampling[1]
								## ;and we multiply with the number of steps i_k to get the final index 

        return t_f1[idx].numpy() - 1j*t_f2.numpy()[idx] 

    # modified to do all devices in "self.core_vectors" at once. 
    def Heater_FPMOD_interface(self, left, right):
        n_l = complex(self.index_dict_ew[left][0], self.index_dict_ew[left][1])
        n_r = complex(self.index_dict_ew[right][0], self.index_dict_ew[right][1])
        print("1 ", n_r)

        s11 = 2*n_l/(n_l + n_r)
        s12 = (n_r - n_l)/(n_l + n_r)
        s21 = (n_l - n_r)/(n_l + n_r)
        s22 = 2*n_r/(n_l + n_r)

        smatrix_tensor = torch.tensor([[s11, s12],[s21, s22]])
        
        return smatrix_tensor


    def Heater_FPMOD_propagation(self, medium, distance):
        
        n_prop = complex(self.index_dict_ew[medium][0], self.index_dict_ew[medium][1])
        phase_prop = n_prop*2*math.pi/self.wavelength*distance #switched np.pi to math.pi in order to completely get rid of np.
        s11 = torch.exp(torch.tensor(-1j*phase_prop))
        s12 = 0.0
        s21 = 0.0
        s22 = torch.exp(torch.tensor(-1j*phase_prop))
        
        smatrix = torch.tensor([[s11.item(), s12],[s21, s22.item()]])

        # print("prop ", smatrix)
        return smatrix


    def Heater_FPMOD_composition(self, vecs):

        dev_list = self.dev_structure
        #tmatrix = np.diag([1.0+0.0j,1.0+0.0j])
        #tmatrix = torch.diagflat(torch.tensor([1.0+0.0j,1.0+0.0j])) #torch.diag does not work for complex numbers :(
        tmatrix = torch.tensor([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]) #made my own diagonal matrix 
        for i in range(len(dev_list)-1):

            if i == 0:
                tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(dev_list[i][0], dev_list[i+1][0]))
                print("0th \n", tmatrix)
            else:
                print(i, "th iter \n\n")
                print("tb4", tmatrix)
                tmatrix = np.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(dev_list[i][0], dev_list[i][1])), tmatrix)
                print("1st \n", tmatrix)
                tmatrix = np.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(dev_list[i][0], dev_list[i+1][0])), tmatrix)
                print("2nd \n", tmatrix)

        smatrix = self.tmatrix_to_smatrix(tmatrix)
        print("smat \n", smatrix)
        transmittance = torch.abs(smatrix[0][0])**2 

        return transmittance


    ## Transformation Functions --------------------------------------------------------

    # updated to do all devices at once. 
    def smatrix_to_tmatrix(self, smatrix):
        print("s ", smatrix)

        s11 = smatrix[0,0]
        s12 = smatrix[0,1]
        s21 = smatrix[1,0]
        s22 = smatrix[1,1]

        detS = s11*s22 - s12*s21

        numerator = torch.tensor([[detS.item(), s12.item()], [-s21.item(), 1]])
        tmatrix = torch.div(numerator, s22.item())

        print("t ", tmatrix)
        return tmatrix

    def tmatrix_to_smatrix(self, tmatrix):
        print("tmat ", tmatrix)
        t11 = tmatrix[0,0]
        t12 = tmatrix[0,1]
        t21 = tmatrix[1,0]
        t22 = tmatrix[1,1]

        detT = t11*t22 - t12*t21
        print("detT ", detT)
        numerator = torch.tensor([[detT.item(), t12.item()], [-t21.item(), 1]])
        smatrix = torch.div(numerator, t22.item())
        print ("smat ", smatrix)
        return smatrix

    def test_function(self):
        #self.Heater_FPMOD_interface(self.core_vectors, "si", "sio")
        print(self.Heater_FPMOD_composition(self.core_weights)) 


class test:
    c = photocore(n_nodes=4)
    c.test_function()
