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
        # self.core_weights = np.array([self.core_vectors])
        # self.core_weights = np.tile(self.core_vectors, (n_nodes, 1))#self.core_weights.repeat(n_nodes,1)
        self.core_detectors = torch.tensor(core_dets)

    
    def simulate(self, in_pwr=100*mW):

        self.input_pwr = in_pwr 
        start = timeit.default_timer()

        v_out_chan = torch.zeros(self.n_nodes) 
        # p = Pool(20)
        # v_trans_pool = p.map(Heater_FPMod_composition, [self.vectors[i] for i in range(self.n_nodes)])
        # p.close()
        #p.join()
        v_trans = self.Heater_FPMOD_composition(self.core_vectors)

        """
        for vec_ind in range(self.n_nodes):
            v_trans = Heater_FPMOD_composition(self.core_vectors) #self.vectors[vec_ind].composition()
            v_out_chan[vec_ind] = in_pwr*v_trans
        """
        
        v_out_chan = v_trans*in_pwr #torch.tensor(v_trans_pool)*in_pwr 
        split_out_chans = torch.div(v_out_chan, self.n_nodes) #v_out_chan/self.n_nodes

        ### check equivalence between v_trans_pool and v_trans. 


        start = timeit.default_timer()
        w_out_mat = torch.zeros(self.n_nodes, self.n_nodes)
        # p = Pool(20)
        # w_row_line_pool = p.map(Heater_FPMod_composition, [self.weights[i][j] for i in range(self.n_nodes) for j in range(self.n_nodes)])
        # p.close()
        #p.join()
        w_row_line = selt.Heater_FPMOD_composition(self.core_weights)

        # for row_ind in range(self.n_nodes):
        #     w_row_line = np.zeros(self.n_nodes)
        #     w_row_line[row_ind] = self.Heater_FPMOD_composition(self.core_weights[row_ind])
        #     w_out_mat[row_ind, :] = split_out_chans*w_row_line

        split_out_chans_flat = split_out_chans.repeat((self.n_nodes,)) #np.tile(split_out_chans,(self.n_nodes,))

        """
        for row_ind in range(self.n_nodes):
            w_row_line = np.zeros(self.n_nodes)
            for col_ind in range(self.n_nodes):
                w_row_line[col_ind] = self.weights[row_ind][col_ind].composition()
            w_out_mat[row_ind, :] = split_out_chans*w_row_line
        stop = timeit.default_timer()
        """
        # Do we still have to do the following underneath? not after switching to for loop
        w_out_mat_pool = split_out_chans_flat * torch.tensor(w_row_line_pool)
        #print("r ", w_out_mat_pool)
        w_out_mat_pool.reshape(-1,self.n_nodes) 
        #print("r1 ", w_out_mat_pool)
        w_out_mat_pool = torch.reshape(w_out_mat_pool, (-1, self.n_nodes))
        #print("r2 ", w_out_mat_pool)

        #May I delete the commented out code?
        #sum_out_chans = np.zeros(self.n_nodes)
        #for row_ind in range(self.n_nodes):
        #    sum_out_chans[row_ind] = np.sum(w_out_mat[row_ind,:])
        sum_out_chans = torch.sum(w_out_mat_pool.T, 0) #np.add.reduce(w_out_mat_pool.T) 
        #print(sum_out_chans, np.add.reduce(w_out_mat_test.reshape(self.n_nodes,self.n_nodes).T))
        #assert(w_row_line_test == w_row_line)
        #assert(np.equal(sum_out_chans,sum_out_chans_test).all())  ## assertion passed. 08/20/2020 10:20 am [ycunxi]

        det_reading = torch.zeros(self.n_nodes)
        for det_ind in range(self.n_nodes):
            det_reading[det_ind] = self.detectors[det_ind].output(sum_out_chans[det_ind])
        #print(det_reading)

        # Instead of for loop
        # det_output_reading = self.detectors * sum_out_chans
        # print(det_output_reading)

        return det_reading


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
    def Heater_FPMOD_interface(self, v, left, right):
        #n_l = complex(self.index_dict_ew[left][0], self.index_dict_ew[left][1])
        #n_r = complex(self.index_dict_ew[right][0], self.index_dict_ew[right][1])
        n_l = v.index_dicts[left]
        n_r = v.index_dicts[right]
        # print("r ", right, " ", n_r, "\n2 ", n_r2)

        # s11 = 2*n_l/(n_l + n_r)
        # s12 = (n_r - n_l)/(n_l + n_r)
        # s21 = (n_l - n_r)/(n_l + n_r)
        # s22 = 2*n_r/(n_l + n_r)

        reshape_dim = self.n_nodes if n_l.size() == torch.Size([self.n_nodes]) else self.n_nodes*self.n_nodes

        s11 = (2*n_l/(n_l + n_r)).reshape(reshape_dim)
        s12 = ((n_r - n_l)/(n_l + n_r)).reshape(reshape_dim)
        s21 = ((n_l - n_r)/(n_l + n_r)).reshape(reshape_dim)
        s22 = (2*n_r/(n_l + n_r)).reshape(reshape_dim)
        # print("s11 ", s11, "\ns11_2 ", s11_2)
        # print("s12 ", s12, "\ns12_2 ", s12_2)
        # print("s21 ", s21, "\ns21_2 ", s21_2)
        # print("s22 ", s22, "\ns22_2 ", s22_2)

        #smatrix_tensor = torch.tensor([[s11, s12],[s21, s22]])
        #smatrix_tensor_new = torch.stack((s11_2, s12_2, s21_2, s22_2), 1).reshape(16,2,2)
        smatrix_tensor = torch.stack((s11, s12, s21, s22), 0)
        # print(torch.transpose(smatrix_tensor_new, 0, 1))
        #print("interface: original smatrix: \n", smatrix_tensor)
        # c = torch.transpose(smatrix_tensor_new, 0, 1)
        # print(c, "\n", c.reshape(self.n_nodes, 2, 2), "\n", c.reshape(4, 4))
        #print("interface: new smatrix representation for all devices: \n", smatrix_tensor_new)
        #print(torch.stack((s11_2, s12_2, s21_2, s22_2), 1))
        
        # return smatrix_tensor
        return smatrix_tensor


    def Heater_FPMOD_propagation(self, v, medium, distance):
        
        #n_prop = complex(self.index_dict_ew[medium][0], self.index_dict_ew[medium][1])
        n_prop = v.index_dicts[medium]

        #phase_prop = n_prop*2*math.pi/self.wavelength*distance
        phase_prop = n_prop*2*math.pi/self.wavelength*distance
        
        # s11 = torch.exp(torch.tensor(-1j*phase_prop))
        # s12 = 0.0
        # s21 = 0.0
        # s22 = torch.exp(torch.tensor(-1j*phase_prop))

        reshape_dim = self.n_nodes if n_prop.size() == torch.Size([self.n_nodes]) else self.n_nodes*self.n_nodes

        s11 = torch.exp(torch.tensor(-1j*phase_prop)).reshape(reshape_dim)
        s12 = torch.zeros(self.n_nodes) if s11.size() == torch.Size([self.n_nodes]) else torch.zeros(self.n_nodes*self.n_nodes) #(16) 
        s21 = torch.zeros(self.n_nodes) if s11.size() == torch.Size([self.n_nodes]) else torch.zeros(self.n_nodes*self.n_nodes) #(16) #torch.zeros(self.n_nodes)
        s22 = torch.exp(torch.tensor(-1j*phase_prop)).reshape(reshape_dim)
        
        #smatrix = torch.tensor([[s11.item(), s12],[s21, s22.item()]])
        #smatrix_new = torch.stack((s11_2, s12_2, s21_2, s22_2), 1).reshape(16,2,2)
        smatrix = torch.stack((s11, s12, s21, s22), 0)
        #print("propagation: original smatrix: \n", smatrix)
        #print("propagation: new smatrix representation for all devices: \n", smatrix_new)
        
        #print("prop ", smatrix_new)
        #return smatrix
        return smatrix


    def Heater_FPMOD_composition(self, vecs):

        dev_list = self.dev_structure
        #tmatrix = np.diag([1.0+0.0j,1.0+0.0j])
        #tmatrix = torch.diagflat(torch.tensor([1.0+0.0j,1.0+0.0j])) #torch.diag does not work for complex numbers :(
        tmatrix = torch.tensor([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]) #made my own diagonal matrix 
        for i in range(len(dev_list)-1):

            if i == 0:
                tmatrix = self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list[i][0], dev_list[i+1][0]))
                #print("0th \n", tmatrix)
            else:
                #print(i, "th iter \n\n")
                # print("tb4", tmatrix)
                tmatrix = np.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_propagation(vecs, dev_list[i][0], dev_list[i][1])), tmatrix)
                #print("1st \n", tmatrix)
                tmatrix = np.matmul(self.smatrix_to_tmatrix(self.Heater_FPMOD_interface(vecs, dev_list[i][0], dev_list[i+1][0])), tmatrix)
                #print("2nd \n", tmatrix)

        smatrix = self.tmatrix_to_smatrix(tmatrix)
        #print("smat \n", smatrix)
        transmittance = torch.abs(smatrix[0])**2 
        transmittance = transmittance.reshape(self.n_nodes, self.n_nodes) if transmittance.size() == torch.Size([self.n_nodes*self.n_nodes]) else transmittance
        return transmittance


    ## Transformation Functions --------------------------------------------------------

    # updated to do all devices at once. 
    def smatrix_to_tmatrix(self, smatrix):
        # smatrix = torch.transpose(smatrix, 0, 1)
        # smatrix = smatrix.permute(2,0,1)
        #print("s ", smatrix)
        # s11 = smatrix[0,0]
        # s12 = smatrix[0,1]
        # s21 = smatrix[1,0]
        # s22 = smatrix[1,1]

        s11 = smatrix[0] #smatrix[0,0]
        s12 = smatrix[1] #smatrix[0,1]
        s21 = smatrix[2] #smatrix[1,0]
        s22 = smatrix[3] #smatrix[1,1]

        # s11 = smatrix[:,0,0] #smatrix[0,0]
        # s12 = smatrix[:,0,1] #smatrix[0,1]
        # s21 = smatrix[:,1,0] #smatrix[1,0]
        # s22 = smatrix[:,1,1] #smatrix[1,1]
        # print(s11, s12, s21, s22)

        detS = s11*s22 - s12*s21
        #print("detS ", detS)
        #print("s", s11.size())
        #numerator = torch.tensor([[detS.item(), s12.item()], [-s21.item(), 1]])

        # Determines whether the vectors or weights are being processed. 
        # Slight variable changes are necessary to accomodate for differeing dimensions
        # if s11.size() == torch.Size([self.n_nodes]) :
        #     ones_mat = torch.ones(self.n_nodes)
        #     reshape_dim = int(self.n_nodes/2)
        # else :
        #     ones_mat = torch.ones(self.n_nodes, self.n_nodes) #(16)
        #     reshape_dim = self.n_nodes

        if s11.size() == torch.Size([self.n_nodes]): #vectors
            ones_mat = torch.ones(self.n_nodes)
            reshape_rows = self.n_nodes
        else: #weights
            ones_mat = torch.ones(self.n_nodes*self.n_nodes)
            reshape_rows = self.n_nodes*self.n_nodes

        numerator =  torch.stack((detS, s12, -s21, ones_mat), 0)
        #numerator = torch.stack((detS, s12, -s21, ones_mat), 1).reshape(16,2,2)
        #print("numerator ", numerator)
        tmatrix = torch.div(numerator, s22)
        #print("1 ", tmatrix)

        tmatrix = torch.transpose(tmatrix, 0, 1)
        #print("2 ", tmatrix)

        # reshape_dim = 2
        #tmatrix = tmatrix.reshape(self.n_nodes, reshape_dim, reshape_dim)
        tmatrix = tmatrix.reshape(reshape_rows, 2, 2)
        #tmatrix = tmatrix.reshape(16, 2, 2)
        #tmatrix = tmatrix.permute(2,0,1)
    
        # new representation: each column of tmatrix is a unique device. 
        #                     each row represents the coordinates as follows:
        #                     row 0 = s11, row 1 = s12, row 2 = s21, row 3 = s22
        #print("t ", tmatrix)
        return tmatrix 

    def tmatrix_to_smatrix(self, tmatrix):
        #print("tmat ", tmatrix)
        #tmatrix = tmatrix.reshape(self.n_nodes, self.n_nodes) if tmatrix.size() == torch.Size([int(self.n_nodes),int(self.n_nodes/2),int(self.n_nodes/2)]) else tmatrix.reshape(self.n_nodes,self.n_nodes, self.n_nodes)
        tmatrix = tmatrix.reshape(self.n_nodes, 4) if tmatrix.size()[0] == self.n_nodes else tmatrix.reshape(self.n_nodes*self.n_nodes, 4)
        #tmatrix = tmatrix.reshape(4, 4,4)
        #tmatrix = tmatrix.permute(2, 0, 1)
        print(tmatrix)
        tmatrix = torch.transpose(tmatrix, 0, 1)
        #print(tmatrix)

        # t11 = tmatrix[0,0]
        # t12 = tmatrix[0,1]
        # t21 = tmatrix[1,0]
        # t22 = tmatrix[1,1]

        t11 = tmatrix[0] #tmatrix[0,0]
        t12 = tmatrix[1] #tmatrix[0,1]
        t21 = tmatrix[2] #tmatrix[1,0]
        t22 = tmatrix[3] #tmatrix[1,1]

        detT = t11*t22 - t12*t21
        #print("detT ", detT)
        #numerator = torch.tensor([[detT.item(), t12.item()], [-t21.item(), 1]])
        if t11.size() == torch.Size([self.n_nodes]): #vectors
            ones_mat = torch.zeros(self.n_nodes)
        else: #weights
            ones_mat = torch.zeros(self.n_nodes*self.n_nodes)
    
        #ones_mat = torch.zeros(self.n_nodes*self.n_nodes) #if t11.size() == torch.Size([self.n_nodes]) else torch.zeros(self.n_nodes, self.n_nodes)
        numerator =  torch.stack((detT, t12, -t21, ones_mat), 0)
        smatrix = torch.div(numerator, t22)

        #print("smat ", smatrix)

        return smatrix

    def test_function(self):
        #print(self.Heater_FPMOD_interface(self.core_weights, "si", "sio"))
        print(self.Heater_FPMOD_composition(self.core_vectors)) 
        #print(self.core_weights[0])
        print(self.Heater_FPMOD_composition(self.core_weights))
        
       # print(self.simulate())
        #for i in range(4):
        #   print(self.Heater_FPMOD_composition(self.core_weights[i])) 
        

class test:
    c = photocore(n_nodes=2)
    c.test_function()
