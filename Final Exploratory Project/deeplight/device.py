###################################################
### This library contains all active components ###
### made from PCMs.                             ###
###################################################

import numpy as np 
import math
from scipy import interpolate
import torch

um = 1e-6
nm = 1e-9

# torch interpolate impl
i_k = 40 #interpolate step; larger number means for fine-grained interpolate function 
t_n_sampling = torch.tensor([2.85, 1.39]).view(1,1,2)
t_k_sampling = torch.tensor([0.51, 2.08]).view(1,1,2)
t_f1 = torch.nn.functional.interpolate(t_n_sampling, size=i_k, mode='linear', align_corners=True).view(-1)
t_f2 = torch.nn.functional.interpolate(t_k_sampling, size=i_k, mode='linear', align_corners=True).view(-1)
 
class Heater_FPMod: #(torch.Tensor):

    """ This function describes a FP modulator based on heater.

        The structure is
        Air || Metal(Al) || SiO2 || ITO || SiO2 || VO2 || SiO2 || Si || SiO2 || Metal (Al) || Air
    """
    # @staticmethod
    # def __new__(cls, x, extra_data, *args, **kwargs):
    #     return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, wavelength=1.3*um):
        self.wavelength = wavelength
        # index @ 1.3um

        # self.index_dict = {'si': 3.5 + 0.0j, 'sio': 1.5 + 0.0j, 'ito': 1.5 + 0.0j,
        #                    'al': 1.3201 - 12.588j, 'air': 1.0 + 0.0j, 'gold': 0.444 - 8.224j,
        #                    'vo2': 2.85 - 0.51j}
        self.index_dict = torch.nn.ParameterDict({'si': torch.nn.Parameter(torch.tensor([3.5, 0.0])), 
                                                  'sio': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
                                                  'ito': torch.nn.Parameter(torch.tensor([1.5, 0.0])),
                                                  'al': torch.nn.Parameter(torch.tensor([1.3201, -12.588])), 
                                                  'air': torch.nn.Parameter(torch.tensor([1.0, 0.0])), 
                                                  'gold': torch.nn.Parameter(torch.tensor([0.444, -8.224])),
                                                  'vo2': torch.nn.Parameter(torch.tensor([2.85, -0.51]))})

        # Since dev_structure is only ever used for array accesses to get the string names of different materials,
        # I didn't think updating it to a tensor would make much of a difference. 
        # Might update to tensor later on if we want
        self.dev_structure = [('air',0.0*nm), ('al',20*nm), ('sio',30*nm), 
                              ('ito',30*nm), ('sio',15*nm), ('vo2',3*nm),
                              ('sio',15*nm), ('si',40*nm), ('ito',30*nm),
                              ('sio',70*nm), ('al',20*nm), ('air', 0.0*nm)]

        self.heater_volt = 1.5

        vo2New = self.refra_volt(self.heater_volt)
        self.index_dict['vo2'] = torch.nn.Parameter(torch.tensor([vo2New.real, vo2New.imag]))
        
    # def __torch_function__(self, func, types, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     args = [a._t if hasattr(a, '_t') else a for a in args]
    #     ret = func(*args, **kwargs)
    #     return Heater_FPMod(ret, wavelength=self.wavelength)

    def set_heater(self, heater_volt):
        self.heater_volt = heater_volt
        vo2New = self.refra_volt(self.heater_volt)
        self.index_dict['vo2'] = torch.nn.Parameter(torch.tensor([vo2New.real, vo2New.imag]))

    def refra_volt(self, heater_volt):
        heater_volt_sampling = [0.0, 2.0]
        assert(heater_volt_sampling[0]==0)
        
        # Original interp
        #n_sampling = [2.85, 1.39]
        #k_sampling = [0.51, 2.08]
        #f1 = interpolate.interp1d(heater_volt_sampling, n_sampling)
        #f2 = interpolate.interp1d(heater_volt_sampling, k_sampling)
        #print("norm: ", f1(heater_volt)- 1j*f2(heater_volt), heater_volt)
       
        idx = int(i_k * (heater_volt/heater_volt_sampling[1])) ## in the torch interpolate, we define interpolate steps as i_k. so it up/down sample in the linear mode 
								## with i_k steps (evenly split the range heater_volt_sampling). and output of the torch.interpolate is 
								## a list of vectors sampled (in this case, len() = i_k = 40). The result will be the corresponding index
								## -- which is the position of heater_vol over heater_volt_sampling range, i.e., heater_volt/heater_volt_sampling[1]
								## ;and we multiply with the number of steps i_k to get the final index 
        
        return t_f1[idx].numpy() - 1j*t_f2.numpy()[idx]
        #return f1(heater_volt) - 1j*f2(heater_volt)  #(t_f1.numpy()[-2] - 1j*t_f2.numpy()[-2]) 

    def interface(self, left, right):
        n_l = complex(self.index_dict[left][0], self.index_dict[left][1])
        n_r = complex(self.index_dict[right][0], self.index_dict[right][1])
        
        s11 = 2*n_l/(n_l + n_r)
        s12 = (n_r - n_l)/(n_l + n_r)
        s21 = (n_l - n_r)/(n_l + n_r)
        s22 = 2*n_r/(n_l + n_r)

        smatrix_tensor = torch.tensor([[s11, s12],[s21, s22]])
        
        return smatrix_tensor 
        
    def propagation(self, medium, distance):
        
        n_prop = complex(self.index_dict[medium][0], self.index_dict[medium][1])
        phase_prop = n_prop*2*math.pi/self.wavelength*distance #switched np.pi to math.pi in order to completely get rid of np.
        #s11 = np.exp(-1j*phase_prop)
        s11 = torch.exp(torch.tensor(-1j*phase_prop))
        s12 = 0.0
        s21 = 0.0
        #s22 = np.exp(-1j*phase_prop)
        s22 = torch.exp(torch.tensor(-1j*phase_prop))
        
        #smatrix = np.array([[s11, s12],[s21, s22]])
        smatrix = torch.tensor([[s11.item(), s12],[s21, s22.item()]])

        return smatrix

    def composition(self):

        dev_list = self.dev_structure
        #tmatrix = np.diag([1.0+0.0j,1.0+0.0j])
        #tmatrix = torch.diagflat(torch.tensor([1.0+0.0j,1.0+0.0j])) #torch.diag does not work for complex numbers :(
        tmatrix = torch.tensor([[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]) #made my own diagonal matrix 
        for i in range(len(dev_list)-1):

            if i == 0:
                tmatrix = smatrix_to_tmatrix(self.interface(dev_list[i][0], dev_list[i+1][0]))
            else:
                tmatrix = np.matmul(smatrix_to_tmatrix(self.propagation(dev_list[i][0], dev_list[i][1])), tmatrix)
                tmatrix = np.matmul(smatrix_to_tmatrix(self.interface(dev_list[i][0], dev_list[i+1][0])), tmatrix)

        smatrix = tmatrix_to_smatrix(tmatrix)

        transmittance = torch.abs(smatrix[0][0])**2 
        print("trans:", transmittance)
        return transmittance

class Det():

    def __init__(self, responsivity=1):
        self.responsivity = responsivity

    # def __torch_function__(self, func, types, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     args = [a._t if hasattr(a, 'responsivity') else a for a in args]
    #     ret = func(*args, **kwargs)
    #     return Det(ret, responsivity=self.responsivity)
    
    def output(self,in_pwr):
        return in_pwr*self.responsivity
        
    
def smatrix_to_tmatrix(smatrix):

    s11 = smatrix[0,0]
    s12 = smatrix[0,1]
    s21 = smatrix[1,0]
    s22 = smatrix[1,1]

    detS = s11*s22 - s12*s21

    # Old
    # t11 = detS/s22
    # t12 = s12/s22
    # t21 = -s21/s22
    # t22 = 1/s22
    # #tmatrix = np.array([[t11, t12],[t21, t22]])
    # tmatrix = torch.tensor([[t11.item(), t12.item()],[t21.item(), t22.item()]], dtype=torch.cfloat)

    # Divide in parallel with torch.
    numerator = torch.tensor([[detS.item(), s12.item()], [-s21.item(), 1]])
    tmatrix = torch.div(numerator, s22.item())

    return tmatrix

def tmatrix_to_smatrix(tmatrix):

    t11 = tmatrix[0,0]
    t12 = tmatrix[0,1]
    t21 = tmatrix[1,0]
    t22 = tmatrix[1,1]

    detT = t11*t22 - t12*t21

    # Old
    # s11 = detT/t22
    # s12 = t12/t22
    # s21 = -t21/t22
    # s22 = 1/t22
    # #smatrix = np.array([[s11, s12],[s21, s22]])
    # smatrix = torch.tensor([[s11.item(), s12.item()],[s21.item(), s22.item()]], dtype=torch.cfloat)
    
    # Divide in parallel with torch.
    numerator = torch.tensor([[detT.item(), t12.item()], [-t21.item(), 1]])
    smatrix = torch.div(numerator, t22.item())

    return smatrix


