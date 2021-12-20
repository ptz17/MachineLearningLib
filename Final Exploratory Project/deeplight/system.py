import numpy as np 
import torch
from .device import Heater_FPMod, Det
import timeit
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool


## unit constant
um = 1e-6
nm = 1e-9
mW = 1e-3
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


class photocore:
    
    def __init__(self, n_nodes=3, wavelength=1.3*um):

        self.n_nodes = n_nodes
        self.cali_num = 1000 
        ## Initialization of vectors 
        self.vectors = []
        for vec_ind in range(n_nodes):
            self.vectors.append(Heater_FPMod(wavelength=wavelength))

        # Attempting to create a tensor of Heater_FPMods
        #print("Orig ", self.vectors)
        #vecs = torch.as_tensor(self.vectors)
        #vecs = torch.from_numpy(np.array(Heater_FPMod(wavelength=wavelength)))
        #print("torch ", vecs)

        # Tiling does work to create an array of Heater_FPMods, however, will not work in 
        # the long run because the elements share the same reference. 
        # The code below shows that setting one element of a tiled array of Heater_FPMods will
        # set that value for all elements.
        #vecs = np.tile([Heater_FPMod(wavelength=wavelength)], n_nodes)
        #self.vectors[1].set_heater(1.9) # original vectors
        #print(self.vectors[0].heater_volt, self.vectors[1].heater_volt)
        #vecs[1].set_heater(1.8) # tiled vecs
        #print(vecs[0].heater_volt, vecs[1].heater_volt)


        ## Initialization of weights
        self.weights = []
        for row_ind in range(n_nodes):
            row_weights = []
            for col_ind in range(n_nodes):
                row_weights.append(Heater_FPMod(wavelength=wavelength))
            self.weights.append(row_weights) 

        ## Initialization of detector
        self.detectors = []
        for det_ind in range(n_nodes):
            self.detectors.append(Det(responsivity=1))
    

    def simulate(self, in_pwr=100*mW):

        self.input_pwr = in_pwr 
        start = timeit.default_timer()

        v_out_chan = torch.zeros(self.n_nodes) #np.zeros(self.n_nodes)
        p = Pool(20)
        v_trans_pool = p.map(Heater_FPMod.composition, [self.vectors[i] for i in range(self.n_nodes)])
        p.close()
        #p.join()
        """
        for vec_ind in range(self.n_nodes):
            v_trans = self.vectors[vec_ind].composition()
            v_out_chan[vec_ind] = in_pwr*v_trans
        """
        v_out_chan = torch.tensor(v_trans_pool)*in_pwr #np.array(v_trans_pool)*in_pwr
        split_out_chans = torch.div(v_out_chan, self.n_nodes) #v_out_chan/self.n_nodes

        #print("Old v_trans_pool \n", v_trans_pool)

        start = timeit.default_timer()
        w_out_mat = torch.zeros(self.n_nodes, self.n_nodes) #np.zeros([self.n_nodes, self.n_nodes])
        p = Pool(20)
        w_row_line_pool = p.map(Heater_FPMod.composition, [self.weights[i][j] for i in range(self.n_nodes) for j in range(self.n_nodes)])
        p.close()
        #p.join()
        split_out_chans_flat = split_out_chans.repeat((self.n_nodes,)) #np.tile(split_out_chans,(self.n_nodes,))

        """
        for row_ind in range(self.n_nodes):
            w_row_line = np.zeros(self.n_nodes)
            for col_ind in range(self.n_nodes):
                w_row_line[col_ind] = self.weights[row_ind][col_ind].composition()
            w_out_mat[row_ind, :] = split_out_chans*w_row_line
        stop = timeit.default_timer()
        """
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

        # Issues with torch.tensor dtype again --> can't make a tensor of Dets
        # --> I tried to optimize the calculation above by attempting to do element-wise
        #     multiplication instead of a for loop. 
        # responsivities = map(Det.responsivity, self.detectors)
        # det_reading2 = torch.tensor(responsivities) * sum_out_chans
        # print(det_reading2)

        return det_reading

    def calibration(self, in_pwr=100*mW):
        
        """ For simplicity, every modulator is the same """

        self.volt_sweep = torch.linspace(start=0.0, end=1.5, steps=self.cali_num) #np.linspace(start=0.0, stop=1.5, num=self.cali_num, endpoint=True)
        self.v_curves_w_max = torch.zeros(len(self.volt_sweep)) #np.zeros(len(self.volt_sweep))
        self.v_curves_w_min = torch.zeros(len(self.volt_sweep))
        self.w_curves_v_max = torch.zeros(len(self.volt_sweep))

        ## 1. set weight at max and sweep vector
        for row_ind in range(self.n_nodes):
            for col_ind in range(self.n_nodes):
                self.weights[row_ind][col_ind].set_heater(0.0)

        for volt_ind in range(len(self.volt_sweep)):
            self.vectors[0].set_heater(self.volt_sweep[volt_ind])
            self.v_curves_w_max[volt_ind] = self.simulate(in_pwr)[0] 

        self.vectors[0].set_heater(1.5)
        
        self.reset()

        ## 2. set vector at max and sweep weight
        for vec_ind in range(self.n_nodes):
            self.vectors[vec_ind].set_heater(0.0)

        for volt_ind in range(len(self.volt_sweep)):
            self.weights[0][0].set_heater(self.volt_sweep[volt_ind])
            self.w_curves_v_max[volt_ind] = self.simulate(in_pwr)[0]

        #print("old w_curves_v_max ", self.w_curves_v_max)
        self.weights[0][0].set_heater(1.5)

        self.reset()

        ## 3. set weight at min and sweep vector
        for volt_ind in range(len(self.volt_sweep)):
            self.vectors[0].set_heater(self.volt_sweep[volt_ind])
            self.v_curves_w_min[volt_ind] = self.simulate(in_pwr)[0]
        self.vectors[0].set_heater(1.5)

        ##### Find the correct tuning range ######
        self.v_max_w_max = torch.max(self.v_curves_w_max)
        self.v_min_w_max = torch.min(self.v_curves_w_max)
        self.v_max_w_min = torch.max(self.v_curves_w_min)
        self.v_min_w_min = torch.min(self.v_curves_w_min)
        self.full_range = self.v_max_w_max + self.v_min_w_min - self.v_max_w_min - self.v_min_w_max 
        #print(self.v_max_w_max, self.v_min_w_min, self.v_max_w_min, self.v_min_w_max)
        #print("Old full range \n", self.full_range)

    def set_vector_voltage(self,vector,vector_chan=0):
        
        if vector > 0:
            target_output = (self.v_max_w_max - self.v_min_w_max) * vector + self.v_min_w_max
            set_volt_vec_zero = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max - self.v_min_w_max))]
            set_volt_vec = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max - target_output))]

        else:
            target_output = (self.v_min_w_max - self.v_max_w_max) * np.abs(vector) + self.v_max_w_max
            set_volt_vec_zero = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max - self.v_max_w_max))]
            set_volt_vec = self.volt_sweep[np.argmin(np.abs(self.v_curves_w_max- target_output))]

        return set_volt_vec, set_volt_vec_zero

    def reset(self):

        for vec_ind in range(self.n_nodes):
            self.vectors[vec_ind].set_heater(1.5)

        for row_ind in range(self.n_nodes):
            for col_ind in range(self.n_nodes):
                self.weights[row_ind][col_ind].set_heater(1.5) 

    def set_weight_voltage(self,weight,weight_chan=(0,0)):

        w_max = torch.max(self.w_curves_v_max)
        w_min = torch.min(self.w_curves_v_max)
        
        if weight > 0:
            target_output = (w_max - w_min) * weight + w_min
            set_volt_weight_zero = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - self.v_max_w_min))]
            set_volt_weight = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - target_output))]

        else:
            target_output = (w_min - w_max) * np.abs(weight) + w_max
            set_volt_weight_zero = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - self.v_max_w_max))]
            set_volt_weight = self.volt_sweep[np.argmin(np.abs(self.w_curves_v_max - target_output))]

        return set_volt_weight, set_volt_weight_zero

    def multiply(self, matrix, vector):
        
        num_size = len(vector)
        set_volt_vec = torch.zeros(self.n_nodes)
        set_volt_vec_zero = torch.zeros(self.n_nodes)
        set_volt_w = torch.zeros([self.n_nodes, self.n_nodes])
        set_volt_w_zero = torch.zeros([self.n_nodes, self.n_nodes])
        for vec_ind in range(self.n_nodes):
            vec_val = vector[vec_ind]
            set_volt_vec[vec_ind], set_volt_vec_zero[vec_ind] = self.set_vector_voltage(vec_val)
        for row_ind in range(self.n_nodes):
            for col_ind in range(self.n_nodes):
                w_val = matrix[row_ind,col_ind]
                set_volt_w[row_ind, col_ind], set_volt_w_zero[row_ind, col_ind] = self.set_weight_voltage(w_val)
        import copy 
        ## o1
        for row_ind in range(self.n_nodes):
            self.vectors[row_ind].set_heater(set_volt_vec[row_ind])
            for col_ind in range(self.n_nodes):
                self.weights[row_ind][col_ind].set_heater(set_volt_w[row_ind,col_ind])
        o1 = self.simulate(self.input_pwr)
        #copy1 = copy.deepcopy(self)
        
        ## o2
        for row_ind in range(self.n_nodes):
            self.vectors[row_ind].set_heater(set_volt_vec_zero[row_ind])
            for col_ind in range(self.n_nodes):
                self.weights[row_ind][col_ind].set_heater(set_volt_w[row_ind,col_ind])
        o2 = self.simulate(self.input_pwr)
        #copy2 = copy.deepcopy(self)

        ## o3
        for row_ind in range(self.n_nodes):
            self.vectors[row_ind].set_heater(set_volt_vec[row_ind])
            for col_ind in range(self.n_nodes):
                self.weights[row_ind][col_ind].set_heater(set_volt_w_zero[row_ind,col_ind])

        #copy3 = copy.deepcopy(self)
        o3 = self.simulate(self.input_pwr)

        ## o4
        for row_ind in range(self.n_nodes):
            self.vectors[row_ind].set_heater(set_volt_vec_zero[row_ind])
            for col_ind in range(self.n_nodes):
                self.weights[row_ind][col_ind].set_heater(set_volt_w_zero[row_ind,col_ind])

        #copy4 = copy.deepcopy(self)
        o4 = self.simulate(self.input_pwr)
        #print("o1-o4:", o1,o2,o3,o4, self.full_range)
        output = (o1 - o2 - o3 + o4)/self.full_range

        return output

    def matmul(self, mat1, mat2):
        """
        This is a wrapper around the hardware multiply function
        to run general matrix-matrix multiplication.

        Arguments:
            mat1: N x M numpy matrix
            mat2: M x P numpy matrix
        """
        print(mat1.shape, mat2.shape)
        n1, m1 = mat1.shape
        m2, p2 = mat2.shape

        assert (m1 == m2), ValueError('Incompatible dimension of matrices')

        block_N = int(np.ceil(np.max([n1, m1])/self.n_nodes))
        N_mat = block_N*self.n_nodes

        output_expand = torch.zeros([int(N_mat), int(p2)])

        mat1_expand = torch.zeros([int(N_mat), int(N_mat)])
        mat1_expand[0:n1, 0:m1] = torch.tensor(mat1) 

        mat2_expand = torch.zeros([int(N_mat), int(p2)])
        mat2_expand[0:m2,:] = torch.tensor(mat2)

        for col2_ind in range(int(p2)):
            for block_row_ind in range(block_N):
                for block_col_ind in range(block_N):
                    output_expand[block_row_ind*self.n_nodes:(block_row_ind+1)*self.n_nodes, 
                                  col2_ind] += self.multiply(mat1_expand[block_row_ind*self.n_nodes:(block_row_ind+1)*self.n_nodes, 
                                  block_col_ind*self.n_nodes:(block_col_ind+1)*self.n_nodes], mat2_expand[block_col_ind*self.n_nodes:(block_col_ind+1)*self.n_nodes,col2_ind])

        output = output_expand[0:n1, 0:p2]
        return output 

 
