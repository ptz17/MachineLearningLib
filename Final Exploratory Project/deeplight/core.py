import numpy as np
from deeplight.system_torch import photocore
import os.path
import pickle
import sys

class loadcore:
    def __init__(self, n_nodes, cali_num):
        self.cali_num = cali_num
        self.n_nodes = n_nodes
        self.filename = 'photocore' + str(n_nodes) + "_" + str(cali_num)
    def gen(self):
        if os.path.exists(self.filename):
            file = open(self.filename, 'rb')
            phcore = pickle.load(file) 
            print("deeplight/core.py: load %s core" % self.filename)
        else:
            phcore = photocore(n_nodes=self.n_nodes)
            print("deeplight/core.py: calibration")
            phcore.cali_num = self.cali_num
            phcore.calibration()
            print("deeplight/core.py: calibration ends")
            file = open(self.filename, 'wb')
            pickle.dump(phcore, file)
            file.close()
        return phcore


