### Program used to begin exploring training for the device material layering.
import torch
from torch import linalg as LA
from deeplight.system_torch import photocore as new_photocore
import pickle
import sys
import random
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True) # Very useful for debugging and stack-tracing gradients

def main(args):
    # if (len(sys.argv)!=4):
    #     print("Wrong args!!\nusage:v python thisfile.py n_nodes mm_size cali_num")
    #     exit(0)

    ### Initialize variables
    n_nodes = 2 #int(sys.argv[1])
    mm_size = 2 #int(sys.argv[2]) 
    cali_num = 2 #int(sys.argv[3])
    print(n_nodes,mm_size)
    filename = 'photocore' + str(n_nodes) + "_" + str(cali_num)
    um = 1e-6
    nm = 1e-9
    mW = 1e-3

    # 8, 1, 9, 2, 4, 5, 0, 7, 6, 3
    dev_structure = [('air',0.0*nm), ('al',20*nm), ('sio',30*nm), 
                                ('ito',30*nm), ('sio',15*nm), ('vo2',3*nm),
                                ('sio',15*nm), ('si',40*nm), ('ito',30*nm),
                                ('sio',70*nm), ('al',20*nm), ('air', 0.0*nm)]
    idx = list(range(10))
    random.shuffle(idx)
    dev_str = [dev_structure[i] for i in idx]
    f = open('device_str_fbnorm.txt','a+')
    print(idx, file=f)

    ### Create a core using the default dev_structure for now (this may be randomized for future exploration)
    for i in range(1):
        dev_structure = dev_str 
        # This is a MM test code. For generating Core, check out core_gen.py
        phcore = new_photocore(dev_structure = dev_structure, n_nodes=n_nodes, materials_oneHot=torch.empty((0)), thicknesses=torch.empty((0))) #add oneHot arg --> if oneHot exists, dont use dev_structure.  
        #print("calibration")
        phcore.cali_num = cali_num
        phcore.calibration()
        #print("calibration ends")
 
    lr = 1e-25
    losses = []
    materials_oneHot_updated = torch.empty((0))
    thicknesses_updated = torch.empty((0))

    ### Cycle through a variety of matrix multiplication samples
    epochs = 11
    batchSize = 10
    for i in range(epochs): # Number of epochs
        print(i)
        torch.manual_seed(i)
        mat1 = torch.rand(batchSize, mm_size, mm_size)
        mat2 = torch.rand(mm_size, mm_size)
        mat1.requires_grad = True
        mat2.requires_grad = True
        t = torch.matmul(mat1, mat2) 

        # Construct photocore with materials oneHot input
        print("Construct photocore ...")
        phcore = new_photocore(dev_structure = dev_structure, n_nodes=n_nodes, materials_oneHot=materials_oneHot_updated, thicknesses=thicknesses_updated) #add oneHot arg --> if oneHot exists, dont use dev_structure. 
        #print("calibration")
        phcore.cali_num = cali_num
        phcore.calibration()
        #print("calibration ends")

        ph = phcore.matmul(mat1, mat2)
        diff = t - ph
        # Average over batch size to get gradient for updating mini-batch sgd. 
        avg_diff = torch.mean(diff, dim=0)
        # print(avg_diff)
        n = LA.norm(avg_diff) 
        print("Loss: ", n)
        # print("Grad before: Materials\n", phcore.oneHot.grad, "\nThickness\n", phcore.dev_structure_thickness.grad)
        
        n.backward(retain_graph=True) 
        
        materials_oneHot_updated = phcore.oneHot - phcore.oneHot.grad*lr
        print("Updated materials: ", materials_oneHot_updated)
        materials_oneHot_updated = torch.tensor(materials_oneHot_updated, requires_grad=True)

        print("Gradients for Materials: \n", phcore.oneHot.grad)
        
        losses.append(n)

    print(f'Epochs = {epochs} and Batch Size = {batchSize}')
    plt.plot(losses) 
    plt.title(f'Loss Curve for Learned Materials')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

  
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))