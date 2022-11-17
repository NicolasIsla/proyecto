import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt

def recorte(data,dim=21):
    muestras,ancho,largo,canales=data.shape
    print(data.shape)
    #len(data[0])
    my_x = np.ndarray(shape=(muestras, 21, 21,3))
    # Recorte

    # nuevo dataset 
    new_data=np.zeros(shape=(muestras, 21, 21,3))
    for k in range(muestras):
        new_data[k] = data[k][ 20:41 , 20:41 ]
    # Redimension y a tensor
    # my_x = np.transpose(new_data , 0, 3, 1, 2) 
    tensor_x = torch.Tensor(new_data)
    print(tensor_x.shape)
    tensor_x = tensor_x.permute(0, 3, 1, 2)
    my_datatrain = TensorDataset(tensor_x,tensor_x) 
    return my_datatrain  