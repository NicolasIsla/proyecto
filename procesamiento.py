import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

def recorte(data,dim=21):
    muestras,ancho,largo,canales=data.shape
    # Recorte
    extremo_izq=(ancho-dim)//2
    extremo_der=extremo_izq+dim
    # nuevo dataset 
    new_data=np.zeros([muestras,dim,dim,canales])
    for i in range(muestras):
        new_data[i]=data[i][extremo_izq:extremo_der,
                                                extremo_izq:extremo_der]
    # Redimension y a tensor
    my_x = new_data.reshape(muestras, canales, dim, dim) 
    tensor_x = torch.Tensor(my_x) 
    my_datatrain = TensorDataset(tensor_x,tensor_x) 
    return my_datatrain