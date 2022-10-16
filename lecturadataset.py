## descarga, descomprimir archivos y lectura pkl
# from google.colab import drive
import zipfile_deflate64 as zipfile
import pickle

def lectura():
    
    #zip_ref = zipfile.ZipFile("./td_ztf_stamp_17_06_20.zip", 'r')
    #zip_ref.extractall("./")
    #zip_ref.close()
    with open('./td_ztf_stamp_17_06_20.pkl', 'rb') as f:
        data = pickle.load(f)

    return data
    # print(data.keys())
    # print(data["Train"].keys())

