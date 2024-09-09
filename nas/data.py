import torch, os
from torch.utils.data import  Dataset
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat
import pandas as pd
import nni
import logging

_logger = logging.getLogger(__name__)

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

@nni.trace
class MISO_Data_v1(Dataset):
    def __init__(self, lag_range, target):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_data(lag_range, target)
        self.x=torch.tensor(self.preprocessed_df[self.features].values,dtype=torch.float32).reshape(-1, 1, len(self.features))
        self.y=torch.tensor(self.preprocessed_df[self.target].values,dtype=torch.float32).reshape(-1, 1).to(device)
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x).to(device)
        
    def load_data(self, lag_range, target):
        #parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))
        data_dir = "./data/" #os.path.join(parent_dir,"data")
        # Load sensor raw data into a pandas data frame
        rawdf = pd.read_csv(os.path.join(data_dir,'MISO_220621-0627_1mnDataset_K96.csv'))
        
        self.features = ['pressure(hPa)', 'rh(%)', 'temp_sensor(C)', 'k96_lpl_raw', 'k96_spl_raw', 'k96_mpl_raw', "k96_h2o(ppm) ", "k96_ch4(ppm) "]
        if target == "ref_h2o(ppm)":
            self.features += ["k96_h2o(ppm) _" + str(i) for i in range(1, lag_range)]
            extra_prefix = "k96_h2o(ppm) " 
        elif target == "ref_ch4(ppm)":
            self.features += ["k96_ch4(ppm) _" + str(i) for i in range(1, lag_range)]
            extra_prefix = "k96_ch4(ppm) " 
            
        values = DataFrame(rawdf[extra_prefix].values)
        lag_df = concat([values.shift(j) for j in range(lag_range)], axis=1)
        lag_df.columns = [extra_prefix + "_" +str(j) for j in range(lag_range)]
        self.preprocessed_df =  pd.concat([rawdf, lag_df], axis=1)[lag_range:]
        self.target = target
        
        
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    


class MISO_Data_v2(Dataset):
    def __init__(self, lag_range, target):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_data(lag_range, target)
        self.x=torch.tensor(self.preprocessed_df[self.features].values,dtype=torch.float32).reshape(-1, 1, len(self.features))
        self.y=torch.tensor(self.preprocessed_df[self.target].values,dtype=torch.float32).reshape(-1, 1).to(device)
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x).to(device)
        
    def load_data(self, lag_range, target):
        #parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))
        data_dir = "./data/" #os.path.join(parent_dir,"data")
        # Load sensor raw data into a pandas data frame
        rawdf = pd.read_csv(os.path.join(data_dir,'DatasetForMasoud.csv'))
        # "lpl_conc","mpl_conc","lpl_ir","mpl_ir",bme_pres,bme_rh,bme_temp,ref_co2,ref_h2o,bme_h2o"
        # self.features = ['pressure(hPa)', 'rh(%)', 'temp_sensor(C)', 'k96_lpl_raw', 'k96_spl_raw', 'k96_mpl_raw', "k96_h2o(ppm) ", "k96_ch4(ppm) "]
        # if target == "ref_h2o(ppm)":
        #     self.features += ["k96_h2o(ppm) _" + str(i) for i in range(1, lag_range)]
        #     extra_prefix = "k96_h2o(ppm) " 
        # elif target == "ref_ch4(ppm)":
        #     self.features += ["k96_ch4(ppm) _" + str(i) for i in range(1, lag_range)]
        #     extra_prefix = "k96_ch4(ppm) " 
            
        # values = DataFrame(rawdf[extra_prefix].values)
        # lag_df = concat([values.shift(j) for j in range(lag_range)], axis=1)
        # lag_df.columns = [extra_prefix + "_" +str(j) for j in range(lag_range)]
        # self.preprocessed_df =  pd.concat([rawdf, lag_df], axis=1)[lag_range:]
        # self.target = target
        
        
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]