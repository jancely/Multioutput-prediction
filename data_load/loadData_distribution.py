import os
import time
import pandas as pd

from torch.utils.data import Dataset

# from pyhdf.SD import SD

from utils.tools import StandardScaler
from mpl_toolkits import basemap

from pylab import *
from data.global_base import read_global

import warnings
warnings.filterwarnings('ignore')

class Dataset_Load(Dataset):
    def __init__(self, root_path, data_path='N2OALL.csv',
                 target='lnRR', redudent='Family1'):
        # size [seq_len, label_len, pred_len]
        # info
        self.target = target
        self.redudent = redudent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # print(os.getcwd())    #G:\Project_Code\Adaboost_Regression\models
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), encoding='ISO-8859-1')
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns);
        # cols.remove(self.target + self.redudent);
        # cols = list(set(cols) - set(self.target) - set(self.redudent))
        cols = [i for i in cols if i not in self.target and i not in self.redudent]
        # print('cols are', cols)
        col_y = self.target
        df_x = df_raw[cols]
        df_y = df_raw[col_y]

        # self.data_y = df_y.values
        # self.data_x = df_x.values
        self.data_y = df_y
        self.data_x = df_x

        # return data_x, data_y

    def __getitem__(self, index):
        seq_x = self.data_x
        seq_y = self.data_y

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x)


class Dataset_Pred(Dataset):
    def __init__(self, cru_datafilename, gpcc_filename, clay_filename,
                 bd_filename, soc_filename, ph_filename, Cropcomb_filename,
                 landfilename, silt_filename, sand_filename):

        self.cru_file = cru_datafilename
        self.gpcc_file = gpcc_filename
        self.clay_file = clay_filename
        self.silt_file = silt_filename
        self.sand_file = sand_filename
        self.bd_file = bd_filename
        self.soc_file = soc_filename
        self.ph_file = ph_filename
        self.Cropcomb_file = Cropcomb_filename
        self.landfile = landfilename

        self.__read_data__()

    def __read_data__(self):
        # cru_datafilename = '.\global_data\MAT\cru_ts4.06.2021.2021.tmp.dat.nc'
        self.soc_avg, self.landcover_all = read_global(self.cru_file, self.gpcc_file, self.clay_file, self.bd_file,
                              self.soc_file, self.ph_file, self.Cropcomb_file, self.landfile, self.silt_file, self.sand_file)
        return self.soc_avg, self.landcover_all

    def __getitem__(self, index):
            return self.soc_avg, self.landcover_all

    def __len__(self):
        return len(self.soc_avg)
