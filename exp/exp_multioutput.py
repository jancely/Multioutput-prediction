import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from exp.exp_basic import Exp_Basic
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from data.loadData_distribution import Dataset_Load, Dataset_Pred
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import time

import warnings
warnings.filterwarnings('ignore')


def _get_data(args, flag):
    if flag == 'pred':
        Data = Dataset_Pred(args.cru_file, args.gpcc_file, args.clay_file,
                            args.bd_file, args.soc_file, args.ph_file,
                            args.crop_file, args.land_file, args.silt_file, args.sand_file)
    else:
        Data = Dataset_Load(
            root_path=args.root_path,
            data_path=args.data_path,
            target=args.target,
            redudent=args.redudent)

    return Data

class Exp_Adaboost(Exp_Basic):
    def __init__(self, args):
        super(Exp_Adaboost, self).__init__(args)

        self.seed = self.args.seed
        self.material = args.material
        self.root_path = self.args.root_path
        self.data_path = self.args.data_path
        self.target = self.args.target
        self.redudent = self.args.redudent
        self._build_model(self.seed, self.material)

    def _build_model(self, random_state, material):
        mate_dict = {
            'SOC': {'n_estimators': 100, 'random_state': random_state, 'max_depth': 13},  
            'NL': {'n_estimators': 21, 'random_state': random_state, 'max_depth': 7},    
            'CO2': {'n_estimators': 50, 'random_state': random_state, 'max_depth': 13},
            'N2O': {'n_estimators': 37, 'random_state': random_state, 'max_depth': 7},
        }
        args_info = mate_dict[material]
        estimators = args_info['n_estimators']
        random_state = args_info['random_state']
        depth = args_info['max_depth']

        rf = RandomForestRegressor(n_estimators=estimators, random_state=random_state, max_depth=depth)

        self.regression = MultiOutputRegressor(rf)

        return self.regression


    def train(self, setting):
        args = self.args
        
        test_dict = {
            'SOC': 0.2,  #已确定 90 13
            'NL':  0.25,    #已确定  21  7  25 8
            'CO2': 0.2,  #已确定
            'N2O': 0.2
        }
        
        #get data
        Data = _get_data(args=args, flag='train')[0]
        dx, dy = Data[0], Data[1]
        # print(columns)
        
        #split data into train and test
        train_x, test_x, train_y, test_y = train_test_split(dx, dy, test_size=test_dict[self.material], random_state=args.seed)

        #build model
        # model = self._build_model(self.seed, self.material)
        self.regression.fit(train_x, train_y)

        Ecalculate train and test r2 score
        R2train = self.regression.score(train_x, train_y)
        R2test = self.regression.score(test_x, test_y)

        print("|| Train R2: {0:.7f} Test R2: {1:.7f}".format(R2train, R2test))

        return R2train, R2test, weight, self.regression


class Exp_Predict(Exp_Basic):
    def __init__(self, args):
        super(Exp_Predict, self).__init__(args)

        self.args = args
        # print('self.args', self.args)

    def predict(self, model):
        Predict = _get_data(self.args, flag='pred')[0]
        Predict_x, landcover = Predict[0], Predict[1]

        Ypredicted = model.predict(Predict_x)

        preds = np.array(Ypredicted)
        # print(preds.shape)      #(259200, 2)
        pred_lnRR = preds[:, 0].reshape(360, -1)
        pred_substence = preds[:, 1].reshape(360, -1)

        pred_lnRR = pred_lnRR * landcover
        pred_substence = pred_substence * landcover

        # np.save(folder_path + str(args.seed) + '_real_prediction', preds)

        return pred_lnRR, pred_substence


