import argparse
import os
import torch
import numpy as np
import pandas as pd
import _pickle as cPickle

from exp.exp_multioutput import Exp_Adaboost, Exp_Predict
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='[Adaboost] Adaboost regression for fertilizer prediction')

parser.add_argument('--model', type=str, required=True, default='Adaboost', help='model name')
parser.add_argument('--root_path', type=str, required=True, default='./raw_data/',  help='root path of the data file')
parser.add_argument('--data_path', type=str, required=True, default='N2OALL.csv', help='data file')
parser.add_argument('--random_state', type=int, required=True, default=100, help='random state of adaboost regressor')
parser.add_argument('--target', type=str, required=True, default=['lnRR', 'N2O'], help='target prediction columns')
parser.add_argument('--do_predict', type=str, required=True, default=True, help='predict global fertilizer substance')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--output', type=str, default='./prediction/', help='prediction save path')

parser.add_argument('--redudent', type=str, default=['Crop1', 'Crop2', 'Family', 'LE', 'Cropcombination'],
                    help='input sequence length of Informer encoder')
parser.add_argument('--cru_file', type=str, default='./global_data/MAT/cru_ts4.06.2021.2021.tmp.dat.nc', help='cru data')
parser.add_argument('--gpcc_file', type=str, default='.\global_data\MAP\\normals_1991_2020_v2022_05.nc', help='gpcc data')
parser.add_argument('--clay_file', type=str, default='.\global_data\T_CLAY.nc4', help='clay data')
parser.add_argument('--silt_file', type=str, default='.\global_data\T_SILT.nc4', help='clay data')
parser.add_argument('--sand_file', type=str, default='.\global_data\T_SAND.nc4', help='clay data')
parser.add_argument('--bd_file', type=str, default='.\global_data\T_BULK_DEN.nc4', help='bd data')
parser.add_argument('--soc_file', type=str, default='.\global_data\T_OC.nc4', help='soc data')
parser.add_argument('--ph_file', type=str, default='.\global_data\T_PH_H2O.nc4', help='ph data')
parser.add_argument('--crop_file', type=str, default='G:\Project_Code\Adaboost_Regression\Cropcombination.xlsx', help='crop data')
parser.add_argument('--land_file', type=str, default='.\global_data\MCD12C1.A2021001.061.2022217040006.hdf', help='land data')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

# parser.add_argument('-')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')

target = args.target[1]
setting = '{}_{}'.format(args.model, target)

Exp = Exp_Adaboost

R2trainall = []
R2testall = []
# Prediction_lnRR = []
# Prediction_N2O = []
Weights = []

judge = 0
checkpoint_path = os.path.join(args.checkpoints, setting)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

for i in range(args.random_state):
    args.seed = i * 15 + 1
    args.material = args.target[1]
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    R2train, R2test, weigth, model = exp.train(setting)
    if judge >= R2test:
        pass
    else:
        best_model = model
        best_model_path = checkpoint_path + '/' + 'checkpoint.pth'
        with open(best_model_path, 'wb') as f:
            cPickle.dump(model, f)

        judge = R2test
    # R2train, R2test= exp.train(setting)

    R2trainall.append(R2train)
    R2testall.append(R2test)
    Weights.append(weigth)

#
mean_trainR2 = np.round(np.mean(R2trainall), 4)
mean_train_std = np.round(np.std(R2trainall), 3)
mean_testR2 = np.round(np.mean(R2testall), 4)
mean_test_std = np.round(np.std(R2testall), 3)

print("Global| Train R2: {0:.7f} +/- {1: .3f} Test R2: {2:.7f} +/- {3: .3f}".format(mean_trainR2, mean_train_std,
                                                                                    mean_testR2, mean_test_std))


if args.do_predict:
    print('\n>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    Exp_Pre = Exp_Predict(args)

    lnRR, N2O = Exp_Pre.predict(best_model)
    # Prediction_lnRR.append(lnRR)
    # Prediction_N2O.append(N2O)


# result save
folder_path = './results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Prediction_lnRR = np.mean(Prediction_lnRR, axis=0)
# Prediction_N2O = np.mean(Prediction_N2O, axis=0)

lnRR_N2O = pd.DataFrame(lnRR)
pred_N2O = pd.DataFrame(N2O)
weights = pd.DataFrame(Weights)
# pred.columns = ['lnRR', 'N2O']
writer1 = pd.ExcelWriter(folder_path + str(args.target[1]) + '_lnRR.xlsx')
writer2 = pd.ExcelWriter(folder_path + str(args.target[1]) + '.xlsx')
writer3 = pd.ExcelWriter(folder_path + str(args.target[1]) + '_weight.xlsx')
lnRR_N2O.to_excel(writer1, float_format='%.4f')
pred_N2O.to_excel(writer2, float_format='%.4f')
weights.to_excel(writer3, float_format='%.4f')
writer1._save()
writer2._save()
writer3._save()



