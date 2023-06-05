#  Multioutput prediction of global cropland yields

This is a programme that predict global lnRR of cropland based on the observations.

- Agminated prediction task for globle crop prediction
- Reliable machine learning model for agricultural and research
- Outputs lives at MultiOutput_Regression/output

## Tutorial
### Installation

- python == 3.9.2
- sklearn == 1.2.2
- torch == 2.0.1
- pandas == 2.0.1
- numpy == 1.23.5

      pip install scikit-learn==1.2.2


### Read data
Read raw data from csv file.

    df_raw = pd.read_csv(os.path.join(root_path, datd_path))
    cols = list(df_raw.colunms);
    cols = [i for i in cols if i not in target and i not in redudent]
    col_y = target
    
    df_x = df_raw[cols]
    df_y = df_raw[col_y]


### Build multiclass regression model
    estimators = args_info['n_estimator']
    random_state = args_info['random_state']
    depth = args_info['max_depth']

    rf = RandomForestRegressor(n_estimators=estimators, random_state=random_state, max_depth=depth)
    regression = MultiOutputRegressor(rf)

    
### Define train
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2)
    regression.fit(train_x, train_y)

    r2_train = regression.score(train_x, train_y)
    r2_test = regression.score(test_x, test_y)

    print("|| Train R2: {0:.7F}    Test R2: {1:.7F}").format(r2_train, r2_test)

### Do predict
    Ypredicted = regression.predict(Predict_x)
    preds = np.array(Ypredicted)
    pred_lnRR = preds[:, 0].reshape(360, -1)
    pred_substence = preds[:, 1].reshape(360, -1)

    pred_lnRR = pred_lnRR * landcover
    pred_substence = pred_substence * landcover

## Program running


    python main.py --model='Adaboost' --root_path='./raw_Data' --data_path='SOCALL.csv' --random_state=100 --traget=['lnRR', 'SOC'] --do_predict=True


## Model performance

![Featurs' particial dependency correlation lines](https://github.com/jancely/Multioutput-prediction/blob/main/Features.tiff))

![Global predicted cropland lnRR from our optimized model](https://github.com/jancely/Multioutput-prediction/blob/main/Predict_lnRR.tiff)
