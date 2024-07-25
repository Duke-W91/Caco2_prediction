import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

for predictor in ['XGBoost', 'SVM', 'GB', 'RF']:
    results = pd.DataFrame(index=np.arange(1, 11))
    for i in range(1, 11):
## random split data
        train_way = str(i) + "_train_set_descriptor_random.csv"
        dev_way = str(i) + "_dev_set_descriptor_random.csv"
        test_way = str(i) + "_test_set_descriptor_random.csv"
        
        train = pd.read_csv(train_way)
        dev = pd.read_csv(dev_way)
        test = pd.read_csv(test_way)
## fp
#         X_train = train.iloc[:, 1:2049]
#         y_train = train.iloc[:, 2049]
#         X_dev = dev.iloc[:, 1:2049]
#         y_dev = dev.iloc[:, 2049]
#         X_test = test.iloc[:, 1:2049]
#         y_test = test.iloc[:, 2049]
##rdkit2d
        X_train = train.iloc[:, 1:201]
        y_train = train.iloc[:, 201]
        X_dev = dev.iloc[:, 1:201]
        y_dev = dev.iloc[:, 201]
        X_test = test.iloc[:, 1:201]
        y_test = test.iloc[:, 201]


        if predictor == 'XGBoost':
            regr = XGBRegressor(learning_rate=0.05, n_estimators=800, max_depth=7,gamma=0.1, colsample_bytree=0.8, subsample=0.8,
                                random_state=0)
            regr.fit(X_train, y_train)
        elif predictor == 'SVM':
            regr = SVR(C=4, epsilon=0.03, gamma=0.01)
            regr.fit(X_train, y_train)

        elif predictor == 'GB':
            regr = GradientBoostingRegressor(learning_rate=0.2, n_estimators=1100,
                                             random_state=0)
            regr.fit(X_train, y_train)

        elif predictor == 'RF':
            regr = RandomForestRegressor(n_estimators=500, random_state=0)
            regr.fit(X_train, y_train)


        y_train_predict = regr.predict(X_train)
        y_dev_predict = regr.predict(X_dev)
        y_test_predict = regr.predict(X_test)

        r2_train = r2_score(y_train, y_train_predict)
        mse_train = mean_squared_error(y_train, y_train_predict, squared=False)
        mae_train = mean_absolute_error(y_train, y_train_predict)
        r2_dev = r2_score(y_dev, y_dev_predict)
        mse_dev = mean_squared_error(y_dev, y_dev_predict, squared=False)
        mae_dev = mean_absolute_error(y_dev, y_dev_predict)
        r2_test = r2_score(y_test, y_test_predict)
        mse_test = mean_squared_error(y_test, y_test_predict, squared=False)
        mae_test = mean_absolute_error(y_test, y_test_predict)

        results.loc[i, "train_score"] = r2_train
        results.loc[i, "train_rmse"] = mse_train
        results.loc[i, "train_mae"] = mae_train
        results.loc[i, "test_score"] = r2_test
        results.loc[i, "test_rmse"] = mse_test
        results.loc[i, "test_mae"] = mae_test
        results.loc[i, "val_score"] = r2_dev
        results.loc[i, "val_rmse"] = mse_dev
        results.loc[i, "val_mae"] = mae_dev

    results.to_csv(r'results_{}_rdkit2d_performance.csv'.format(predictor))
