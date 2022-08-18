# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from darts.models import TransformerModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import coefficient_of_variation, mae, mape, marre, mase, mse
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import coefficient_of_variation, mae, mape, marre, mase, mse

LEN = 1200
AHEAD = 60
HIS = 120
LIST_TAR = [2]
LIST_COVS = [1, 3, 4, 9, 10, 11, 12, 13, 14, 16]

switch = 0
if __name__ == '__main__':
    for path, dir_list, file_list in os.walk(r"D:/data/wedling gun"):
        for file_name in file_list:
            if file_name == 'E016 2021 9.csv':
                switch = 0
            if switch == 0:
                # if file_name == 'E001 2021 10.csv':
                #     continue
                print('Begin '+file_name)
                file = os.path.join(path, file_name)
                df = pd.read_csv(file, index_col='time', low_memory=False)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.resample('S').asfreq()
                df = df.apply(pd.to_numeric, errors='ignore')
                df[df['in_Sheet_thickness'] > 5] = np.nan
                df[df['in_Sheet_thickness'] <= 0] = np.nan

                list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
                df.iloc[:, list] = StandardScaler().fit_transform(df.iloc[:, list])  # 数值类型特征标准化
                L = df.iloc[:, list]
                L_imputed = df.iloc[:, list]
                L_imputed.fillna(method='ffill', inplace=True)

                data_tar = L_imputed.iloc[-LEN:, LIST_TAR]
                data_covs = L_imputed.iloc[-LEN:, LIST_COVS]

                series_tar = TimeSeries.from_dataframe(data_tar, freq='S').astype(np.float32)
                train, test = series_tar[:-AHEAD], series_tar[-AHEAD:]
                series_covs = TimeSeries.from_dataframe(data_covs, freq='S').astype(np.float32)
                past_covs, future_cosv = series_covs[:-AHEAD], series_covs[-AHEAD:]

                model_Regression = RegressionModel(
                    lags=HIS,
                    lags_past_covariates=HIS,
                    output_chunk_length=AHEAD,
                    model=BayesianRidge()
                )
                model_Regression.fit(train, past_covariates=past_covs)
                pred_Regression = model_Regression.predict(series=train, past_covariates=past_covs, n=AHEAD)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 16))
                train[-HIS:].plot(label="train")
                test.plot(label="actual")

                output = {'file_name':[file_name],
                          'mae': [mae(test, pred_Regression)],
                          'mape': [mape(test, pred_Regression)],
                          # 'marre': [marre(test, pred_Regression)],
                          'mse': [mse(test, pred_Regression)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016_BayesianRidge.csv', mode='a', header = None)
                print('End ' + file_name)

