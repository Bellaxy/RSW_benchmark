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
from darts.models import RandomForest
from darts.models import TCNModel

LEN = 1200
AHEAD = 60
HIS = 120
LIST_TAR = [2]
LIST_COVS = [1, 3, 4, 9, 10, 11, 12, 13, 14, 16]

switch = 0
if __name__ == '__main__':
    for path, dir_list, file_list in os.walk(r"D:/data/wedling gun"):
        for file_name in file_list:
            if file_name == 'E029 2021 99_2506.csv':
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

                model_TCNModel = TCNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    kernel_size=3,
                    num_filters=3,
                    dilation_base=2,
                    weight_norm=False,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_TCNModel.fit(train, past_covariates=past_covs)
                predl_TCNModel = model_TCNModel.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_TCNModel.plot(label="forecast")


                output = {'file_name':[file_name],
                          'mae': [mae(test, predl_TCNModel)],
                          'mape': [mape(test, predl_TCNModel)],
                          # 'marre': [marre(test, predl_TCNModel)],
                          'mse': [mse(test, predl_TCNModel)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016_TCN.csv', mode='a', header = None)
                print('End ' + file_name)

                model_TransformerModel = TransformerModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    torch_device_str='cuda'

                )
                model_TransformerModel.fit(train, past_covariates=past_covs)
                predl_TransformerModel = model_TransformerModel.predict(series=train, n=AHEAD,
                                                                        past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_TransformerModel.plot(label="forecast")

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_TransformerModel)],
                          'mape': [mape(test, predl_TransformerModel)],
                          # 'marre': [marre(test, predl_TransformerModel)],
                          'mse': [mse(test, predl_TransformerModel)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016_Transformer.csv', mode='a', header=None)
                print('End ' + file_name)

