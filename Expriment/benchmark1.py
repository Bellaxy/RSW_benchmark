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
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.metrics import coefficient_of_variation, mae, mape, marre, mase, mse
from darts.models import RandomForest
from darts.models import LinearRegressionModel
from darts.models import LightGBMModel
from darts.models import BlockRNNModel
from darts.models import TCNModel
from darts.models import TransformerModel

LEN = 1200
AHEAD = 60
HIS = 120


switch = 0

if __name__ == '__main__':
    print('------------------E016----------------------')
    LIST_TAR = [2]
    LIST_COVS = [1, 3, 4, 9, 10, 11, 12, 13, 14, 16]

    for path, dir_list, file_list in os.walk(r"D:/data/E016"):
        for file_name in file_list:
            if file_name == '.csv':
                switch = 0
            if switch == 0:
                # if file_name == 'E001 2021 10.csv':
                #     continue

                file = os.path.join(path, file_name)
                print(file)
                df = pd.read_csv(file, index_col='time', low_memory=False)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.resample('S').asfreq()
                df = df.apply(pd.to_numeric, errors='ignore')
                df[df['in_Sheet_thickness'] > 5] = np.nan
                df[df['in_Sheet_thickness'] <= 0] = np.nan

                list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
                df.iloc[:, list] = StandardScaler().fit_transform(df.iloc[:, list])  # 数值类型特征标准化
                # L = df.iloc[:, list]
                L_imputed = df.iloc[:, list]
                L_imputed.fillna(method='ffill', inplace=True)

                data_tar = L_imputed.iloc[-LEN:, LIST_TAR]
                data_covs = L_imputed.iloc[-LEN:, LIST_COVS]

                series_tar = TimeSeries.from_dataframe(data_tar, freq='S').astype(np.float32)
                train, test = series_tar[:-AHEAD], series_tar[-AHEAD:]
                series_covs = TimeSeries.from_dataframe(data_covs, freq='S').astype(np.float32)
                past_covs, future_cosv = series_covs[:-AHEAD], series_covs[-AHEAD:]

                # model_Regression = RegressionModel(
                #     lags=HIS,
                #     lags_past_covariates=HIS,
                #     output_chunk_length=AHEAD,
                #     model=BayesianRidge()
                # )
                # model_Regression.fit(train, past_covariates=past_covs)
                # pred_Regression = model_Regression.predict(series=train, past_covariates=past_covs, n=AHEAD)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                # output = {'file_name': [file_name],
                #           'mae': [mae(test, pred_Regression)],
                #           'mape': [mape(test, pred_Regression)],
                #           # 'marre': [marre(test, pred_Regression)],
                #           'mse': [mse(test, pred_Regression)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_BayesianRidge.csv', mode='a', header=None)
                # print('End ' + 'BayesianRidge')
                #
                # model_RandomForest = RandomForest(
                #     lags=HIS,
                #     lags_past_covariates=HIS,
                #     output_chunk_length=AHEAD,
                #     n_estimators=100,
                #     max_depth=None
                #
                # )
                # model_RandomForest.fit(train, past_covariates=past_covs)
                # predl_RandomForest = model_RandomForest.predict(series=train, n=AHEAD, past_covariates=series_covs)
                #
                # output = {'file_name': [file_name],
                #           'mae': [mae(test, predl_RandomForest)],
                #           'mape': [mape(test, predl_RandomForest)],
                #           # 'marre': [marre(test, predl_RandomForest)],
                #           'mse': [mse(test, predl_RandomForest)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_RandomForest.csv', mode='a', header=None)
                # print('End ' + 'model_RandomForest')
                #
                #
                # model_LinearRegressionModel = LinearRegressionModel(
                #     lags=HIS,
                #     lags_past_covariates=HIS,
                #     output_chunk_length=AHEAD,
                #     random_state=43
                #
                # )
                # model_LinearRegressionModel.fit(train, past_covariates=series_covs)
                # predl_LinearRegressionModel = model_LinearRegressionModel.predict(series=train, n=AHEAD,
                #                                                                   past_covariates=past_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                # output = {'file_name':[file_name],
                #           'mae': [mae(test, predl_LinearRegressionModel)],
                #           'mape': [mape(test, predl_LinearRegressionModel)],
                #           # 'marre': [marre(test, predl_LinearRegressionModel)],
                #           'mse': [mse(test, predl_LinearRegressionModel)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_LinearRegression.csv', mode='a', header = None)
                # print('End ' + 'LinearRegressionModel')
                #
                # model_LightGBMModel = LightGBMModel(
                #     lags=HIS,
                #     lags_past_covariates=HIS,
                #     output_chunk_length=AHEAD,
                #     random_state=43,
                #
                # )
                # model_LightGBMModel.fit(train, past_covariates=past_covs)
                # predl_LightGBMModel = model_LightGBMModel.predict(series=train, n=AHEAD, past_covariates=series_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                # output = {'file_name': [file_name],
                #           'mae': [mae(test, predl_LightGBMModel)],
                #           'mape': [mape(test, predl_LightGBMModel)],
                #           # 'marre': [marre(test, predl_LightGBMModel)],
                #           'mse': [mse(test, predl_LightGBMModel)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_LightGBM.csv', mode='a', header=None)
                # print('End ' + 'LightGBM')
                print(' begin NBEATS')
                model_NBEATS = NBEATSModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    random_state=42,
                    torch_device_str='cuda'
                    #     pl_trainer_kwargs={
                    #       "accelerator": "gpu",
                    #       "gpus": [0]
                    #     }
                )
                model_NBEATS.fit(train, past_covariates=past_covs, epochs=100, verbose=True)
                predl_NBEATS = model_NBEATS.predict(series=train, n=AHEAD)

                # # scale back:
                # pred = scaler.i
                # nverse_transform(pred)


                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_NBEATS)],
                          'mape': [mape(test, predl_NBEATS)],
                          # 'marre': [marre(test, pred_Regression)],
                          'mse': [mse(test, predl_NBEATS)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E02/E016_NBEATS.csv', mode='a', header=None)
                print('End ' + 'NBEATS')
                # print(' begin RNN')
                # model_RNN = BlockRNNModel(
                #     input_chunk_length=HIS,
                #     output_chunk_length=AHEAD,
                #     model='RNN',  # “LSTM” or “GRU”
                #     random_state=42,
                #     hidden_size=64,
                #     n_rnn_layers=2,
                #     dropout=0.2,
                #     torch_device_str='cuda'
                #
                # )
                # model_RNN.fit(train, past_covariates=past_covs)
                # predl_RNN = model_RNN.predict(series=train, n=AHEAD, past_covariates=series_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                #
                # output = {'file_name': [file_name],
                #           'mae': [mae(test, predl_RNN)],
                #           'mape': [mape(test, predl_RNN)],
                #           # 'marre': [marre(test, predl_RNN)],
                #           'mse': [mse(test, predl_RNN)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_RNN.csv', mode='a', header=None)
                # print('End ' + 'RNN')
                #
                # #
                # print(' begin GRU')
                # model_GRU = BlockRNNModel(
                #     input_chunk_length=HIS,
                #     output_chunk_length=AHEAD,
                #     model='GRU',  # “LSTM” or “GRU”
                #     random_state=42,
                #     hidden_size=64,
                #     n_rnn_layers=2,
                #     dropout=0.2,
                #     torch_device_str='cuda'
                #
                # )
                # model_GRU.fit(train, past_covariates=past_covs)
                # predl_GRU = model_GRU.predict(series=train, n=AHEAD, past_covariates=series_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                #
                #
                # output_GRU = {'file_name': [file_name],
                #               'mae': [mae(test, predl_GRU)],
                #               'mape': [mape(test, predl_GRU)],
                #               # 'marre': [marre(test, predl_RNN)],
                #               'mse': [mse(test, predl_GRU)]
                #               }
                # output_GRU_df = pd.DataFrame(output_GRU)
                #
                # output_GRU_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_GRU.csv', mode='a', header=None)
                # print('End ' + 'GRU')
                # #
                # print('begin LSTM')
                # model_LSTM = BlockRNNModel(
                #     input_chunk_length=HIS,
                #     output_chunk_length=AHEAD,
                #     model='LSTM',  # “LSTM” or “GRU”
                #     random_state=42,
                #     hidden_size=64,
                #     n_rnn_layers=2,
                #     dropout=0.2,
                #     torch_device_str='cuda'
                #
                # )
                # model_LSTM.fit(train, past_covariates=past_covs)
                # predl_LSTM = model_LSTM.predict(series=train, n=AHEAD, past_covariates=series_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                #
                # output_LSTM = {'file_name': [file_name],
                #                'mae': [mae(test, predl_LSTM)],
                #                'mape': [mape(test, predl_LSTM)],
                #                # 'marre': [marre(test, predl_LSTM)],
                #                'mse': [mse(test, predl_LSTM)]
                #                }
                # output_LSTM_df = pd.DataFrame(output_LSTM)
                #
                # output_LSTM_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_LSTM.csv', mode='a', header=None)
                # print('End ' + 'LSTM')
                # print('begin TCN ')
                # model_TCNModel = TCNModel(
                #     input_chunk_length=HIS,
                #     output_chunk_length=AHEAD,
                #     kernel_size=3,
                #     num_filters=3,
                #     dilation_base=2,
                #     weight_norm=False,
                #     dropout=0.2,
                #     torch_device_str='cuda'
                #
                # )
                # model_TCNModel.fit(train, past_covariates=past_covs)
                # predl_TCNModel = model_TCNModel.predict(series=train, n=AHEAD, past_covariates=series_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                #
                # output = {'file_name': [file_name],
                #           'mae': [mae(test, predl_TCNModel)],
                #           'mape': [mape(test, predl_TCNModel)],
                #           # 'marre': [marre(test, predl_TCNModel)],
                #           'mse': [mse(test, predl_TCNModel)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_TCN.csv', mode='a', header=None)
                # print('End ' + 'TCN')
                # print('begin TFT')
                # model_TransformerModel = TransformerModel(
                #     input_chunk_length=HIS,
                #     output_chunk_length=AHEAD,
                #     torch_device_str='cuda'
                #
                # )
                # model_TransformerModel.fit(train, past_covariates=past_covs)
                # predl_TransformerModel = model_TransformerModel.predict(series=train, n=AHEAD,
                #                                                         past_covariates=series_covs)
                #
                # # # scale back:
                # # pred = scaler.inverse_transform(pred)
                #
                # output = {'file_name': [file_name],
                #           'mae': [mae(test, predl_TransformerModel)],
                #           'mape': [mape(test, predl_TransformerModel)],
                #           # 'marre': [marre(test, predl_TransformerModel)],
                #           'mse': [mse(test, predl_TransformerModel)]
                #           }
                # output_df = pd.DataFrame(output)
                #
                # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E016/E016_Transformer.csv', mode='a', header=None)
                # print('End ' + 'TFT')

    print('------------------E012----------------------')
    LIST_TAR = [4]
    LIST_COVS=[0,1,2,3,4,5,9,10,11,13,14,16]
    for path, dir_list, file_list in os.walk(r"D:/data/E012"):
        for file_name in file_list:
            if file_name == '.csv':
                switch = 0
            if switch == 0:
                # if file_name == 'E001 2021 10.csv':
                #     continue

                file = os.path.join(path, file_name)
                print(file)
                df = pd.read_csv(file, index_col='time', low_memory=False)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.resample('S').asfreq()
                df = df.apply(pd.to_numeric, errors='ignore')
                df[df['in_Sheet_thickness'] > 5] = np.nan
                df[df['in_Sheet_thickness'] <= 0] = np.nan

                list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
                df.iloc[:, list] = StandardScaler().fit_transform(df.iloc[:, list])  # 数值类型特征标准化
                # L = df.iloc[:, list]
                L_imputed = df.iloc[:, list]
                L_imputed.fillna(method='ffill', inplace=True)

                data_tar = L_imputed.iloc[-LEN:, LIST_TAR]
                data_covs = L_imputed.iloc[-LEN:, LIST_COVS]

                series_tar = TimeSeries.from_dataframe(data_tar, freq='S').astype(np.float32)
                train, test = series_tar[:-AHEAD], series_tar[-AHEAD:]
                series_covs = TimeSeries.from_dataframe(data_covs, freq='S').astype(np.float32)
                past_covs, future_cosv = series_covs[:-AHEAD], series_covs[-AHEAD:]
    # #
    # #             model_Regression = RegressionModel(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 model=BayesianRidge()
    # #             )
    # #             model_Regression.fit(train, past_covariates=past_covs)
    # #             pred_Regression = model_Regression.predict(series=train, past_covariates=past_covs, n=AHEAD)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 16))
    # #             train[-HIS:].plot(label="train")
    # #             test.plot(label="actual")
    # #
    # #             output = {'file_name': [file_name],
    # #                       'mae': [mae(test, pred_Regression)],
    # #                       'mape': [mape(test, pred_Regression)],
    # #                       # 'marre': [marre(test, pred_Regression)],
    # #                       'mse': [mse(test, pred_Regression)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_BayesianRidge.csv', mode='a', header=None)
    # #             print('End ' + 'BayesianRidge')
    # #
    # #             model_RandomForest = RandomForest(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 n_estimators=100,
    # #                 max_depth=None
    # #
    # #             )
    # #             model_RandomForest.fit(train, past_covariates=past_covs)
    # #             predl_RandomForest = model_RandomForest.predict(series=train, n=AHEAD, past_covariates=series_covs)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 10))
    # #             train[-150:].plot(label="train")
    # #             test.plot(label="actual")
    # #             predl_RandomForest.plot(label="forecast")
    # #
    # #             output = {'file_name': [file_name],
    # #                       'mae': [mae(test, predl_RandomForest)],
    # #                       'mape': [mape(test, predl_RandomForest)],
    # #                       # 'marre': [marre(test, predl_RandomForest)],
    # #                       'mse': [mse(test, predl_RandomForest)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_RandomForest.csv', mode='a', header=None)
    # #             print('End ' + 'model_RandomForest')
    # #
    # #
    # #             model_LinearRegressionModel = LinearRegressionModel(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 random_state=43
    # #
    # #             )
    # #             model_LinearRegressionModel.fit(train, past_covariates=series_covs)
    # #             predl_LinearRegressionModel = model_LinearRegressionModel.predict(series=train, n=AHEAD,
    # #                                                                               past_covariates=past_covs)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 10))
    # #             train[-HIS:].plot(label="train")
    # #             test.plot(label="actual")
    # #             predl_LinearRegressionModel.plot(label="forecast")
    # #
    # #
    # #             output = {'file_name':[file_name],
    # #                       'mae': [mae(test, predl_LinearRegressionModel)],
    # #                       'mape': [mape(test, predl_LinearRegressionModel)],
    # #                       # 'marre': [marre(test, predl_LinearRegressionModel)],
    # #                       'mse': [mse(test, predl_LinearRegressionModel)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_LinearRegression.csv', mode='a', header = None)
    # #             print('End ' + 'LinearRegressionModel')
    # #
    # #             model_LightGBMModel = LightGBMModel(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 random_state=43,
    # #
    # #             )
    # #             model_LightGBMModel.fit(train, past_covariates=past_covs)
    # #             predl_LightGBMModel = model_LightGBMModel.predict(series=train, n=AHEAD, past_covariates=series_covs)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 10))
    # #             train[-150:].plot(label="train")
    # #             test.plot(label="actual")
    # #             predl_LightGBMModel.plot(label="forecast")
    # #
    # #             output = {'file_name': [file_name],
    # #                       'mae': [mae(test, predl_LightGBMModel)],
    # #                       'mape': [mape(test, predl_LightGBMModel)],
    # #                       # 'marre': [marre(test, predl_LightGBMModel)],
    # #                       'mse': [mse(test, predl_LightGBMModel)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_LightGBM.csv', mode='a', header=None)
    # #             print('End ' + 'LightGBM')
                print(' begin NBEATS')
                model_NBEATS = NBEATSModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    random_state=42,
                    torch_device_str='cuda'
                    #     pl_trainer_kwargs={
                    #       "accelerator": "gpu",
                    #       "gpus": [0]
                    #     }
                )
                model_NBEATS.fit(train, past_covariates=past_covs, epochs=100, verbose=True)
                predl_NBEATS = model_NBEATS.predict(series=train, n=AHEAD)

                # # scale back:
                # pred = scaler.i
                # nverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-HIS:].plot(label="train")
                test.plot(label="actual")
                predl_NBEATS.plot(label="forecast")

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_NBEATS)],
                          'mape': [mape(test, predl_NBEATS)],
                          # 'marre': [marre(test, pred_Regression)],
                          'mse': [mse(test, predl_NBEATS)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E01/E012_NBEATS.csv', mode='a', header=None)
                print('End ' + 'NBEATS')
    #             print(' begin RNN')
    #             model_RNN = BlockRNNModel(
    #                 input_chunk_length=HIS,
    #                 output_chunk_length=AHEAD,
    #                 model='RNN',  # “LSTM” or “GRU”
    #                 random_state=42,
    #                 hidden_size=64,
    #                 n_rnn_layers=2,
    #                 dropout=0.2,
    #                 torch_device_str='cuda'
    #
    #             )
    #             model_RNN.fit(train, past_covariates=past_covs)
    #             predl_RNN = model_RNN.predict(series=train, n=AHEAD, past_covariates=series_covs)
    #
    #             # # scale back:
    #             # pred = scaler.inverse_transform(pred)
    #
    #             plt.figure(figsize=(35, 10))
    #             train[-150:].plot(label="train")
    #             test.plot(label="actual")
    #             predl_RNN.plot(label="forecast")
    #
    #             output = {'file_name': [file_name],
    #                       'mae': [mae(test, predl_RNN)],
    #                       'mape': [mape(test, predl_RNN)],
    #                       # 'marre': [marre(test, predl_RNN)],
    #                       'mse': [mse(test, predl_RNN)]
    #                       }
    #             output_df = pd.DataFrame(output)
    #
    #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_RNN.csv', mode='a', header=None)
    #             print('End ' + 'RNN')
    #
    #             #
    #             print(' begin GRU')
    #             model_GRU = BlockRNNModel(
    #                 input_chunk_length=HIS,
    #                 output_chunk_length=AHEAD,
    #                 model='GRU',  # “LSTM” or “GRU”
    #                 random_state=42,
    #                 hidden_size=64,
    #                 n_rnn_layers=2,
    #                 dropout=0.2,
    #                 torch_device_str='cuda'
    #
    #             )
    #             model_GRU.fit(train, past_covariates=past_covs)
    #             predl_GRU = model_GRU.predict(series=train, n=AHEAD, past_covariates=series_covs)
    #
    #             # # scale back:
    #             # pred = scaler.inverse_transform(pred)
    #
    #             plt.figure(figsize=(35, 10))
    #             train[-150:].plot(label="train")
    #             test.plot(label="actual")
    #             predl_GRU.plot(label="forecast")
    #             output_GRU = {'file_name': [file_name],
    #                           'mae': [mae(test, predl_GRU)],
    #                           'mape': [mape(test, predl_GRU)],
    #                           # 'marre': [marre(test, predl_RNN)],
    #                           'mse': [mse(test, predl_GRU)]
    #                           }
    #             output_GRU_df = pd.DataFrame(output_GRU)
    #
    #             output_GRU_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_GRU.csv', mode='a', header=None)
    #             print('End ' + 'GRU')
    #             #
    #             print('begin LSTM')
    #             model_LSTM = BlockRNNModel(
    #                 input_chunk_length=HIS,
    #                 output_chunk_length=AHEAD,
    #                 model='LSTM',  # “LSTM” or “GRU”
    #                 random_state=42,
    #                 hidden_size=64,
    #                 n_rnn_layers=2,
    #                 dropout=0.2,
    #                 torch_device_str='cuda'
    #
    #             )
    #             model_LSTM.fit(train, past_covariates=past_covs)
    #             predl_LSTM = model_LSTM.predict(series=train, n=AHEAD, past_covariates=series_covs)
    #
    #             # # scale back:
    #             # pred = scaler.inverse_transform(pred)
    #
    #             plt.figure(figsize=(35, 10))
    #             train[-150:].plot(label="train")
    #             test.plot(label="actual")
    #             predl_LSTM.plot(label="forecast")
    #             output_LSTM = {'file_name': [file_name],
    #                            'mae': [mae(test, predl_LSTM)],
    #                            'mape': [mape(test, predl_LSTM)],
    #                            # 'marre': [marre(test, predl_LSTM)],
    #                            'mse': [mse(test, predl_LSTM)]
    #                            }
    #             output_LSTM_df = pd.DataFrame(output_LSTM)
    #
    #             output_LSTM_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_LSTM.csv', mode='a', header=None)
    #             print('End ' + 'LSTM')
    #             print('begin TCN ')
    #             model_TCNModel = TCNModel(
    #                 input_chunk_length=HIS,
    #                 output_chunk_length=AHEAD,
    #                 kernel_size=3,
    #                 num_filters=3,
    #                 dilation_base=2,
    #                 weight_norm=False,
    #                 dropout=0.2,
    #                 torch_device_str='cuda'
    #
    #             )
    #             model_TCNModel.fit(train, past_covariates=past_covs)
    #             predl_TCNModel = model_TCNModel.predict(series=train, n=AHEAD, past_covariates=series_covs)
    #
    #             # # scale back:
    #             # pred = scaler.inverse_transform(pred)
    #
    #             plt.figure(figsize=(35, 10))
    #             train[-150:].plot(label="train")
    #             test.plot(label="actual")
    #             predl_TCNModel.plot(label="forecast")
    #
    #             output = {'file_name': [file_name],
    #                       'mae': [mae(test, predl_TCNModel)],
    #                       'mape': [mape(test, predl_TCNModel)],
    #                       # 'marre': [marre(test, predl_TCNModel)],
    #                       'mse': [mse(test, predl_TCNModel)]
    #                       }
    #             output_df = pd.DataFrame(output)
    #
    #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_TCN.csv', mode='a', header=None)
    #             print('End ' + 'TCN')
    #             print('begin TFT')
    #             model_TransformerModel = TransformerModel(
    #                 input_chunk_length=HIS,
    #                 output_chunk_length=AHEAD,
    #                 torch_device_str='cuda'
    #
    #             )
    #             model_TransformerModel.fit(train, past_covariates=past_covs)
    #             predl_TransformerModel = model_TransformerModel.predict(series=train, n=AHEAD,
    #                                                                     past_covariates=series_covs)
    #
    #             # # scale back:
    #             # pred = scaler.inverse_transform(pred)
    #
    #             plt.figure(figsize=(35, 10))
    #             train[-150:].plot(label="train")
    #             test.plot(label="actual")
    #             predl_TransformerModel.plot(label="forecast")
    #
    #             output = {'file_name': [file_name],
    #                       'mae': [mae(test, predl_TransformerModel)],
    #                       'mape': [mape(test, predl_TransformerModel)],
    #                       # 'marre': [marre(test, predl_TransformerModel)],
    #                       'mse': [mse(test, predl_TransformerModel)]
    #                       }
    #             output_df = pd.DataFrame(output)
    #
    #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E012_Transformer.csv', mode='a', header=None)
    #             print('End ' + 'TFT')
    #
    print('------------------E028----------------------')
    LIST_TAR = [4]
    LIST_COVS=[0,1,2,3,4,5,9,10,11,13,14,16]
    for path, dir_list, file_list in os.walk(r"D:/data/E028"):
        for file_name in file_list:
            if file_name == 'E028 2021 85.csv':
                switch = 0
            if switch == 0:
                # if file_name == 'E001 2021 10.csv':
                #     continue

                file = os.path.join(path, file_name)
                print(file)
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
    #
    # #             model_Regression = RegressionModel(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 model=BayesianRidge()
    # #             )
    # #             model_Regression.fit(train, past_covariates=past_covs)
    # #             pred_Regression = model_Regression.predict(series=train, past_covariates=past_covs, n=AHEAD)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 16))
    # #             train[-HIS:].plot(label="train")
    # #             test.plot(label="actual")
    # #
    # #             output = {'file_name': [file_name],
    # #                       'mae': [mae(test, pred_Regression)],
    # #                       'mape': [mape(test, pred_Regression)],
    # #                       # 'marre': [marre(test, pred_Regression)],
    # #                       'mse': [mse(test, pred_Regression)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_BayesianRidge.csv', mode='a', header=None)
    # #             print('End ' + file_name)
    # #
    # #             model_RandomForest = RandomForest(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 n_estimators=100,
    # #                 max_depth=None
    # #
    # #             )
    # #             model_RandomForest.fit(train, past_covariates=past_covs)
    # #             predl_RandomForest = model_RandomForest.predict(series=train, n=AHEAD, past_covariates=series_covs)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 10))
    # #             train[-150:].plot(label="train")
    # #             test.plot(label="actual")
    # #             predl_RandomForest.plot(label="forecast")
    # #
    # #             output = {'file_name': [file_name],
    # #                       'mae': [mae(test, predl_RandomForest)],
    # #                       'mape': [mape(test, predl_RandomForest)],
    # #                       # 'marre': [marre(test, predl_RandomForest)],
    # #                       'mse': [mse(test, predl_RandomForest)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_RandomForest.csv', mode='a', header=None)
    # #             print('End ' + file_name)
    # #
    # #
    # #             model_LinearRegressionModel = LinearRegressionModel(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 random_state=43
    # #
    # #             )
    # #             model_LinearRegressionModel.fit(train, past_covariates=series_covs)
    # #             predl_LinearRegressionModel = model_LinearRegressionModel.predict(series=train, n=AHEAD,
    # #                                                                               past_covariates=past_covs)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 10))
    # #             train[-HIS:].plot(label="train")
    # #             test.plot(label="actual")
    # #             predl_LinearRegressionModel.plot(label="forecast")
    # #
    # #
    # #             output = {'file_name':[file_name],
    # #                       'mae': [mae(test, predl_LinearRegressionModel)],
    # #                       'mape': [mape(test, predl_LinearRegressionModel)],
    # #                       # 'marre': [marre(test, predl_LinearRegressionModel)],
    # #                       'mse': [mse(test, predl_LinearRegressionModel)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_LinearRegression.csv', mode='a', header = None)
    # #             print('End ' + file_name)
    # #
    # #             model_LightGBMModel = LightGBMModel(
    # #                 lags=HIS,
    # #                 lags_past_covariates=HIS,
    # #                 output_chunk_length=AHEAD,
    # #                 random_state=43,
    # #
    # #             )
    # #             model_LightGBMModel.fit(train, past_covariates=past_covs)
    # #             predl_LightGBMModel = model_LightGBMModel.predict(series=train, n=AHEAD, past_covariates=series_covs)
    # #
    # #             # # scale back:
    # #             # pred = scaler.inverse_transform(pred)
    # #
    # #             plt.figure(figsize=(35, 10))
    # #             train[-150:].plot(label="train")
    # #             test.plot(label="actual")
    # #             predl_LightGBMModel.plot(label="forecast")
    # #
    # #             output = {'file_name': [file_name],
    # #                       'mae': [mae(test, predl_LightGBMModel)],
    # #                       'mape': [mape(test, predl_LightGBMModel)],
    # #                       # 'marre': [marre(test, predl_LightGBMModel)],
    # #                       'mse': [mse(test, predl_LightGBMModel)]
    # #                       }
    # #             output_df = pd.DataFrame(output)
    # #
    # #             output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_LightGBM.csv', mode='a', header=None)
    # #             print('End ' + file_name)
    # #
                model_NBEATS = NBEATSModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    random_state=42,
                    torch_device_str='cuda'
                    #     pl_trainer_kwargs={
                    #       "accelerator": "gpu",
                    #       "gpus": [0]
                    #     }
                )
                model_NBEATS.fit(train, past_covariates=past_covs, epochs=100, verbose=True)
                predl_NBEATS = model_NBEATS.predict(series=train, n=AHEAD)

                # # scale back:
                # pred = scaler.i
                # nverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-HIS:].plot(label="train")
                test.plot(label="actual")
                predl_NBEATS.plot(label="forecast")

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_NBEATS)],
                          'mape': [mape(test, predl_NBEATS)],
                          # 'marre': [marre(test, pred_Regression)],
                          'mse': [mse(test, predl_NBEATS)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E03/E028_NBEATS.csv', mode='a', header=None)
                print('End ' + file_name)

                model_RNN = BlockRNNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    model='RNN',  # “LSTM” or “GRU”
                    random_state=42,
                    hidden_size=64,
                    n_rnn_layers=2,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_RNN.fit(train, past_covariates=past_covs)
                predl_RNN = model_RNN.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_RNN.plot(label="forecast")

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_RNN)],
                          'mape': [mape(test, predl_RNN)],
                          # 'marre': [marre(test, predl_RNN)],
                          'mse': [mse(test, predl_RNN)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_RNN.csv', mode='a', header=None)
                print('End ' + file_name)

                #
                model_GRU = BlockRNNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    model='GRU',  # “LSTM” or “GRU”
                    random_state=42,
                    hidden_size=64,
                    n_rnn_layers=2,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_GRU.fit(train, past_covariates=past_covs)
                predl_GRU = model_GRU.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_GRU.plot(label="forecast")
                output_GRU = {'file_name': [file_name],
                              'mae': [mae(test, predl_GRU)],
                              'mape': [mape(test, predl_GRU)],
                              # 'marre': [marre(test, predl_RNN)],
                              'mse': [mse(test, predl_GRU)]
                              }
                output_GRU_df = pd.DataFrame(output_GRU)

                output_GRU_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_GRU.csv', mode='a', header=None)
                print('End ' + file_name)
                #
                model_LSTM = BlockRNNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    model='LSTM',  # “LSTM” or “GRU”
                    random_state=42,
                    hidden_size=64,
                    n_rnn_layers=2,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_LSTM.fit(train, past_covariates=past_covs)
                predl_LSTM = model_LSTM.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_LSTM.plot(label="forecast")
                output_LSTM = {'file_name': [file_name],
                               'mae': [mae(test, predl_LSTM)],
                               'mape': [mape(test, predl_LSTM)],
                               # 'marre': [marre(test, predl_LSTM)],
                               'mse': [mse(test, predl_LSTM)]
                               }
                output_LSTM_df = pd.DataFrame(output_LSTM)

                output_LSTM_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_LSTM.csv', mode='a', header=None)
                print('End ' + file_name)

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

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_TCNModel)],
                          'mape': [mape(test, predl_TCNModel)],
                          # 'marre': [marre(test, predl_TCNModel)],
                          'mse': [mse(test, predl_TCNModel)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_TCN.csv', mode='a', header=None)
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

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E028_Transformer.csv', mode='a', header=None)
                print('End ' + file_name)
    #
    print('------------------E029----------------------')
    LIST_TAR = [4]
    LIST_COVS=[0,1,2,3,4,5,9,10,11,13,14,16]
    for path, dir_list, file_list in os.walk(r"D:/data/E029"):
        for file_name in file_list:
            if file_name == 'E029 2021 95.csv':
                switch = 0
            if switch == 0:
                # if file_name == 'E001 2021 10.csv':
                #     continue

                file = os.path.join(path, file_name)
                print(file)
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
                #
    #             # model_Regression = RegressionModel(
    #             #     lags=HIS,
    #             #     lags_past_covariates=HIS,
    #             #     output_chunk_length=AHEAD,
    #             #     model=BayesianRidge()
    #             # )
    #             # model_Regression.fit(train, past_covariates=past_covs)
    #             # pred_Regression = model_Regression.predict(series=train, past_covariates=past_covs, n=AHEAD)
    #             #
    #             # # # scale back:
    #             # # pred = scaler.inverse_transform(pred)
    #             #
    #             # plt.figure(figsize=(35, 16))
    #             # train[-HIS:].plot(label="train")
    #             # test.plot(label="actual")
    #             #
    #             # output = {'file_name': [file_name],
    #             #           'mae': [mae(test, pred_Regression)],
    #             #           'mape': [mape(test, pred_Regression)],
    #             #           # 'marre': [marre(test, pred_Regression)],
    #             #           'mse': [mse(test, pred_Regression)]
    #             #           }
    #             # output_df = pd.DataFrame(output)
    #             #
    #             # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_BayesianRidge.csv', mode='a', header=None)
    #             # print('End ' + file_name)
    #             #
    #             # model_RandomForest = RandomForest(
    #             #     lags=HIS,
    #             #     lags_past_covariates=HIS,
    #             #     output_chunk_length=AHEAD,
    #             #     n_estimators=100,
    #             #     max_depth=None
    #             #
    #             # )
    #             # model_RandomForest.fit(train, past_covariates=past_covs)
    #             # predl_RandomForest = model_RandomForest.predict(series=train, n=AHEAD, past_covariates=series_covs)
    #             #
    #             # # # scale back:
    #             # # pred = scaler.inverse_transform(pred)
    #             #
    #             # plt.figure(figsize=(35, 10))
    #             # train[-150:].plot(label="train")
    #             # test.plot(label="actual")
    #             # predl_RandomForest.plot(label="forecast")
    #             #
    #             # output = {'file_name': [file_name],
    #             #           'mae': [mae(test, predl_RandomForest)],
    #             #           'mape': [mape(test, predl_RandomForest)],
    #             #           # 'marre': [marre(test, predl_RandomForest)],
    #             #           'mse': [mse(test, predl_RandomForest)]
    #             #           }
    #             # output_df = pd.DataFrame(output)
    #             #
    #             # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_RandomForest.csv', mode='a', header=None)
    #             # print('End ' + file_name)
    #             #
    #             #
    #             # model_LinearRegressionModel = LinearRegressionModel(
    #             #     lags=HIS,
    #             #     lags_past_covariates=HIS,
    #             #     output_chunk_length=AHEAD,
    #             #     random_state=43
    #             #
    #             # )
    #             # model_LinearRegressionModel.fit(train, past_covariates=series_covs)
    #             # predl_LinearRegressionModel = model_LinearRegressionModel.predict(series=train, n=AHEAD,
    #             #                                                                   past_covariates=past_covs)
    #             #
    #             # # # scale back:
    #             # # pred = scaler.inverse_transform(pred)
    #             #
    #             # plt.figure(figsize=(35, 10))
    #             # train[-HIS:].plot(label="train")
    #             # test.plot(label="actual")
    #             # predl_LinearRegressionModel.plot(label="forecast")
    #             #
    #             #
    #             # output = {'file_name':[file_name],
    #             #           'mae': [mae(test, predl_LinearRegressionModel)],
    #             #           'mape': [mape(test, predl_LinearRegressionModel)],
    #             #           # 'marre': [marre(test, predl_LinearRegressionModel)],
    #             #           'mse': [mse(test, predl_LinearRegressionModel)]
    #             #           }
    #             # output_df = pd.DataFrame(output)
    #             #
    #             # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_LinearRegression.csv', mode='a', header = None)
    #             # print('End ' + file_name)
    #             #
    #             # model_LightGBMModel = LightGBMModel(
    #             #     lags=HIS,
    #             #     lags_past_covariates=HIS,
    #             #     output_chunk_length=AHEAD,
    #             #     random_state=43,
    #             #
    #             # )
    #             # model_LightGBMModel.fit(train, past_covariates=past_covs)
    #             # predl_LightGBMModel = model_LightGBMModel.predict(series=train, n=AHEAD, past_covariates=series_covs)
    #             #
    #             # # # scale back:
    #             # # pred = scaler.inverse_transform(pred)
    #             #
    #             # plt.figure(figsize=(35, 10))
    #             # train[-150:].plot(label="train")
    #             # test.plot(label="actual")
    #             # predl_LightGBMModel.plot(label="forecast")
    #             #
    #             # output = {'file_name': [file_name],
    #             #           'mae': [mae(test, predl_LightGBMModel)],
    #             #           'mape': [mape(test, predl_LightGBMModel)],
    #             #           # 'marre': [marre(test, predl_LightGBMModel)],
    #             #           'mse': [mse(test, predl_LightGBMModel)]
    #             #           }
    #             # output_df = pd.DataFrame(output)
    #             #
    #             # output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_LightGBM.csv', mode='a', header=None)
    #             # print('End ' + file_name)
    #
                model_NBEATS = NBEATSModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    random_state=42,
                    torch_device_str='cuda'
                    #     pl_trainer_kwargs={
                    #       "accelerator": "gpu",
                    #       "gpus": [0]
                    #     }
                )
                model_NBEATS.fit(train, past_covariates=past_covs, epochs=100, verbose=True)
                predl_NBEATS = model_NBEATS.predict(series=train, n=AHEAD)

                # # scale back:
                # pred = scaler.i
                # nverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-HIS:].plot(label="train")
                test.plot(label="actual")
                predl_NBEATS.plot(label="forecast")

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_NBEATS)],
                          'mape': [mape(test, predl_NBEATS)],
                          # 'marre': [marre(test, pred_Regression)],
                          'mse': [mse(test, predl_NBEATS)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E04/E029_NBEATS.csv', mode='a', header=None)
                print('End ' + file_name)

                model_RNN = BlockRNNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    model='RNN',  # “LSTM” or “GRU”
                    random_state=42,
                    hidden_size=64,
                    n_rnn_layers=2,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_RNN.fit(train, past_covariates=past_covs)
                predl_RNN = model_RNN.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_RNN.plot(label="forecast")

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_RNN)],
                          'mape': [mape(test, predl_RNN)],
                          # 'marre': [marre(test, predl_RNN)],
                          'mse': [mse(test, predl_RNN)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_RNN.csv', mode='a', header=None)
                print('End ' + file_name)

                #
                model_GRU = BlockRNNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    model='GRU',  # “LSTM” or “GRU”
                    random_state=42,
                    hidden_size=64,
                    n_rnn_layers=2,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_GRU.fit(train, past_covariates=past_covs)
                predl_GRU = model_GRU.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_GRU.plot(label="forecast")
                output_GRU = {'file_name': [file_name],
                              'mae': [mae(test, predl_GRU)],
                              'mape': [mape(test, predl_GRU)],
                              # 'marre': [marre(test, predl_RNN)],
                              'mse': [mse(test, predl_GRU)]
                              }
                output_GRU_df = pd.DataFrame(output_GRU)

                output_GRU_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_GRU.csv', mode='a', header=None)
                print('End ' + file_name)
                #
                model_LSTM = BlockRNNModel(
                    input_chunk_length=HIS,
                    output_chunk_length=AHEAD,
                    model='LSTM',  # “LSTM” or “GRU”
                    random_state=42,
                    hidden_size=64,
                    n_rnn_layers=2,
                    dropout=0.2,
                    torch_device_str='cuda'

                )
                model_LSTM.fit(train, past_covariates=past_covs)
                predl_LSTM = model_LSTM.predict(series=train, n=AHEAD, past_covariates=series_covs)

                # # scale back:
                # pred = scaler.inverse_transform(pred)

                plt.figure(figsize=(35, 10))
                train[-150:].plot(label="train")
                test.plot(label="actual")
                predl_LSTM.plot(label="forecast")
                output_LSTM = {'file_name': [file_name],
                               'mae': [mae(test, predl_LSTM)],
                               'mape': [mape(test, predl_LSTM)],
                               # 'marre': [marre(test, predl_LSTM)],
                               'mse': [mse(test, predl_LSTM)]
                               }
                output_LSTM_df = pd.DataFrame(output_LSTM)

                output_LSTM_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_LSTM.csv', mode='a', header=None)
                print('End ' + file_name)

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

                output = {'file_name': [file_name],
                          'mae': [mae(test, predl_TCNModel)],
                          'mape': [mape(test, predl_TCNModel)],
                          # 'marre': [marre(test, predl_TCNModel)],
                          'mse': [mse(test, predl_TCNModel)]
                          }
                output_df = pd.DataFrame(output)

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_TCN.csv', mode='a', header=None)
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

                output_df.to_csv('E:/01读博/小论文/Benchmark paper/实验/实验1/E029_Transformer.csv', mode='a', header=None)
                print('End ' + file_name)
