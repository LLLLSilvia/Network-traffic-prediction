import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import *
import numpy as np
import pandas as pd
import csv
import matplotlib as plt
import time

# Hyper Parameters
EPOCH = 30
BATCH_SIZE = 50
FILE_NAME = "F:/Facebook/final.csv"
#LR = 0.015
LR = 0.055
INPUT_SIZE = 9
HIDDEN_SIZE = 100
NUM_LAYERS = 1
K = 10
USE_CUDA = True
# 参数设置 https://blog.csdn.net/xiaodongxiexie/article/details/71131562

class AlexNet_LSTM(nn.Module):
    def __init__(self):
        super(AlexNet_LSTM, self).__init__()

        ############################### AlexNet CNN #################################

        self.features = nn.Sequential(  # input size (1, 23, 23)
            nn.Conv2d(1, 32, kernel_size=11, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),## 11 11 96
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout()
        )  # (192, 7, 7)

        self.features2 = nn.Sequential(  # input size (1, 23, 23)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # # nn.Dropout(),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # # nn.Dropout(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),

            # nn.Conv2d(128, 128, kernel_size=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout(),

        )

        # CNN to LSTM
        self.middle_out = nn.Linear(128 * 6 * 6, INPUT_SIZE)
        ############################### AlexNet CNN #################################

        ############################## LSTM ###################################
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(HIDDEN_SIZE, INPUT_SIZE * INPUT_SIZE)
        ############################## LSTM ###################################

    def forward(self, x, batch_size):
        x = self.features2(x)
        # print("AlexNet CNN output shape:", x.shape)
        x = x.view(x.size(0), -1)
        middle_output = self.middle_out(x)
        middle_output = middle_output.reshape(batch_size, K, INPUT_SIZE)

        # LSTM
        r_out, (h_n, h_c) = self.rnn(middle_output, None)  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])  # return the last value
        out = self.sigmoid(out)
        return out


# simple CNN with LSTM model
class CNN_LSTM(nn.Module):
    # input image like: (batch_size, 1, input_size, input_size)
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        ############################### CNN ###################################
        self.conv1 = nn.Sequential(  # input size: (1, 9, 9)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # output height out_channels取决于卷积核的数量。此时的 out_channels 也会作为下一次卷积时的卷积核的 in_channels；
                kernel_size=3,  # filter size
                stride=1,  # filter moving step
                padding=2,  # padding
            ),  # output size: (16, 11, 11) https://blog.csdn.net/program_developer/article/details/80943707
            nn.BatchNorm2d(16), # 进行数据的归一化处理，防止梯度爆炸
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, 1, 2),  # 16,11,11
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, padding=1),  # output size: (16,6, 6)
            nn.Dropout(), # 防止过拟合
        )

        # CNN
        self.conv2 = nn.Sequential(  # input size: (16, 12, 12)
            nn.Conv2d(16, 32, 5, 1, 2),  # output size: (32, 12, 12)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output size: (32, 6, 6)
            nn.Dropout()
        )

        # CNN
        self.conv3 = nn.Sequential(  # input size: (32, 6, 6)
            nn.Conv2d(32, 64, 5, 1, 2),  # output size: (64, 6, 6)
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output size: (64, 3, 3)
            nn.Dropout()
        )


        # CNN to LSTM
        self.middle_out = nn.Linear(16 * 6 * 6, INPUT_SIZE)
        ############################### CNN ###################################

        ############################## LSTM ###################################
        self.rnn = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(HIDDEN_SIZE, INPUT_SIZE * INPUT_SIZE)
        ############################## LSTM ###################################

    def forward(self, x, batch_size):
        # print("-------------------------------NN forward-------------------------------")
        # CNN
        x = self.conv1(x)
        # print("conv1, x.shape:", x.shape) #[500, 16, 6, 6]
        # x = self.conv2(x)
        # print("conv2, x.shape:", x.shape)
        # x = self.conv3(x)
        # print("x.shape:", x.shape)

        x = x.view(x.size(0), -1)  # (batch_size, (16,6, 6)) -> (batch_size, 16 * 6 * 6) -1自动补齐
        # print("x.view:", x.shape) #[500, 576]
        middle_output = self.middle_out(x)  # (batch_size * K, input_size)  # torch.Size([10, 23])
        # print(middle_output.shape) #[500, 9]
        middle_output = middle_output.reshape(batch_size, K,
                                              INPUT_SIZE)  # reshape to (batch_size, time_step, input_size)
        # print("middle_output.shape:", middle_output.shape)# [50, 10, 9]
        # LSTM
        r_out, _ = self.rnn(middle_output, None)  # None represents zero initial hidden state
        # print("r_out.shape:", r_out.shape)# [50, 10, 300]
        out = self.out(r_out[:, -1, :])  # return the last value
        # print("r_out.shape:", out.shape)# [50, 81]
        out = self.sigmoid(out)
        # print("out.shape:", out.shape)#[50, 81]
        # print("-------------------------------NN forward-------------------------------")
        return out


class PredictTM():
    def __init__(self):
        self.nn_model = CNN_LSTM()
        self.complex_nn_model = AlexNet_LSTM()

        # GPU
        if USE_CUDA:
            self.nn_model.cuda()
            self.complex_nn_model.cuda()

    def read_data(self):
        df = pd.read_csv(FILE_NAME)
        del df["timestamp"]
        data_list = df.values # numpy
        # print(data_list)
        # print(data_list.shape)
        # print(type(data_list))

        max_list = np.max(data_list, axis=0)
        min_list = np.min(data_list, axis=0)
        # print(len(max_list)) 81
        # print(len(min_list))

        # OD pair, when O = D, max = min = 0, so data_list will have some nan value
        # change these values to 0
        data_list = (data_list - min_list) / (max_list - min_list)
        data_list[np.isnan(data_list)] = 0
        # change to TM list
        data = []
        for i in range(data_list.shape[0]):
            data.append(data_list[i].reshape(INPUT_SIZE, INPUT_SIZE))
        data = np.array(data)
        return data, min_list, max_list # data 9*9

    # generate normalized time series data
    # list of ([TM1, TM2, TM3, .. TMk], [TMk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data):
        length = data.shape[0]
        x_data = []
        y_data = []
        for i in range(length - K):
            x = data[i:i + K] # (10, 9, 9)
            y = data[i + K] #(9, 9)
            x_data.append(x)
            y_data.append(y)

        x_data = torch.from_numpy(np.array(x_data)).float()
        y_data = torch.from_numpy(np.array(y_data)).float()

        return x_data, y_data

    # generate batch data
    def generate_batch_loader(self, x_data, y_data):
        torch_dataset = Data.TensorDataset(x_data, y_data)
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,  # random order data
            num_workers=2,  # multiple threading to read data
        )
        return loader

    # inverse normalization
    def inverse_normalization(self, prediction, y, max_list, min_list):
        inverse_prediction = prediction * (max_list - min_list) + min_list
        inverse_y = y * (max_list - min_list) + min_list
        return inverse_prediction, inverse_y

    # save TM result
    '''
        def save_TM(self, TM, file_name):
        f = open(file_name, 'w')
        row, column = TM.shape
        for i in range(row):
            for j in range(column):
                    temp = str(TM[i][j]) + ","
                    f.write(temp)
        f.close()
    '''
    def save_TM(self, TM, file_name):
        tmp=[]
        row, column = TM.shape
        for i in range(row):
            for j in range(column):
                tmp.append(str(TM[i][j]))
        self.write_row_to_csv(tmp, file_name)


    def train(self):
        data, min_list, max_list = self.read_data()
        x_data, y_data = self.generate_series(data)
        #print("x_data.shape:", x_data.shape) # [5990, 10, 9, 9] 75990
        # print("y_data.shape:", y_data.shape) # [5990, 9, 9]
        train_len = int(int(len(x_data) * 0.8) / 50) * 50#4750 60750
        #print("training len:", train_len)
        data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])

        # print(self.nn_model)

        optimizer = torch.optim.Adagrad(params=self.nn_model.parameters(), lr=LR)
        # optimizer = torch.optim.Adagrad(params=self.complex_nn_model.parameters(), lr=LR)
        loss_func = nn.MSELoss()

        #################################### train ####################################
        star_time = time.clock()

        for e in range(EPOCH):
            # print("Epoch:", e)
            for step, (batch_x, batch_y) in enumerate(data_loader):
                # print("batch_x.shape:", batch_x.shape) # [50, 10, 9, 9]
                # print("batch_y.shape", batch_y.shape) # [50, 9, 9]
                # print("step", step)# 95

                # GPU
                if USE_CUDA:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    prediction = self.nn_model.forward(batch_x.reshape(BATCH_SIZE * K, 1, INPUT_SIZE, INPUT_SIZE),
                                                       batch_size=BATCH_SIZE).cuda()
                    # prediction = self.complex_nn_model.forward(batch_x.reshape(BATCH_SIZE*K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE).cuda()
                else:
                    # prediction = self.nn_model.forward(batch_x.reshape(BATCH_SIZE*K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE)
                    prediction = self.complex_nn_model.forward(
                        batch_x.reshape(BATCH_SIZE * K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE)
                loss_train = []
                # break
                #print("prediction.shape:",prediction.shape) # [50, 81]
                batch_y = batch_y.reshape(BATCH_SIZE, INPUT_SIZE * INPUT_SIZE)
                loss1 = loss_func(prediction, batch_y)
                optimizer.zero_grad() # 把梯度置零 把loss关于weight的导数变成0.
                loss1.backward()
                optimizer.step()

                print("Epoch =", e, ", step =", step, ", loss:", loss1)
                loss1=loss1.cpu().data.numpy()
                loss_train.append(loss1)
                #self.write_row_to_csv(loss_train, "F:/Facebook/CNGR1/train_loss_CNN_GRU.csv")

        end_time = time.clock()
        print("traning time-----------------------------------",end_time - star_time)
        #################################### train ####################################

        # save model
        model_name = "F:/Facebook/CNGR1/CNN_GRU_LR=" + str(LR) + "_hidden=" + str(HIDDEN_SIZE) + ".pkl"
        torch.save(self.nn_model.state_dict(), model_name)
        # model_name = "complex_CNN_LSTM_LR=" + str(LR) + "_hidden=" + str(HIDDEN_SIZE) + ".pkl"
        # torch.save(self.complex_nn_model.state_dict(), model_name)

        #################################### test ####################################
        print("----------------------------test-----------------------\n")

        # load model

        self.nn_model.load_state_dict(torch.load(model_name))
        #self.complex_nn_model.load_state_dict(torch.load(model_name))
        # star_time = time.clock()
        for i in range(train_len, len(x_data)):#4750,5990
            star_time=time.clock()
            test_x = x_data[i]
            test_y = y_data[i]
            # print("test_x.shape:", test_x.shape)
            # print("test_y.shape", test_y.shape)

            # GPU
            if USE_CUDA:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                prediction = self.nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1).cuda()
                # prediction = self.complex_nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1).cuda()
            else:
                # prediction = self.nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1)
                prediction = self.complex_nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1)

            # print("prediction.shape:", prediction.shape)# torch.Size([1, 81])

            test_y = test_y.reshape(1, INPUT_SIZE * INPUT_SIZE) # [1, 81]


            # print("Loss for test data " + str(i - 2499) + " is:", loss)
            # # save result
            data = []
            test_loss=[]
            loss2 = loss_func(prediction, test_y)
            loss2 = loss2.cpu().data.numpy()
            test_loss.append(loss2)
            #self.write_row_to_csv(test_loss, "F:/Facebook/CNGR1/test_loss_CNN_GRU.csv")
            #data.append(str(i - train_len + 1))#4749
            if USE_CUDA:
                prediction = prediction.cpu().data.numpy()[0]  # numpy.ndarray
                test_y = test_y.cpu().data.numpy()[0]

                inverse_prediction, inverse_y = self.inverse_normalization(prediction, test_y, max_list, min_list)
                inverse_predictiont = torch.from_numpy(inverse_prediction)
                inverse_yt = torch.from_numpy(inverse_y)
                loss = loss_func(inverse_predictiont, inverse_yt)
                loss=loss.sqrt()
                #loss=(inverse_prediction-inverse_y)/inverse_y
                #data.append(loss)
                data.append(loss.cpu().data.numpy())
            #else:
                #data.append(loss.data.numpy())
            #self.write_row_to_csv(data, "F:/Facebook/CNGR1/RMSE_CNN_GRU.csv")
            end_time=time.clock()
            print("test----------------------------------------time",end_time-star_time)
            # inverse normalization
            if USE_CUDA:
                # print(prediction.cpu().data.numpy().shape)
                inverse_prediction = inverse_prediction.reshape(INPUT_SIZE, INPUT_SIZE)
                #path = "F:/Facebook/CNN-LSTM/" + str(i - train_len + 1) + ".csv"
                #self.write_row_to_csv(inverse_prediction, path)
                path = "F:/Facebook/CNGR1/all.csv"
                #self.save_TM(inverse_prediction, path)

            else:
                # print(prediction.data.numpy().shape)
                prediction = prediction.data.numpy()[0]
                test_y = test_y.data.numpy()[0]
                inverse_prediction, inverse_y = self.inverse_normalization(prediction, test_y, max_list, min_list)
                inverse_prediction = inverse_prediction.reshape(INPUT_SIZE, INPUT_SIZE)
                #path = "F:/Facebook/CNN-LSTM/" + str(i - train_len + 1) + ".csv"
                path = "F:/Facebook/CNGR1/all.csv"
                #self.save_TM(inverse_prediction, path)

        # end_time = time.clock()
        # print((end_time - star_time) / (len(x_data) - train_len))

        #################################### test ####################################

    def write_row_to_csv(self, data, file_name):
        with open(file_name, 'a+', newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerow(data)


if __name__ == "__main__":
    predict_tm_model = PredictTM()
    predict_tm_model.train()

    # np_list = []
    # for i in range(54):
    #     if i % 3 == 0:
    #         np_list.append(np.array([i-3, i-2, i-1]))
    # np_list = np.array(np_list)
    # print(np_list)
    # print(np_list.shape)
    # print("-----------------------")
    # np_list = np_list.reshape(3, 2, 3, 3)
    # print(np_list)
    # print("-----------------------")
    # np_list = np_list.reshape(6, 1, 3, 3)
    # print(np_list)
    # print(np_list.shape)