import csv
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import *
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
import math
import time
BATCH_SIZE = 50

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        #self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Tanh()
        #self.sigmoid = nn.ReLU()
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, _ = self.rnn(x, None)  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])  # return the last value
        out = self.sigmoid(out)
        return out

class PridictTM():
    def __init__(self, file_name, k, input_size, hidden_size, num_layers, epoch, LR):
        # super(PridictTM, self).__init__()
        self.file_name = file_name
        self.k = k
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.LR = LR
        self.input_size = input_size
        self.rnn = RNN(input_size, hidden_size, num_layers)
        self.rnn.cuda()
        print(self.rnn)

    def read_data(self, file_name):
        df = pd.read_csv(file_name)
        del df["timestamp"]
        data_list = df.values
        # print(data_list)
        # print(data_list.shape)
        # print(type(data_list))

        max_list = np.max(data_list, axis=0)
        min_list = np.min(data_list, axis=0)
        # print(len(max_list))
        # print(len(min_list))

        # OD pair, when O = D, max = min = 0, so data_list will have some nan value
        # change these values to 0
        data_list = (data_list - min_list) / (max_list - min_list)
        data_list[np.isnan(data_list)] = 0

        return data_list, min_list, max_list


    # generate normalized time series data
    # list of ([x1, x2, ..., xk], [xk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data, k):
        x_data = []
        y_data = []
        print("data.shape: ", data.shape)
        length = data.shape[0]
        # print(length)
        for i in range(length - k):
            x = data[i:i+k, :]
            y = data[i+k, :]
            x_data.append(x)
            y_data.append(y)
        x_data = torch.from_numpy(np.array(x_data)).float()
        y_data = torch.from_numpy(np.array(y_data)).float()

        return x_data, y_data

    # generate batch data
    def generate_batch_loader(self, x_data, y_data):
        torch_dataset = Data.TensorDataset(x_data, y_data)
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=True,               # random order data
            num_workers=2,              # multiple threading to read data
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
                if not TM[i][j] == 0.0:
                    # temp = str(i + 1) + ' ' + str(j + 1) + ' ' + str(TM[i][j]) + "\n"
                    temp = str(TM[i][j]) + ","
                    f.write(temp)
        f.close()
    '''

    def save_TM(self, TM, file_name):
        row, column = TM.shape
        temp=[]
        for i in range(row):
            for j in range(column):
                temp.append(TM[i][j])
        self.write_row_to_csv(temp, file_name)



    def train(self):

        data, min_list, max_list = self.read_data(self.file_name)
        x_data, y_data = self.generate_series(data, self.k)
        print("x_data.shape:", x_data.shape)
        print("y_data.shape:", y_data.shape)
        train_len = int(int(len(x_data) * 0.8) / 50) * 50#69100
        print("Training length: ", train_len)


        data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])

        optimizer = torch.optim.Adagrad(self.rnn.parameters(), lr=self.LR)
        #optimizer = torch.optim. Adam (self.rnn.parameters(), lr=self.LR)
        loss_func = nn.MSELoss()
        model_name = "F:/Facebook/GRU/GRU_input=" + str(self.input_size) + "_hidden=" + str(self.hidden_size) + "_k=" + str(self.k) + ".pkl"


        star_time = time.clock()

        ################################## train ###############################
        for e in range(self.epoch):
            # print("Epoch: ", e)
            result = []
            for step, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                # print("batch_x.shape:", batch_x.shape)
                # print("batch_y.shape:", batch_y.shape)
                prediction = self.rnn.forward(batch_x).cuda()
                # print("prediction.shape:", prediction.shape)
                # print(prediction)
                # print(prediction[-1].data.numpy())
                result.append(prediction[-1].cpu().data.numpy())
                loss_train = []
                loss1 = loss_func(prediction, batch_y)
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()
                loss1 = loss1.cpu().data.numpy()
                loss_train.append(loss1)
                #self.write_row_to_csv(loss_train, "F:/Facebook/GRU/train_loss_GRU.csv")
                print("Epoch =", e, ", step =", step, ", loss:", loss1)
        end_time = time.clock()
        print('train time ------------------------------------------', end_time - star_time)
        # save model
        #torch.save(self.rnn.state_dict(), model_name)

        ################################## train ###############################



        ################################## test ################################
        print("----------------------------test-----------------------\n")
        # load model

        self.rnn.load_state_dict(torch.load(model_name))
        result = []
        count = 0


        for i in range(train_len, len(x_data)):
            star_time=time.clock()
            test_x = x_data[i].reshape(1, self.k, self.input_size).cuda()
            test_y = y_data[i].reshape(1, self.input_size).cuda()
            # print("test_y.shape:", test_y.shape)
            prediction = self.rnn.forward(test_x).cuda()
            # print("prediction.shape:", prediction.shape)
            test_loss = []
            loss2 = loss_func(prediction, test_y)
            loss2 = loss2.cpu().data.numpy()
            test_loss.append(loss2)
            #self.write_row_to_csv(test_loss, "F:/Facebook/GRU/test_loss_GRU.csv")
            print("Loss for test data " + str(i - train_len + 1) + " is:", loss2)
            # break
            # save result
            # data = []
            # data.append(str(i - train_len + 1))
            # data.append(loss.cpu().data.numpy())
            # self.write_row_to_csv(data, "loss_GRU.csv")
            inverse_prediction, inverse_y = self.inverse_normalization(prediction.cpu().data.numpy()[0],
                                                                       test_y.cpu().data.numpy()[0], max_list, min_list)
            inverse_prediction = inverse_prediction.reshape(int(math.sqrt(self.input_size)),
                                                            int(math.sqrt(self.input_size)))
            inverse_y = inverse_y.reshape(int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))

            inverse_predictiont = torch.from_numpy(inverse_prediction)

            inverse_yt = torch.from_numpy(inverse_y)
            loss = loss_func(inverse_predictiont, inverse_yt)
            loss = loss.sqrt()
            end_time=time.clock()
            print('test time ------------------------------------------', end_time - star_time)
            # loss = loss_func(prediction, test_y)
            # print("Loss for test data " + str(i - train_len + 1) + " is:", loss)
            # break
            # save result
            data = []

            # data.append(str(i - train_len + 1))
            data.append(loss.cpu().data.numpy())
            #self.write_row_to_csv(data, "F:/Facebook/GRU/RMSE_GRU.csv")
            '''
            # inverse normalization
            inverse_prediction, inverse_y = self.inverse_normalization(prediction.cpu().data.numpy()[0], test_y.cpu().data.numpy()[0], max_list, min_list)
            inverse_prediction = inverse_prediction.reshape(int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))
            inverse_y = inverse_y.reshape(int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))
            '''
            path = "F:/Facebook/GRU/result.csv"
            #path = "F:/Facebook/gru/" + str(i - train_len + 1) + ".csv"
            #self.save_TM(inverse_prediction, path)



        ################################## test ################################



    def write_row_to_csv(self, data, file_name):
        with open(file_name, 'a+', newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerow(data)


if __name__ == "__main__":
    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "F:/Facebook/final.csv"
    k = 10
    input_size = 81
    hidden_size = 100
    num_layers = 1#1
    epoch = 30
    LR = 0.055

    predict_tm_model = PridictTM(file_name, k, input_size, hidden_size, num_layers, epoch, LR)
    predict_tm_model.train()




