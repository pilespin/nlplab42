import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



class BowModel(nn.Module):
    def __init__(self, emb_tensor):
        super(BowModel, self).__init__()
        n_embedding, dim = emb_tensor.size()
        self.embedding = nn.Embedding(n_embedding, dim, padding_idx=0)
        self.embedding.weight = Parameter(emb_tensor, requires_grad=False)

        self.out = nn.Linear(dim, 25)
        self.out1 = nn.Linear(25, 25)
        self.out2 = nn.Linear(25, 2)

    def forward(self, input):
        '''
        input is a [batch_size, sentence_length] tensor with a list of token IDs
        '''
        embedded = self.embedding(input)
        # Here we take into account only the first word of the sentence
        # You should change it, e.g. by taking the average of the words of the sentence
        bow = embedded[:, 0]

        bow = embedded.mean(dim=1)
        bow = F.tanh(self.out(bow))
        bow = F.tanh(self.out1(bow))
        bow = F.tanh(self.out2(bow))
        # self.lstm = nn.LSTM(300, 300)
        # bow = self.out(bow)
        # rnn = nn.RNN(10, 20, 2)
        # input = Variable(torch.randn(5, 3, 10))
        # h0 = Variable(torch.randn(2, 3, 20))
        # output, hn = rnn(bow, h0)
        # return F.log_softmax(self.lstm(bow))
        return F.log_softmax(bow)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 2)
#         self.conv2 = nn.Conv2d(6, 12, 2)
#         self.conv3 = nn.Conv2d(12, 18, 2)
#         self.conv4 = nn.MaxPool2d(2)
#         # an affine operation: y = Wx + b
#         # self.fc1 = nn.Linear(450, 300)
#         self.fc2 = nn.Linear(2592, 500)
#         self.fc3 = nn.Dropout(p=0.1)
#         self.fc4 = nn.Linear(500, 27)
#         self.fc5 = nn.Dropout(p=0.1)



#     def forward(self, x):

#         x = F.tanh(self.conv1(x))
#         x = F.tanh(self.conv2(x))
#         x = F.tanh(self.conv3(x))
#         x = F.tanh(self.conv4(x))
#         # x = F.max_pool2d(F.tanh(self.conv3(x)), (2, 2))
        
#         x = self.fc2(x.view(x.size(0), -1))
#         # x = self.fc1(x)
#         # x = F.tanh(self.fc2(x))
#         x = F.tanh(self.fc3(x))
#         x = F.tanh(self.fc4(x))
#         x = F.tanh(self.fc5(x))
        
#         return F.log_softmax(x)
