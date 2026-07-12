import torch.nn as nn
import torch.nn.functional as F # stateless functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #input layer
        self.conv1 = nn.Conv2d(3, 23, 5) # in_channels, out_channels, kernel_size
        # conv1 (a weight tensor / 4d array) contains 6 filters with shape 5*5*3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(23, 16, 5)

        #hidden layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)

        #output layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #introduces nonlinearity for complex patterns
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #logits, no ReLU
        return x

net_cnn = Net().to(device)

PATH = './cifar_net_cnn.pt'
torch.save(net_cnn.state_dict(), PATH)
