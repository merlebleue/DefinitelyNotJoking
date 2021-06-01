# Milestone 2 : Classification - Tsunami induced building collapse detection

**Team :** DefinitelyNotJoking

**Group members :** 
- Denise Vandeuren (_310373_)
- Neroli Soso (_284591_)
- Julien Ars (_314545_)

**Best submission ID :** 142187

For this milestone, we had to create a model capable of determining whether a building collapsed after a tsunami based on aerial imagery.

We developed a neural network which achieved an accuracy of $93.6 \%$

## Our Code :

*Our code can be found in the notebook Milestone_2.ipynb.* It countains a fully functional training


**What we tried and didn't work:**

For this MileStone we decided to take the convolutional Neural Network approach. Before this we tried to experiment with simpler
multi-layered Neural Networks, however it ended up having very large validation losses and presumably due to a larger amount of
parameters (nodes) ended up overfitting very fast.

One Non convolutional Network example we tried:

-----------------------------------------------------------------

class NoConv(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # Input tensor is of shape [batch_size, 6, 160, 160]
        self.fc1 = nn.Linear(6 * 160 * 160, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x= self.fc5(x)

        return x

-----------------------------------------------------------------

We also tried to implement a five layer ConvNet:  (but the val accuracy never went down even after 20 epoch for any optimizer)

class FiveConvNet(nn.Module):
    """A basic Model with 5 convolutions"""
  
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(6, 20,  20,padding=10) #160*160
        self.conv2 = nn.Conv2d(20, 32, 20, padding=10) #160*160
        self.conv3 = nn.Conv2d(32,16, 10, padding=5) #80*80
        self.conv4 = nn.Conv2d(16,10, 10, padding=5) #40*40
        self.conv5 = nn.Conv2d(10, 10, 10) #20*20
        self.fc6 = nn.Linear(1440, 400) #12*12*10
        self.fc7 = nn.Linear(400, 100)
        self.fc8 = nn.Linear(100, 10)
        self.fc9 = nn.Linear(10,1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)
        x = F.sigmoid(self.conv5(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.leaky_relu(self.fc8(x))
        x = F.sigmoid(self.fc9(x))
        return x








