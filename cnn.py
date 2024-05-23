import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Defining a simple CNN for MNIST and FMNIST
class CNN(nn.Module):
    def __init__(self, input_size):
        chn, w, _ = input_size  # assuming a square input
        super().__init__()
        self.conv1 = nn.Conv2d(chn, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        d = ((w-4)//2-4)//2
        self.fc1 = nn.Linear(16 * d * d, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNNSoftmax(CNN):
    def forward(self, x):
        return F.softmax(super().forward(x), dim=1)
   
def train(train_loader, img_size, m_pth):
    print(f'Classifier missing at {m_pth}, ')
    print('Training one now, might take couple of minutes')
    m = CNN(img_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)

    for _ in tqdm(range(30), total=30):  
        running_loss = 0.0
        for _, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()   
            outputs = m(inputs.cuda())   
            loss = criterion(outputs, labels.cuda())
            loss.backward(), optimizer.step()
            running_loss += loss.item()  
        # print(f'[Epoch: {epoch + 1}] loss: {running_loss / i:.3f}')
    print('Finished Training')
    torch.save(m.state_dict(), m_pth)

def get_acc(m, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            preds = m(imgs.cuda())
            _, preds = torch.max(preds, 1)
            total += len(lbls)
            correct += (preds == lbls.cuda()).sum().item()
    print('Accuracy: [{:.2f}]'.format(100 * correct/total))