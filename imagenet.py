import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
NUM_EPOCH = 10

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        #backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

    ## Define the training dataloader
def train():
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    def subsetting(dt,m):
        indc=[]
        count=0
        cls = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
        for kl in dt:
            datas,kn= kl
            cls[kn]=cls.get(kn)+1
            if cls[kn]<=m:
                indc.append(count)
            count = count+1
        fdt = torch.utils.data.Subset(dt,indc)
        return fdt
    
    traindata = subsetting(trainset,50)
    testdata = subsetting(testset,50)
    
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=4, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=4, shuffle=True, num_workers=4)

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model.cuda()

    epochloss = {}
    epochmodel = {}
    epochoptimizer = {}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),lr=0.001, momentum=0.9)
    
    def testingepoch(x,m,o):
        epoch_loss = 0.0
        for j, ttdata in enumerate(testloader, 0):
            # get the inputs
            ttinputs, ttlabels = ttdata
            if torch.cuda.is_available():
                ttinputs = ttinputs.cuda()
                ttlabels = ttlabels.cuda()
            ttoutputs = m(ttinputs)
            loss = criterion(ttoutputs, ttlabels)
            epoch_loss += loss.item()
            if j % 20 == 19:    # print every 20 mini-batches
                print('VALIDATION STATUS - Running epoch %d, batch %d' %(x + 1, j + 21))
        print("\n")
        epoch_loss = epoch_loss/10000
        epochloss[x] = epoch_loss
        epochmodel[x] = m.state_dict()
        epochoptimizer[x] = o.state_dict()


    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        print('\nTraining model for epoch %d' %(epoch+1))
        for i, trdata in enumerate(trainloader, 0):
            # get the inputs
            trinputs, trlabels = trdata
            if torch.cuda.is_available():
                trinputs = trinputs.cuda()
                trlabels = trlabels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            troutputs = model(trinputs)
            trloss = criterion(troutputs, trlabels)
            trloss.backward()
            optimizer.step()
            if i % 20 == 19:    # print status every 20 mini-batches
                print('STATUS - Running epoch %d, batch %d' %(epoch + 1, i + 21))
        if (epoch+1) % 1 == 0:    # test on validation set for every 3 epochs
            print('\nRunning model on validation set after %d epoch(s)' %(epoch+1))
            testingepoch(epoch+1,model,optimizer)
                
    print('Finished Training.')
    print(epochloss)
    #print(epochmodel)
    bestloss = min(epochloss, key=epochloss.get)
    
    plt.figure(figsize=(8,8))
    plt.plot(epochloss.keys(), epochloss.values())
    plt.ylabel('Entropy Loss')
    plt.title('Loss on validation set for every 3 epochs')
    plt.xlabel('Epochs')
    plt.show()
    
    torch.save({'best_model': epochmodel[bestloss],'best_optimizer': epochoptimizer[bestloss],'best_epoch': bestloss,'best_loss': epochloss[bestloss]}, 'bestmodel_n.pth')
    print('Model trained in %d epochs with validation error of %.3f has beesn saved' % (bestloss + 1, epochloss[bestloss]))





if __name__ == '__main__':
    train()
