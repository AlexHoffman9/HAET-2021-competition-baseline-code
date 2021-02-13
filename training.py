'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import datetime

from models import ResNet18
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils import progress_bar

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(args, net, testloader, criterion, device):
    network_save_file = args.net_sav
    checkpoint = torch.load(network_save_file)
    net.load_state_dict(checkpoint['net'])
    training_acc = checkpoint['acc']
    training_epoch = checkpoint['epoch']

    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
  
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # results_file = os.path.join(os.path.dirname(args.net_sav), 'results.csv')
    # if os.path.exists(results_file):
    #     append_write = 'a' # append if already exists
    # else:
    #     append_write = 'w' # make a new file if not
    # with open(results_file, append_write) as writer:
    #     writer.write(args.net_sav+','+ str(100.0*correct/total)+'\n')
    return 100.*correct/total


# Training
def train(args, net, trainloader, optimizer, device, criterion, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.net_sav)
    return train_loss

def main():
    start_time=time.time()
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--net_sav', default='./checkpoint/ckpt.pth', type=str, help='network save file')
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    # setup tensorboard
    dt = datetime.datetime.now()
    dt_string = dt.strftime("%m-%d-(%H:%M)")
    folder_str = dt_string + 'lr' + str(args.lr) + 'batch'+str(args.batch_size)
    writer_dir = os.path.join(os.path.dirname(args.net_sav), folder_str)
    tb_writer = SummaryWriter(log_dir=writer_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # net = Net()#Define your network here
    net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False, num_classes=10) # added num_classes=10 so we can use torch impl
    # net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False, num_classes=10) # added num_classes=10 so we can use torch impl
    # net = ResNet18()
    net = net.to(device)
    # for name,param in net.named_parameters():
    #     if 'linear' in name:
    #         print(name, param.shape)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    start_epoch=0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    epoch=0
    while time.time() - start_time < 1.4*10*60:
        print((time.time()-start_time)/60, ' mins elapsed')
        tr_loss = train(args, net, trainloader, optimizer, device, criterion, epoch)
        test_acc = test(args, net, testloader, criterion, device) # wastes training time, but useful for experimentation purporses
        tb_writer.add_scalar('tr_loss', tr_loss, epoch)
        tb_writer.add_scalar('test_acc', test_acc, epoch)
        scheduler.step()
        epoch+=1
    # acc = test(args, net, testloader, criterion, device)
    tb_writer.close()


if __name__=='__main__':
    main()







