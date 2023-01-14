import torch
import torch.nn as nn
import time
from torch import optim
from torch.utils.data import DataLoader
from Cards import Cards
from MyResNet18 import Resnet18


def main():  # dataset_1 dataset-resized
    train_data = Cards(train=True,mode = 'train\n')
    test_data = Cards(train=False,mode = 'test\n')
    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    model = Resnet18()
    print(model)
    device = torch.device('cuda:0')
    model = model.to(device)
    criteon = nn.CrossEntropyLoss().to(device)#交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.01)#优化器
    best_val_acc = 0
    for epoch in range(50):
        model.train()
        since = time.time()
        for x, label in train_loader:
            x, label = x.to(device), label.to(device)#传给GPU
            logits = model(x)#返回值
            loss = criteon(logits, label)#计算损失
            optimizer.zero_grad()#梯度清零
            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch, '训练集 loss: ', loss.item())#item是转成数字
        model.eval()
        with torch.no_grad():
            # 测试集
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                loss = criteon(logits, label)
                pred = logits.argmax(dim=1)#获得每行最大值列号
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
            val_acc = total_correct / total_num
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "./model_parameter.pkl")
            time_elapsed = time.time() - since
            print('Epoch: ', epoch, '测试集 loss: ', loss.item())#item是转成数字
            print('Epoch: ',epoch,' Training complete in {:.0f}min {:.0f}seconds'.format(time_elapsed // 60, time_elapsed % 60))
            timeCost = 'Training time {:.0f}min {:.0f}seconds'.format(time_elapsed // 60, time_elapsed % 60)
            print('Epoch: ',epoch, ' test acc: ', val_acc)
    print('The best acc is '+ str(best_val_acc))

if __name__ == '__main__':
    main()