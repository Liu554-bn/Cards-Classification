from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from pathlib import Path

class Cards(Dataset):
    def __init__(self, train=True, mode='train\n'):

        data_dir = './archive'  # 相对路径就能运行，好家伙
        fname = data_dir+'/cards.csv'
        fname = Path(fname)
        with open(fname, 'r') as f:
            lines = f.readlines()[1:]
        self.imgs = []

        for i in lines:
            if i.split(',')[4] == mode:
                self.imgs.append(i.split(','))
        for line in self.imgs:
            if line[1] == 'train/ace of clubs/output':
                print('Pre handle the data')
                self.imgs.remove(line)
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if not train:
            # 验证集或测试集
            self.transform = T.Compose([
                T.Resize(224),  # 最短边224像素
                T.CenterCrop(224),  # 中心裁剪
                T.ToTensor(),  # 转成tensor
                normalize  # 归一化
            ])
        else:
            # 训练集，做数据增广
            self.transform = T.Compose([
                T.Resize(224),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),  # 按概率水平翻转
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_line = self.imgs[index]
        img_path = './archive/' + img_line[1]
        label = int(img_line[0])
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    train_data = Cards(train=True,mode = 'train\n')
    test_data = Cards(train=False,mode = 'test\n')