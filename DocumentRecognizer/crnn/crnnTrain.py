from __future__ import print_function
from torch.utils.data import DataLoader
import argparse
import random
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import os
from DocumentRecognizer.crnn.crnnNets import *
import Common.utils as utils
from DocumentRecognizer.crnn.crnnDataSet import ChineseTextLoader
import time
import DocumentRecognizer.crnn.converter as Converter
from Common.utils import alphabetChinese
import cv2


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(crnn, train_loader, criterion, iteration):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    for i_batch, (image, index) in enumerate(train_loader):
        image = image.to(device) # type:torch.Tensor
        label = utils.get_batch_label(dataset, index)
        preds = crnn(image)

        batch_size = image.size(0)
        index = np.array(index.data.numpy())
        text, length = converter.encode(label)
        text = text.to(device)
        length = length.to(device)
        preds = preds.to(torch.float64)
        preds = preds.to(device)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
        # print(preds.shape, text.shape, preds_size.shape, length.shape)
        # torch.Size([41, 16, 6736]) torch.Size([160]) torch.Size([16]) torch.Size([16])
        criterion = criterion.to(device)
        cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        if (i_batch + 1) % 10 == 0:
            print('[%d/%d][%d/%d] Cost: %f Loss: %f' %
                  (iteration, args.epoch, i_batch, len(train_loader), cost.item(), loss_avg.val()))
            loss_avg.reset()


def main(crnn, train_loader, val_loader, criterion, optimizer):
    crnn = crnn.to(device)
    criterion = criterion.to(device)
    Iteration = 0
    while Iteration < args.epoch:
        train(crnn, train_loader, criterion, Iteration)
        Iteration += 1
        t = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        torch.save(crnn.state_dict(), 'trained_models/crnn_' + t)


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0  # replace all nan/inf in gradients to zero


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--batch_size', type=int, default='1', help='batch size')
    parser.add_argument('--model_path', type=str, default=None, help="pretrained model path")
    parser.add_argument('--epoch', type=int, default=10, help='epoch for train')
    args = parser.parse_args()
    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # store model path
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    # read train set
    converter = Converter.strLabelConverter(alphabetChinese)
    dataset = ChineseTextLoader(converter, "C:/Users/xiangpu/Downloads/images/",
                                "C:/Users/xiangpu/Downloads/360label/360_test.txt",
                                alphabetChinese, (280, 32))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    nclass = len(converter.dict) + 1
    nc = 1
    criterion = torch.nn.CTCLoss(reduction='sum')

    # cnn and rnn
    crnn = CRNN(32, nc, nclass, 256)

    if args.model_path is not None:
        print('using pretrained model!!')
        crnn.load_weights(args.model_path)
    else:
        print('no pretrined model assigned!!')

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    optimizer = optim.RMSprop(crnn.parameters(), lr=0.0001)

    crnn.register_backward_hook(backward_hook)
    main(crnn, train_loader, None, criterion, optimizer)
