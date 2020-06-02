import torch.optim as optim
import torch
import cv2
import numpy as np
import os
import DocumentDetector.nets as nets
import Common.utils as utils
import configparser
import time
import logging
import datetime
import copy
import random
import argparse
import socket

DRAW_PREFIX = './anchor_draw'
if socket.gethostname() == 'zxp':
    part3 = '/root/myDataSet/SceneText/part3'
else:
    part3 = '/home/xingpu/myDataSet/part3'
DATASET_LIST = [part3]
MODEL_SAVE_PATH = 'trained_models'


def loop_files(path):
    files = []
    l = os.listdir(path)
    for f in l:
        if f[-3:] == 'jpg':
            files.append(os.path.join(path, f))
    return files


def create_train_val():
    train_image_list = []
    for dataset in DATASET_LIST:
        train_image = loop_files(dataset)
        train_image_list += train_image
    return train_image_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    cf = configparser.ConfigParser()
    cf.read('config')

    log_dir = 'trained_models'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    log_handler = logging.FileHandler(os.path.join(log_dir, log_file_name), 'w')
    log_format = formatter = logging.Formatter('%(asctime)s: %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    gpu_id = cf.get('global', 'gpu_id')
    epoch = cf.getint('global', 'epoch')
    val_batch_size = cf.getint('global', 'val_batch')
    logger.info('Total epoch: {0}'.format(epoch))

    using_cuda = cf.getboolean('global', 'using_cuda')
    display_img_name = cf.getboolean('global', 'display_file_name')
    display_iter = cf.getint('global', 'display_iter')
    val_iter = cf.getint('global', 'val_iter')
    save_iter = cf.getint('global', 'save_iter')

    lr_front = cf.getfloat('parameter', 'lr_front')
    lr_behind = cf.getfloat('parameter', 'lr_behind')
    change_epoch = cf.getint('parameter', 'change_epoch') - 1
    logger.info('Learning rate: {0}, {1}, change epoch: {2}'.format(lr_front, lr_behind, change_epoch + 1))
    print('Using gpu id(available if use cuda): {0}'.format(gpu_id))
    print('Train epoch: {0}'.format(epoch))
    print('Use CUDA: {0}'.format(using_cuda))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    no_grad = [
        'cnn.VGG_16.convolution1_1.weight',
        'cnn.VGG_16.convolution1_1.bias',
        'cnn.VGG_16.convolution1_2.weight',
        'cnn.VGG_16.convolution1_2.bias'
    ]

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    net = nets.CTPN()  # type:torch.nn.Module
    for name, value in net.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
    device = torch.device('cuda' if torch.cuda.is_available() and using_cuda else 'cpu')
    net.load_state_dict(torch.load('trained_models/vgg16.model'))
    if args.model_path is None:
        print("random init weight")
        utils.init_weight(net)
    else:
        print('load pretrained model from path: {}'.format(args.model_path))
        net.load_state_dict(torch.load(args.model_path, map_location=device))
    net = net.to(device)
    net.train()

    criterion = nets.CTPNLoss(using_cuda=using_cuda)

    train_im_list = create_train_val()
    total_iter = len(train_im_list)
    print("total training image num is %s" % len(train_im_list))

    train_loss_list = []
    test_loss_list = []

    for i in range(epoch):
        if i >= change_epoch:
            lr = lr_behind
        else:
            lr = lr_front
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        iteration = 1
        total_loss = 0
        total_cls_loss = 0
        total_v_reg_loss = 0
        total_o_reg_loss = 0
        start_time = time.time()

        random.shuffle(train_im_list)

        for im in train_im_list:
            gt_path = im[:-3] + 'txt'
            if not os.path.exists(gt_path):
                print('Ground truth file of image {0} not exists.'.format(im))
                continue
            gt_txt = utils.read_gt_file(gt_path)
            img = cv2.imread(im)
            if img is None:
                iteration += 1
                continue

            img, gt_txt = utils.scale_img(img, gt_txt)
            tensor_img = img[np.newaxis, :, :, :]
            tensor_img = tensor_img.transpose((0, 3, 1, 2))
            tensor_img = torch.FloatTensor(tensor_img).to(device)

            vertical_pred, score, side_refinement = net(tensor_img)
            del tensor_img

            # transform bbox gt to anchor gt for training
            positive = []
            negative = []
            vertical_reg = []
            side_refinement_reg = []

            visual_img = copy.deepcopy(img)

            try:
                # loop all bbox in one image
                for box in gt_txt:
                    # generate anchors from one bbox
                    gt_anchor, visual_img = utils.generate_gt_anchor(img, box, draw_img_gt=visual_img)
                    positive1, negative1, vertical_reg1, side_refinement_reg1 = utils.tag_anchor(gt_anchor, score, box)
                    positive += positive1
                    negative += negative1
                    vertical_reg += vertical_reg1
                    side_refinement_reg += side_refinement_reg1
            except:
                print("warning: img %s raise error!" % im)
                iteration += 1
                continue

            if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
                iteration += 1
                continue

            # cv2.imshow("labeled img", visual_img)
            # cv2.waitKey(0)
            optimizer.zero_grad()
            loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                               negative, vertical_reg, side_refinement_reg)
            loss.backward()
            optimizer.step()
            iteration += 1
            # save gpu memory by transferring loss to float
            total_loss += float(loss)
            total_cls_loss += float(cls_loss)
            total_v_reg_loss += float(v_reg_loss)
            total_o_reg_loss += float(o_reg_loss)

            if iteration % display_iter == 0:
                end_time = time.time()
                total_time = end_time - start_time
                print(
                    'Epoch: {2}/{3}, Iteration: {0}/{1}, loss: {4}, cls_loss: {5}, v_reg_loss: {6}, o_reg_loss: {7}, {8}'.format(
                        iteration, total_iter, i, epoch, total_loss / display_iter, total_cls_loss / display_iter,
                                                         total_v_reg_loss / display_iter,
                                                         total_o_reg_loss / display_iter, im))

                logger.info('Epoch: {2}/{3}, Iteration: {0}/{1}'.format(iteration, total_iter, i, epoch))
                logger.info('loss: {0}'.format(total_loss / display_iter))
                logger.info('classification loss: {0}'.format(total_cls_loss / display_iter))
                logger.info('vertical regression loss: {0}'.format(total_v_reg_loss / display_iter))
                logger.info('side-refinement regression loss: {0}'.format(total_o_reg_loss / display_iter))

                train_loss_list.append(total_loss)

                total_loss = 0
                total_cls_loss = 0
                total_v_reg_loss = 0
                total_o_reg_loss = 0
                start_time = time.time()

            if iteration % save_iter == 0:
                print('saving training model')
                torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'training_model'))

        t = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        print('Model saved at {0}/ctpn_{1}.model'.format(MODEL_SAVE_PATH, t))
        torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn_{}.model'.format(t)))
        os.remove(os.path.join(MODEL_SAVE_PATH, 'training_model'))
