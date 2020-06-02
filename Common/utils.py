import cv2
import numpy as np
import torch
import codecs
import math


def cal_IoU(cy1, h1, cy2, h2):
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    y_top_min = min(y_top1, y_top2)
    y_bottom_max = max(y_bottom1, y_bottom2)
    union = y_bottom_max - y_top_min + 1
    intersection = h1 + h2 - union
    iou = float(intersection) / float(union)
    if iou < 0:
        return 0.0
    else:
        return iou


def calcY(x, param):
    a = param[0]
    b = param[1]
    c = param[2]
    if b != 0:
        return int(-(c + a * x) / b)
    else:
        return 0


def calcLine(box):
    x1 = box[0]
    x2 = box[2]
    x3 = box[4]
    x4 = box[6]
    y1 = box[1]
    y2 = box[3]
    y3 = box[5]
    y4 = box[7]

    # l12
    a12 = y2 - y1
    b12 = x1 - x2
    c12 = x2 * y1 - x1 * y2
    line12 = [a12, b12, c12]

    # l34
    a34 = y4 - y3
    b34 = x3 - x4
    c34 = x4 * y3 - x3 * y4
    line34 = [a34, b34, c34]

    # l14
    a14 = y4 - y1
    b14 = x1 - x4
    c14 = x4 * y1 - x1 * y4
    line14 = [a14, b14, c14]

    # l23
    a23 = y3 - y2
    b23 = x2 - x3
    c23 = x3 * y2 - x2 * y3
    line23 = [a23, b23, c23]

    return [line14, line23, line12, line34]


def sortCoords(box):
    coords = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]]
    coords_x = [box[0], box[2], box[4], box[6]]
    coords_x.sort()
    coords_left = []
    coords_right = []
    for i in range(4):
        if coords[i][0] == coords_x[0] or coords[i][0] == coords_x[1]:
            coords_left.append(coords[i])
        else:
            coords_right.append(coords[i])
    new_box = []
    if coords_left[0][1] < coords_left[1][1]:
        new_box += coords_left[0]
        new_box += coords_left[1]
    else:
        new_box += coords_left[1]
        new_box += coords_left[0]
    if coords_right[0][1] > coords_right[1][1]:
        new_box += coords_right[0]
        new_box += coords_right[1]
    else:
        new_box += coords_right[1]
        new_box += coords_right[0]

    return new_box


def cal_y_top_and_bottom(raw_img, position_pair, box):
    y_top = []
    y_bottom = []
    box = sortCoords(box)
    lines = calcLine(box)
    whichline = []
    for k in range(len(position_pair)):
        box_x = [box[0], box[2], box[4], box[6]]
        box_x.sort()
        box_left_left = box_x[0]
        box_left_right = box_x[1]
        box_right_left = box_x[2]
        box_right_right = box_x[3]
        anchor_middle_x = position_pair[k][0] + 7.5

        if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
            anchor_y_top = calcY(anchor_middle_x, lines[0])
            anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(1)
            continue

        elif anchor_middle_x > box_left_left and anchor_middle_x < box_left_right:  # 位于左边界上的
            if lines[2][1] == 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                continue
            k_l12 = -(lines[2][0] / lines[2][1])
            anchor_y_top = calcY(anchor_middle_x, lines[0])
            anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            if k_l12 > 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[2])
            else:
                anchor_y_top = calcY(anchor_middle_x, lines[2])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(2)
            continue

        elif anchor_middle_x <= box_left_left:
            anchor_middle_x = position_pair[k][1]  # 如果左边界外的，则用anchor右边缘替代中线
            if anchor_middle_x > box_right_right:  # 如果box很小时，anchor右边界超出box右边界，此时将anchor_middle_x替换为box_left_right
                anchor_middle_x = box_left_right
            if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(3)
                continue
            else:
                if lines[2][1] == 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                    y_top.append(anchor_y_top)
                    y_bottom.append(anchor_y_bottom)
                    continue
                k_l12 = -(lines[2][0] / lines[2][1])
                if k_l12 > 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[2])
                else:
                    anchor_y_top = calcY(anchor_middle_x, lines[2])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(4)
                continue

        elif anchor_middle_x > box_right_left and anchor_middle_x < box_right_right:  # 位于右边界上的
            if lines[3][1] == 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                continue
            k_l34 = -(lines[3][0] / lines[3][1])
            if k_l34 > 0:
                anchor_y_top = calcY(anchor_middle_x, lines[3])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            else:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[3])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(5)
            continue

        elif anchor_middle_x >= box_right_right:
            anchor_middle_x = position_pair[k][0]  # anchor左边界替代中线
            if anchor_middle_x < box_left_left:  # 如果box很小时，anchor左边界超出box左边界，此时将anchor_middle_x替换为box_right_left
                anchor_middle_x = box_right_left
            if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(6)
                continue
            else:
                if lines[3][1] == 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                    y_top.append(anchor_y_top)
                    y_bottom.append(anchor_y_bottom)
                    continue
                k_l34 = -(lines[3][0] / lines[3][1])
                if k_l34 > 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[3])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                else:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[3])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(7)
                continue

    # print(y_top)
    # print(y_bottom)
    # print(whichline)
    return y_top, y_bottom


def draw_box_h_and_c(img, position, cy, h, anchor_width=16, color=(0, 255, 0), thickness=1):
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    pt = [x_left, y_top, x_right, y_bottom]
    return draw_box_2pt(img, pt, color=color, thickness=thickness)


def draw_box_4pt(img, pt, color=(0, 255, 0), thickness=2):
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(8)]
    img = cv2.line(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness)
    img = cv2.line(img, (pt[2], pt[3]), (pt[4], pt[5]), color, thickness)
    img = cv2.line(img, (pt[4], pt[5]), (pt[6], pt[7]), color, thickness)
    img = cv2.line(img, (pt[6], pt[7]), (pt[0], pt[1]), color, thickness)
    return img


def generate_gt_anchor(img, box, anchor_width=16, draw_img_gt=None):
    """
    calsulate ground truth fine-scale box
    :param img: input image
    :param box: ground truth box (4 point)
    :param anchor_width:
    :return: tuple (position, h, cy)
    """
    if not isinstance(box[0], float):
        box = [float(box[i]) for i in range(len(box))]

    result = []
    left_anchor_num = int(
        math.floor(max(min(box[0], box[6]), 0) / anchor_width))  # the left side anchor of the text box, downwards
    right_anchor_num = int(math.ceil(
        min(max(box[2], box[4]), img.shape[1]) / anchor_width))  # the right side anchor of the text box, upwards

    # handle extreme case, the right side anchor may exceed the image width
    if right_anchor_num * 16 + 15 > img.shape[1]:
        right_anchor_num -= 1

    # combine the left-side and the right-side x_coordinate of a text anchor into one pair
    position_pair = [(i * anchor_width, (i + 1) * anchor_width - 1) for i in range(left_anchor_num, right_anchor_num)]
    y_top, y_bottom = cal_y_top_and_bottom(img, position_pair, box)

    # print("image shape: %s, pair_num: %s, top_num:%s, bot_num:%s" % (img.shape, len(position_pair), len(y_top), len(y_bottom)))

    for i in range(len(position_pair)):
        position = int(position_pair[i][0] / anchor_width)  # the index of anchor box
        h = y_bottom[i] - y_top[i] + 1  # the height of anchor box
        cy = (float(y_bottom[i]) + float(y_top[i])) / 2.0  # the center point of anchor box
        result.append((position, cy, h))
        draw_img_gt = draw_box_h_and_c(draw_img_gt, position, cy, h)
    draw_img_gt = draw_box_4pt(draw_img_gt, box, color=(0, 0, 255), thickness=1)
    return result, draw_img_gt


def cal_y(cy, h):
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return y_top, y_bottom


def valid_anchor(cy, h, height):
    top, bottom = cal_y(cy, h)
    if top < 0:
        return False
    if bottom > (height * 16 - 1):
        return False
    return True


def tag_anchor(gt_anchor, cnn_output, gt_box):
    anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]  # from 11 to 273, divide 0.7 each time
    # whole image h and w
    height = cnn_output.shape[2]
    width = cnn_output.shape[3]
    positive = []
    negative = []
    vertical_reg = []
    side_refinement_reg = []
    x_left_side = min(gt_box[0], gt_box[6])
    x_right_side = max(gt_box[2], gt_box[4])
    left_side = False
    right_side = False
    for a in gt_anchor:

        if a[0] >= int(width - 1):
            continue

        if x_left_side in range(a[0] * 16, (a[0] + 1) * 16):
            left_side = True
        else:
            left_side = False

        if x_right_side in range(a[0] * 16, (a[0] + 1) * 16):
            right_side = True
        else:
            right_side = False

        iou = np.zeros((height, len(anchor_height)))
        temp_positive = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height[j], height):
                    continue
                iou[i][j] = cal_IoU((float(i) * 16.0 + 7.5), anchor_height[j], a[1], a[2])

                if iou[i][j] > 0.7:
                    temp_positive.append((a[0], i, j, iou[i][j]))
                    if left_side:
                        o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))
                    if right_side:
                        o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))

                if iou[i][j] < 0.5:
                    negative.append((a[0], i, j, iou[i][j]))

                if iou[i][j] > 0.5:
                    vc = (a[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height[j])
                    vh = math.log10(float(a[2]) / float(anchor_height[j]))
                    vertical_reg.append((a[0], i, j, vc, vh, iou[i][j]))

        if len(temp_positive) == 0:
            max_position = np.where(iou == np.max(iou))
            temp_positive.append((a[0], max_position[0][0], max_position[1][0], np.max(iou)))

            if left_side:
                o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))
            if right_side:
                o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))

            if np.max(iou) <= 0.5:
                vc = (a[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height[max_position[1][0]])
                vh = math.log10(float(a[2]) / float(anchor_height[max_position[1][0]]))
                vertical_reg.append((a[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))
        positive += temp_positive
    return positive, negative, vertical_reg, side_refinement_reg


def scale_img(img, gt, shortest_side=600):
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side) / float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))
    h_scale = float(img.shape[0]) / float(height)
    w_scale = float(img.shape[1]) / float(width)
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    return img, scale_gt


def read_gt_file(path, have_BOM=False):
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        if have_BOM:
            box = [int(round(float(pt[i]))) for i in range(8)]
        else:
            box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result


def init_weight(net):
    for i in range(len(net.rnn.blstm.lstm.all_weights)):
        for j in range(len(net.rnn.blstm.lstm.all_weights[0])):
            torch.nn.init.normal_(net.rnn.blstm.lstm.all_weights[i][j], std=0.01)

    torch.nn.init.normal_(net.FC.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.FC.bias, val=0)

    torch.nn.init.normal_(net.vertical_coordinate.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.vertical_coordinate.bias, val=0)

    torch.nn.init.normal_(net.score.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.score.bias, val=0)

    torch.nn.init.normal_(net.side_refinement.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.side_refinement.bias, val=0)


def draw_box_2pt(img, pt, color=(0, 255, 0), thickness=1):
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(4)]
    img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness=thickness)
    return img


def draw_ploy_4pt(img, pt, color=(0, 255, 255), thickness=1):
    pts = np.array([[pt[0], pt[1]], [pt[2], pt[3]], [pt[4], pt[5]], [pt[6], pt[7]]], np.int32)
    print(pts)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img, [pts], True, color, thickness)


def trans_to_2pt(position, cy, h, anchor_width=16):
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return [x_left, y_top, x_right, y_bottom]
