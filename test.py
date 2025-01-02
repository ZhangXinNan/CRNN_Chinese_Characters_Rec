import os

import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import editdistance
import copy

 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str,
                        default=r'C:\data_public\OCR\CRNN_Chinese_Characters_Rec\images',
                        help='the path to your image')
    parser.add_argument('--file_list', default='lib/dataset/txt/test.txt')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    # print(type(preds), preds.shape, preds)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # print(type(preds_size), preds_size.shape, preds_size)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print('results: {0}'.format(sim_pred))
    return sim_pred


if __name__ == '__main__':

    config, args = parse_arg()
    # char_list = [c for c in config.DATASET.ALPHABETS]
    # char_list.sort()
    with open(config.DATASET.CHAR_FILE, 'rb') as file:
        char_list = [char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())]
    print([(i, c) for i, c in enumerate(char_list)])
    # print(config.DATASET.ALPHABETS)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    # checkpoint = torch.load(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # print(converter.alphabet)
    # print(converter.dict)
    started = time.time()
    # for filename in os.listdir(args.image_path):
    with open(args.file_list, 'r') as fi:
        file_list = [s.strip().split(' ') for s in fi]
        print(len(file_list))

    box_num = 0
    box_num_right = 0
    char_num = 0
    char_num_right = 0
    char_map = {}
    for data in file_list:
        # print(data)
        filename = data[0]
        # transcription = ''.join([config.DATASET.ALPHABETS[int(i)] for i in data[1:]])
        # transcription = ''.join([converter.alphabet[int(i)] for i in data[1:]])
        transcription = ''.join([char_list[int(i)] for i in data[1:]])
        image_path = os.path.join(args.image_path, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # print(image_path, img.shape)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pred = recognition(config, img, model, converter, device)
        print(box_num, filename, img.shape)
        print(transcription)
        print(pred)
        d = editdistance.eval(transcription, pred)
        box_num += 1
        char_num += len(transcription)
        if transcription == pred:
            box_num_right += 1
            char_num_right += len(transcription)
        else:
            char_num_right += max(len(transcription), len(pred)) - d
        for c in transcription:
            if c not in char_map:
                char_map[c] = [0, 0]
            char_map[c][0] += 1
            if c in pred:
                char_map[c][1] += 1
        # break
        if box_num > 1000:
            break

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
    print(char_map)
    print("文本行准确率：", box_num, box_num_right, box_num_right / box_num)
    print("字符准确率：", char_num, char_num_right, char_num_right / char_num)

