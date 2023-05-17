import os
import json
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import shutil
import numpy as np
import torch.nn as nn
import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from utils import enet_weighing, median_freq_balancing
import torch.nn as nn
from collections import OrderedDict, Counter
import random
# from utils import add_mask_to_source_multi_classes, add_mask_to_source
from pathlib import Path

from dataset import SegDataset
from metric import SegmentationMetric

def get_train_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_tag", type=str, default='v0')
    parser.add_argument("--net_name", type=str, default='unet')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--train_dir", type=str, default='datasets/beforeBZ fengqin2mm-4mpa-30mm-M001-N001')
    parser.add_argument("--val_dir", type=str, default='datasets/beforeBZ fengqin2mm-4mpa-30mm-M001-N001')
    parser.add_argument("--test_dir", type=str, default='datasets/beforeBZ fengqin2mm-4mpa-30mm-M001-N001')
    parser.add_argument("--demo_dir", type=str, default='datasets/beforeBZ fengqin2mm-4mpa-30mm-M001-N001')
    parser.add_argument("--train_aug", type=str, default=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--mode", type=str, default='demo')
    return parser.parse_args()

def choose_net(name, num_classes):
    if name == 'unet':
        from unet import UNet
        return UNet(n_classes=num_classes)
    else:
        return None
    
def trainval(args):
    net = choose_net(args.net_name, args.num_classes)

    save_dir = './Results/' + args.save_tag + '-' + args.net_name + '-h' + str(args.height) + '-w' + str(args.width)
    print('Save dir is:', save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + '/train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    torch.backends.cudnn.enabled = True

    train_dir = SegDataset(args.train_dir, num_classes=args.num_classes, input_hw=(args.height, args.width), train_aug=args.train_aug)
    print('Length of train_dir:', len(train_dir))
    
    train_dataloader = DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_steps = len(train_dataloader)
    print('Length of train_steps:', train_steps)

    val_dir = SegDataset(args.val_dir, num_classes=args.num_classes, input_hw=(args.height, args.width), train_aug=False)
    val_dataloader = DataLoader(val_dir, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    writer = SummaryWriter(save_dir + '/log')
    if os.path.exists(save_dir + '/log'):
        shutil.rmtree(save_dir + '/log')
    
    if args.epoch is not None:
        epoch = args.epoch
    else:
        epoch = 20000 // train_steps

    iter_cnt = 1
    min_eval_loss = 1000
    net.cuda()
    for epo in range(1, epoch + 1):
        net.train()
        for batch_id, (batch_data, batch_label) in enumerate(train_dataloader):
            batch_data = batch_data.cuda()
            batch_label = batch_label.squeeze(1).cuda()
            output = net(batch_data)
            loss = criterion(output, batch_label)
            iter_loss = loss.item()
            print('Epoch:{} Batch:[{}/{}] Loss:{}'.format(epo, str(batch_id + 1).zfill(3), train_steps, round(iter_loss, 4)))
            writer.add_scalar('Train loss', iter_loss, iter_cnt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_cnt += 1

        net.eval()
        epo_eval_loss = 0
        for batch_id, (batch_data, batch_label) in enumerate(val_dataloader):
            with torch.no_grad():
                batch_data = batch_data.cuda()
                batch_label = batch_label.squeeze(1).cuda()
                output = net(batch_data)
                loss = criterion(output, batch_label)
            epo_eval_loss += loss.item()

        if epo_eval_loss < min_eval_loss:
            min_eval_loss = epo_eval_loss
            save_file = save_dir + '/min_eval_loss.pt'
            torch.save(net.state_dict(), save_file)
            print('-' * 50)
            print('Saved checkpoint as min_eval_loss:', min_eval_loss)
            print('-' * 50)


def test(args):
    net = choose_net(args.net_name, args.num_classes)

    save_dir = './Results/' + args.save_tag + '-' + args.net_name + '-h' + str(args.height) + '-w' + str(args.width)

    pt_path = save_dir + '/min_eval_loss.pt'
    assert os.path.exists(pt_path)

    test_dir = SegDataset(args.test_dir, num_classes=args.num_classes, input_hw=(args.height, args.width), train_aug=False)
    test_dataloader = DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_steps = len(test_dataloader)

    net.load_state_dict(torch.load(pt_path))
    net.cuda()

    metric = SegmentationMetric(args.num_classes)

    net.eval()
    for batch_id, (batch_data, batch_label) in enumerate(test_dataloader):
        with torch.no_grad():
            batch_data = batch_data.cuda()
            batch_label = batch_label.squeeze(1).cuda()
            output = net(batch_data)
            # print(batch_label.shape, output.shape, batch_label.dtype, output.dtype)

        metric.update(output, batch_label)

        if batch_id % (test_steps // 10) == 0:
            output = output[0,:,:,:].unsqueeze(0)
            output = np.array(torch.max(output.data, 1)[1].squeeze().cpu())
            output = (output * 255).astype('uint8')
            cv2.imwrite(save_dir + '/' + str(batch_id) + '_out.png', output)

            data = np.array(batch_data[0,:,:,:].permute(1, 2, 0).cpu())
            data = (data * 255).astype('uint8')
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir + '/' + str(batch_id) + '_in.png', data)

            label = np.array(batch_label[0,:,:].cpu())
            label = (label * 255).astype('uint8')
            cv2.imwrite(save_dir + '/' + str(batch_id) + '_gt.png', label)

    pixAcc, mIoU = metric.get()
    with open(save_dir + '/test_metric.txt', 'w') as f:
        f.write('pixAcc:{}\nmIoU:{}'.format(pixAcc, mIoU))


def demo(net, pt_path, input_hw, demo_dir):
    net.load_state_dict(torch.load(pt_path))
    net.cuda()

    net.eval()
    for img_path in Path(demo_dir).glob('*.jpg'):
        img_path = str(img_path)
        print(img_path)

        image = Image.open(img_path)
        h, w = image.height, image.width

        image = image.resize(input_hw)

        transform = transforms.ToTensor()
        img_tensor = transform(image)

        with torch.no_grad():
            batch_data = img_tensor.unsqueeze(0).cuda()
            output = net(batch_data)

        output = np.array(torch.max(output.data, 1)[1].squeeze().cpu())
        output = output.astype('float')
        output = cv2.resize(output, (w, h))
        output = (output * 255).astype('uint8')
        cv2.imwrite(img_path[:-4] + '_out.png', output)


def main(args):
    if args.mode == 'trainval':
        trainval(args)
    elif args.mode == 'test':
        test(args)
    else:
        from unet import UNet
        net = UNet(n_classes=args.num_classes)
        pt_path = './Results/' + args.save_tag + '-' + args.net_name + '-h' + str(args.height) + '-w' + str(args.width) + '/min_eval_loss.pt'
        # input_hw = (896, 1280)
        input_hw = (512, 512)
        demo(net, pt_path, input_hw, args.demo_dir)

if __name__ == "__main__":
    args = get_train_args()

    main(args)
