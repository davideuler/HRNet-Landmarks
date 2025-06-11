import os
import cv2
import csv
import copy
import time
import tqdm
import argparse
import numpy as np
from nets import nn
from timm import utils
from utils import util
from utils import config
from utils.dataset import Dataset

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

config = config.Config()


def train(args):
    util.setup_seed()

    model = nn.get_face_alignment_net(config).cuda()
    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    dataset = Dataset(config, is_train=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 50], 0.1)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    best = float('inf')
    num_batch = len(loader)
    with open('./weights/logs.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'NME'])
            writer.writeheader()
        for epoch in range(args.epochs):
            scheduler.step()
            model.train()
            p_bar = enumerate(loader)

            losses = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_batch)

            for i, (inp, target, meta) in p_bar:
                inp = inp.cuda()
                target = target.cuda()

                output = model(inp)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)

                losses.update(loss.item(), inp.size(0))

                if args.local_rank == 0:
                    s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', losses.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                nme = test(model)
                writer.writerow({'NME': str(f'{nme:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                f.flush()

                # Update best NME
                if best > nme:
                    best = nme

                # Model Save
                ckpt = {'model': copy.deepcopy(model).half()}
                # Save last and best result
                torch.save(ckpt, './weights/last.pt')
                if best == nme:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt
                print(f"Best NME = {best:.3f}")

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')
        util.strip_optimizer('./weights/last.pt')

    torch.cuda.empty_cache()


def test(model=None):
    if model is None:
        model = torch.load(f'./weights/best.pt', map_location='cuda')['model'].float()

    dataset = Dataset(config, is_train=False)
    loader = DataLoader(dataset, 4, False, num_workers=4, pin_memory=True)

    model.half()
    model.eval()

    nme_count = 0
    nme_batch_sum = 0

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(tqdm.tqdm(loader, '%20s' % 'NME')):
            inp = inp.cuda()
            inp = inp.half()
            output = model(inp)
            score_map = output.data.cpu()
            pred = util.decode_pred(score_map, meta['center'], meta['scale'], [64, 64])

            nme_temp = util.compute_nme(pred, meta)

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + pred.size(0)

    nme = nme_batch_sum / nme_count
    print(f"Last NME = {nme:.4f}")
    model.float()
    return nme

@torch.no_grad()
def demo():
    std = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)

    model = torch.load('./weights/best.pt', 'cuda')
    model = model['model'].float()

    detector = util.FaceDetector('./weights/detection.onnx')

    model.half()
    model.eval()

    scale = 1.2
    stream = cv2.VideoCapture('video.mp4')

    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    w = int(stream.get(3))
    h = int(stream.get(4))

    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
        success, frame = stream.read()
        if success:
            boxes = detector.detect(frame, (640, 640))
            boxes = boxes.astype('int32')
            for box in boxes:
                x_min = box[0]
                y_min = box[1]
                x_max = box[2]
                y_max = box[3]
                box_w = x_max - x_min
                box_h = y_max - y_min

                # remove a part of top area for alignment, see paper for details
                x_min -= int(box_w * (scale - 1) / 2)
                y_min += int(box_h * (scale - 1) / 2)
                x_max += int(box_w * (scale - 1) / 2)
                y_max += int(box_h * (scale - 1) / 2)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w - 1)
                y_max = min(y_max, h - 1)
                box_w = x_max - x_min + 1
                box_h = y_max - y_min + 1
                image = frame[y_min:y_max, x_min:x_max, :]
                image = cv2.resize(image, (256, 256))
                image = image.astype('float32')
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # inplace
                cv2.subtract(image, mean, image)  # inplace
                cv2.multiply(image, 1 / std, image)  # inplace
                image = image.transpose((2, 0, 1))
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).unsqueeze(0)

                image = image.cuda()
                image = image.half()

                output = model(image)
                output = output.cpu().detach().numpy()  # convert to numpy
                num_lms = output.shape[1]
                H, W = output.shape[2], output.shape[3]

                for i in range(num_lms):
                    heatmap = output[0, i, :, :]
                    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    x, y = int(x / W * box_w), int(y / H * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 0), 2)

            cv2.imshow('INFERENCE', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    stream.release()
    cv2.destroyAllWindows()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train:
        train(args)
    if args.test:
        test()
    if args.demo:
        demo()

if __name__ == "__main__":
    main()
