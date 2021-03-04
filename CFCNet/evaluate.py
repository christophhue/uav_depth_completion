#!/usr/bin/env python
#SBATCH --job-name=fusenet
#SBATCH --nodes=1
#SBATCH --cpus=4
#SBATCH --gres=gpu:1
#SBATCH --time="UNLIMITED"

import time
from options.options import AdvanceOptions
from models import create_model
from util.visualizer import Visualizer
from dataloaders.nyu_dataloader import NYUDataset
from dataloaders.kitti_dataloader import KITTIDataset
from dataloaders.own_dataset_loader_val import OwnDataset
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo,ORBSampling
import numpy as np
import random
import torch
import cv2
import utils
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


cmap = plt.cm.viridis

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))* 255.0
    return image_numpy.astype(imtype)

def colored_depthmap(depth):
    depth = depth.squeeze(0)
    depth = depth[0].cpu().float().numpy()
    if depth.shape[0] == 1:
        depth = np.tile(depth, (3, 1, 1))
    cmap = plt.cm.viridis
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_opt = AdvanceOptions().parse(False)


    ## select sparsifier
    if test_opt.sD_sampler == "uniform":
        sparsifier = UniformSampling(test_opt.nP, max_depth=np.inf)
    if test_opt.sD_sampler == "orb":
        sparsifier = ORBSampling(max_depth=np.inf)
    if test_opt.sD_sampler == "stereo":
        sparsifier = SimulatedStereo(num_samples=test_opt.nP, max_depth=np.inf)



    test_dataset = OwnDataset(split='val', modality='rgbdm', sparsifier=sparsifier)

    ### Please use this dataloder if you want to use NYU
    # test_dataset = NYUDataset(test_opt.test_path, type='val',
    # 		modality='rgbdm', sparsifier=sparsifier)


    test_opt.phase = 'val'
    test_opt.batch_size = 1
    test_opt.num_threads = 1
    test_opt.serial_batches = True
    test_opt.no_flip = True

    ##manually specify the best epoch; model is loaded from that epoch
    test_opt.epoch = 9

    test_data_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=test_opt.batch_size, shuffle=False, num_workers=test_opt.num_threads, pin_memory=True)

    test_dataset_size = len(test_data_loader)
    print('#test images = %d' % test_dataset_size)

    model = create_model(test_opt, test_dataset)
    model.eval()
    model.setup(test_opt)
    visualizer = Visualizer(test_opt)
    test_loss_iter = []
    gts = None
    preds = None
    epoch_iter = 0
    model.init_test_eval()
    epoch = 0
    num = 5
    # How many images to save in an image
    if not os.path.exists('vis'):
        os.makedirs('vis')
    with torch.no_grad():
        iterator = iter(test_data_loader)
        i = 0
        while True:
            try:
                nn = next(iterator)
            except IndexError:
                print("Catch and Skip!")
                continue
            except StopIteration:
                break

            data, target, scale = nn[0], nn[1], nn[2]
            model.set_new_input(data.float(),target,scale)
            model.forward()
            model.test_depth_evaluation()
            model.get_loss()
            epoch_iter += test_opt.batch_size
            losses = model.get_current_losses()
            test_loss_iter.append(model.loss_dcca.item())
            rgb_input = model.rgb_image
            depth_input = model.sparse_depth
            rgb_sparse = model.sparse_rgb
            depth_target = model.depth_image
            depth_est = model.depth_est
            if i % num == 0:
                img_merge = utils.merge_into_row_with_pred_visualize(rgb_input, depth_input, rgb_sparse, depth_target,
                                                                     depth_est)
            elif i % num < num - 1:
                row = utils.merge_into_row_with_pred_visualize(rgb_input, depth_input, rgb_sparse, depth_target,
                                                               depth_est)
                img_merge = utils.add_row(img_merge, row)
            elif i % num == num - 1:
                filename = 'vis/' + str(i) + '.png'
                utils.save_image(img_merge, filename)

            i += 1

            print('test epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(test_dataset) * test_opt.batch_size), end='\r')
            print(
          'RMSE={result.rmse:.4f}({average.rmse:.4f}) '
          'MSE={result.mse:.4f}({average.mse:.4f}) '
          'MAE={result.mae:.4f}({average.mae:.4f}) '
          'Delta1={result.delta1:.4f}({average.delta1:.4f}) '
          'Delta2={result.delta2:.4f}({average.delta2:.4f}) '
          'Delta3={result.delta3:.4f}({average.delta3:.4f}) '
          'REL={result.absrel:.4f}({average.absrel:.4f}) '
          'Lg10={result.lg10:.4f}({average.lg10:.4f}) '.format(
         result=model.test_result, average=model.test_average.average()))
    avg_test_loss = np.mean(np.asarray(test_loss_iter))
