import numpy as np
from PIL import Image
import dataloaders.transforms as transforms
import torchvision
from dataloaders.dataloadern import MyDataloader

iheight, iwidth = 480, 640 # raw image size

class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgbdm'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (224, 224)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)# random scaling
        random_size = (int(s*224),int(s*224))
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
       # transform = torchvision.transforms.Compose([
       #     torchvision.transforms.Resize(self.output_size), # this is for computational efficiency, since rotation can be slow
        #    torchvision.transforms.RandomRotation(angle),
        #    torchvision.transforms.Resize(random_size),
        #    torchvision.transforms.CenterCrop(self.output_size),
        #    torchvision.transforms.RandomHorizontalFlip(do_flip)
        #])
        transform2 = transforms.Compose([
            transforms.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform2(rgb)
        #rgb_n = Image.fromarray(np.uint8(rgb_np * 255))
        #rgb_np = self.color_jitter(rgb_n) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform2(depth_np)
        #depth_np = np.asfarray(depth_np, dtype='float') / 255

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
