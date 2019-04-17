from os.path import join
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class AnticlineDataset(Dataset):
    def __init__(self, root_dir, transforms=None, trainFlag = 'train'):
        self.root_dir = root_dir
        self.transforms = transforms
        self.trainFlag = trainFlag

        # read image paths and points for mask
        with open(join(root_dir, "dataset.json")) as file:
            self.labels = json.load(file)[trainFlag]

        self.imagePaths = list(self.labels.keys())

        print("loading image size")
        self.imageSize = []
        for path in self.imagePaths:
            image = Image.open(join(self.root_dir, path))
            self.imageSize.append(image.size)

    def __getitem__(self, idx):
        # load the image as a PIL Image
        path = self.imagePaths[idx]
        image = Image.open(join(self.root_dir, path))

        # loading points for mask
        pointsForMask = self.labels[path]

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # process the points to bboxs
        bboxImg = []
        labels = []
        for instance in pointsForMask:
            instanceX = [instance[0][i] for i in range(len(instance[0])) if i % 2 == 0]
            instanceY = [instance[0][i] for i in range(len(instance[0])) if i % 2 == 1]
            x1 = np.min(instanceX)
            x2 = np.max(instanceX)
            y1 = np.min(instanceY)
            y2 = np.max(instanceY)
            bboxImg.append([x1, y1, x2, y2])
            labels.append(1)    # only anticline
        # classes
        labels = torch.tensor(labels)

        # create a BoxList from the boxes
        target = BoxList(bboxImg, image.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)

        # loading mask
        masks = pointsForMask
        masks = SegmentationMask(masks, image.size)
        target.add_field("masks", masks)

        if self.transforms:
            image, target = self.transforms(image, target)

        # return the image, the boxlist and the idx in your dataset
        return image, target, idx

    def __len__(self):
        return len(self.imagePaths)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_width, img_height = self.imageSize[idx]
        return {"height": img_height, "width": img_width}
