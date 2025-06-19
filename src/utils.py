from sklearn.preprocessing import normalize as snorm
from matplotlib import cm
import datetime
import pickle as pkl
import shutil
import time
import os
import numpy as np  
import cv2  
import matplotlib.pyplot as plt  
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pickle as pkl
import os.path as osp
from scipy import sparse
import json
import pandas as pd

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import shutil
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import json
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def erode_mask(mask, erosion_size=50):
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask

def sample_grid_points(eroded_mask, grid_spacing=100):
    points = []
    rows, cols = eroded_mask.shape
    for y in range(grid_spacing, rows - grid_spacing, grid_spacing):
        for x in range(grid_spacing, cols - grid_spacing, grid_spacing):
            if eroded_mask[y, x] == 255:
                points.append([x, y])
    return points

def create_json(mask, image_path, image_height, image_width, erosion_size=10, grid_spacing=50):
    eroded_mask = erode_mask(mask, erosion_size)
    points = sample_grid_points(eroded_mask, grid_spacing)
    
    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [{
            "mask": None,
            "label": "Wound",
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "linestrip",
            "flags": {}
        }],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    with open(f"{image_path.split('.')[0]}.json", 'w') as f:
        json.dump(json_data, f, indent=4)
        
def display_images(ims,title="",dpi=100):  
    def norm255(im_):  
        im_ = np.array(im_)  
        im_ = im_ -np.min(im_)  
        im_ = im_/np.max(im_)*255.  
        im_ = im_.astype(np.uint8)  
        return im_.copy()  
    imglist = ims.copy()  
    for i in range(len(imglist)):  
        if i == 0:  
            img = imglist[0]  
            img = norm255(img)  
        else:  
            im1, im2 = img, imglist[i]  
            if len(im1.shape)==2:  
                im1 = np.stack((im1,) * 3, axis=-1)  
            if len(im2.shape)==2:  
                im2 = np.stack((im2,) * 3, axis=-1)  
            im1 = norm255(im1)  
            im2 = norm255(im2)  
            f = im1.shape[0]/im2.shape[0]  
            im2 = cv2.resize(im2, (int(im2.shape[1]*f), im1.shape[0]))  
            img = np.hstack([im1, im2])  
    img = img.astype(np.uint8)   
    plt.figure(dpi=dpi)   
    if len(img.shape)==3:  
        plt.imshow(img)   
    else:  
        plt.imshow(img,cmap="gray")   
    plt.xticks([])   
    plt.yticks([])   
    plt.title(title)  
    plt.axis("off")   
    plt.show()  
    
show2d = display_images



def list_jpg_paths(base_dir):
    jpg_paths = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames.sort()  # 对当前目录下的子目录进行排序
        filenames.sort()  # 对当前目录下的文件进行排序

        for filename in filenames:
            if filename.endswith('.jpg'):
                jpg_path = os.path.abspath(os.path.join(dirpath, filename))
                jpg_paths.append(jpg_path)

    return jpg_paths



def list_tif_paths(base_dir):
    tif_paths = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames.sort()  # 对当前目录下的子目录进行排序
        filenames.sort()  # 对当前目录下的文件进行排序
        for filename in filenames:
            if filename.endswith('.tif'):
                tif_path = os.path.abspath(os.path.join(dirpath, filename))
                tif_paths.append(tif_path)

    return tif_paths

def interpolate_points(points, num_interpolations):
    if num_interpolations < 1:
        return points
    interpolated_points = []
    for i in range(len(points) - 1):
        interpolated_points.append(points[i])
        for j in range(1, num_interpolations + 1):
            fraction = j / (num_interpolations + 1)
            new_point = points[i] + fraction * (points[i + 1] - points[i])
            interpolated_points.append(new_point)
    interpolated_points.append(points[-1])
    return np.array(interpolated_points)


def largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_mask = np.zeros_like(mask)
    new_mask[labels == largest_label] = 255
    return new_mask

def read_pts(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    points = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'linestrip':
            points.extend(shape['points'])
    return np.array(points)


def save_as_sparse(root,name,data):
    sparse_data = sparse.csr_matrix(data)
    fp = open(osp.join(root,name+"_sparse.pkl"),"wb")
    pkl.dump(sparse_data,fp)
    fp.close()
# save_as_sparse(dir_path,filename,label_masks)

def read_sparse(root,name):
    fp = open(osp.join(root,name+"_sparse.pkl"),"rb")
    sparse_data = pkl.load(fp)
    fp.close()
    label_masks = sparse_data.toarray()
    return label_masks
# label_masks = read_sparse(dir_path,filename)


def save_pkl(root,name,data):
    fp = open(osp.join(root,name+".pkl"),"wb")
    pkl.dump(data,fp)
    fp.close()

def read_pkl(root,name):
    fp = open(osp.join(root,name+".pkl"),"rb")
    data = pkl.load(fp)
    fp.close()
    return data

def get_name(path):
    filename = os.path.basename(path)
    filename_without_extension = os.path.splitext(filename)[0]
    return filename,filename_without_extension

def label_to_color_image(label_mask):
    if label_mask.ndim != 2:
        raise ValueError('label_mask must be 2D array')
    h, w = label_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.unique(label_mask)
    np.random.seed(0)
    hues = np.random.randint(0, 180, size=np.max(labels) + 1)
    colors = np.zeros((len(hues), 3), dtype=np.uint8)
    colors[1:, 0] = hues[1:] 
    colors[1:, 1] = 255 
    colors[1:, 2] = 255 
    color_mask = colors[label_mask]
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_HSV2BGR)
    return color_mask

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([144/255, 35/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show2d_image(ims,title="",dpi=100):
        def norm255(im_):  
            im_ = np.array(im_)  
            im_ = im_ -np.min(im_)  
            im_ = im_/np.max(im_)*255.  
            im_ = im_.astype(np.uint8)  
            return im_.copy()  
        imglist = ims.copy()  
        for i in range(len(imglist)):  
            if i == 0:  
                img = imglist[0]  
                img = norm255(img)  
            else:  
                im1, im2 = img, imglist[i]  
                if len(im1.shape)==2:  
                    im1 = np.stack((im1,) * 3, axis=-1)  
                if len(im2.shape)==2:  
                    im2 = np.stack((im2,) * 3, axis=-1)  
                im1 = norm255(im1)  
                im2 = norm255(im2)  
                f = im1.shape[0]/im2.shape[0]  
                im2 = cv2.resize(im2, (int(im2.shape[1]*f), im1.shape[0]))  
                img = np.hstack([im1, im2])  
        img = img.astype(np.uint8)   
        return img


def display_images(ims,title="",dpi=100):  
    def norm255(im_):  
        im_ = np.array(im_)  
        im_ = im_ -np.min(im_)  
        im_ = im_/np.max(im_)*255.  
        im_ = im_.astype(np.uint8)  
        return im_.copy()  
    imglist = ims.copy()  
    for i in range(len(imglist)):  
        if i == 0:  
            img = imglist[0]  
            img = norm255(img)  
        else:  
            im1, im2 = img, imglist[i]  
            if len(im1.shape)==2:  
                im1 = np.stack((im1,) * 3, axis=-1)  
            if len(im2.shape)==2:  
                im2 = np.stack((im2,) * 3, axis=-1)  
            im1 = norm255(im1)  
            im2 = norm255(im2)  
            f = im1.shape[0]/im2.shape[0]  
            im2 = cv2.resize(im2, (int(im2.shape[1]*f), im1.shape[0]))  
            img = np.hstack([im1, im2])  
    img = img.astype(np.uint8)   
    plt.figure(dpi=dpi)   
    if len(img.shape)==3:  
        # img = img[:, :, ::-1]   
        plt.imshow(img)   
    else:  
        plt.imshow(img,cmap="gray")   
    plt.xticks([])   
    plt.yticks([])   
    plt.title(title)  
    plt.axis("off")   
    plt.show()  

show2d = display_images

def remove_small_components(m,area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < area:
            labels[labels == label] = 0
    new_m = np.where(labels > 0, 255, 0).astype(np.uint8)
    return new_m


class SAM_mask_generator():
    def __init__(self, sam_checkpoint="sam_vit_l_0b3195.pth"):
        model_type = "_".join(sam_checkpoint.split("_")[1:3])
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam = sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.label_count = 1
        self.min_area = 300

    def generate(self,image):
        anns = self.mask_generator.generate(image)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        h, w = sorted_anns[0]['segmentation'].shape
        color_mask = np.zeros((h, w, 3))
        label_mask = np.zeros((h, w), dtype=np.int32)
        for ann in sorted_anns:
            m = ann['segmentation']
            if np.sum(m) < self.min_area:
                continue
            label_mask[m] = self.label_count
            self.label_count = self.label_count + 1
        return label_mask

def image_padding(image,patch_size=512):
    rows, cols = image.shape[:2]
    image_pad1 = image.copy()
    if cols >= patch_size and rows <= patch_size:
        nrows = patch_size
        ncols = int(np.ceil(cols/patch_size)*patch_size)
        image_pad1 = np.pad(image, ((0, patch_size-rows),
                                    (0, ncols-cols), (0, 0)), "constant")
    elif rows >= patch_size and cols <= patch_size:
        ncols = patch_size
        nrows = int(np.ceil(rows/patch_size)*patch_size)
        image_pad1 = np.pad(
            image, ((0, nrows-rows), (0, patch_size-cols), (0, 0)), "constant")
    elif rows >= patch_size and cols >= patch_size:
        ncols = int(np.ceil(cols/patch_size)*patch_size)
        nrows = int(np.ceil(rows/patch_size)*patch_size)
        image_pad1 = np.pad(
            image, ((0, nrows-rows), (0, ncols-cols), (0, 0)), "constant")
    return image_pad1


# color_masks = label_to_color_image(label_masks)
# compress_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
# cv2.imwrite("color_masks.jpg",color_masks,compress_params)