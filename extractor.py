from __future__ import print_function

import os
import cv2
import sys
import h5py
import time
import argparse
import tempfile
import itertools
import numpy as np

from sklearn.preprocessing import normalize
from tqdm import tqdm

os.environ['GLOG_minloglevel'] = '2'
import caffe

class EdgeMACExtractor:

    def __init__(self, device_id=0):
        self.device_id = device_id
        
        cnn_model = 'models/retrievalSfM30k-edgemac-vgg.prototxt'
        cnn_weights = 'models/retrievalSfM30k-edgemac-vgg.caffemodel'
        edge_model = 'models/structured_edge_detection_model_opencv.yml.gz'

        if self.device_id < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.device_id)
        
        self.net = caffe.Net(cnn_model, cnn_weights, caffe.TEST)
        self.net.forward(end='pooldescriptor')  # warm-start
        self.edge = cv2.ximgproc.createStructuredEdgeDetection(edge_model)
    
    @staticmethod
    def augment(x):
        xs = []
        # Obtain multiscaled and flipped images (5 scales x 2 flip = 10 images)
        for s in (0.5, np.sqrt(.5), 1, np.sqrt(2), 2):
            for flip in (False, True):
                y = cv2.resize(x, None, fx=s, fy=s)
                if flip:
                    y = cv2.flip(y, 1)
                xs.append(y)
        return xs
    
    @staticmethod
    def filter_edges(x, w=10, p=0.4947, s=500, t=0.0985):
        # Edge Filtering Step.
        # Default parameters are extracted from the MatConvNet trained model
        return w * (x ** p) / (1 + np.exp(-s * (x - t)))
    
    def __extract_one(self, x):
        if x.ndim < 4:
            x = x[None, None, :, :]
        self.net.blobs['x0'].reshape(*x.shape)
        y = self.net.forward(x0=x)['pooldescriptor'].squeeze(axis=(2,3))
        y = normalize(y)
        return y.squeeze()
    
    def extract_from_image(self, img, mask=None, sketch=False, augment=False, invert=False, debug=False):
        x = img
        m = np.ones_like(x) if mask is None else mask
        
        if sketch:
            if x.ndim > 2:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            if invert:
                x = 255 - x

            """ Sketch preprocessing ported from the MATLAB implementation:
                https://github.com/filipradenovic/cnnimageretrieval
            """

            _, x = cv2.threshold(x, .8 * 255, 255, cv2.THRESH_BINARY)  # binarize
            x = np.pad(x, 30, 'constant', constant_values=(0,))  # pads sketch
            x = cv2.ximgproc.thinning(x)  # thinning

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            x = cv2.dilate(x, kernel)  # dilate
            x = cv2.ximgproc.thinning(x)  # thinning

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            x = cv2.dilate(x, kernel)  # larger dilate

            x = x.astype(np.float32) / 255.
            y = self.__extract_one(x)
            
        else:
            s = 227.0 / max(x.shape)
            x = cv2.resize(x, None, fx=s, fy=s)
            m = cv2.resize(m, None, fx=s, fy=s)
            features = []
            xs = self.augment(x) if augment else [x,]
            ms = self.augment(m) if augment else [m,]
            for x, m in zip(xs, ms):
                x = x.astype(np.float32) / 255.
                x = self.edge.detectEdges(x) * m
                x = np.pad(x, 30, 'constant', constant_values=(0,))
                x = self.filter_edges(x)
                y = self.__extract_one(x)
                features.append(y)
            
            y = np.vstack(features).sum(axis=0, keepdims=True)
            y = normalize(y)

        return y

    def _debug_edges(self, x, m, i, augment=False):
        m = np.ones_like(x) if m is None else m
        s = 227.0 / max(x.shape)
        print(x.shape, m.shape)
        x = cv2.resize(x, None, fx=s, fy=s)
        m = cv2.resize(m, None, fx=s, fy=s)
        xs = self.augment(x) if augment else [x,]
        ms = self.augment(ms) if augment else [m,]
        for j, (x, m) in enumerate(zip(xs, ms)):
            x = x.astype(np.float32) / 255.
            m = m.astype(np.float32) / 255.
            x = self.edge.detectEdges(x)
            cv2.imwrite('/debug/edge_{:05d}_{:02d}.png'.format(i, j), x * 255)
            x = np.pad(x, 30, 'constant', constant_values=(0,))
            m = np.pad(m, 30, 'constant', constant_values=(0,))
            x = self.filter_edges(x)
            cv2.imwrite('/debug/edge_{:05d}_{:02d}_filtered.png'.format(i, j), x * 255)
            x *= m
            cv2.imwrite('/debug/edge_{:05d}_{:02d}_masked.png'.format(i, j), x * 255)

    def extract_from_urls(self, urls, out, sketch=False, size=None, augment=False, invert=False):
        print('Extracting (sketch={}, size={}, augment={}, invert={}): {}'.format(sketch, size if size else 'Orig.', augment, invert, out))
        
        features_db = None
        n_images = len(urls)
        
        for i, url in enumerate(tqdm(urls)):
            img = self.load_and_prepare_image(url, size)
            # self._debug_edges(img, i, None, augment=augment); continue

            features = self.extract_from_image(img, sketch=sketch, augment=augment)
            if features_db is None:
                features_db = h5py.File(out, 'w')
                features_dataset = features_db.create_dataset('edgemac', (n_images, 512), dtype=features.dtype)
                
            features_dataset[i] = features
            if i % 1000 == 0:
                features_db.flush()
                
        features_db.flush()
        return features_dataset

    def extract_from_generator(self, images_n_masks, out, sketch=False, size=None, augment=False, invert=False):
        print('Extracting (sketch={}, size={}, augment={}, invert={}): {}'.format(sketch, size if size else 'Orig.', augment, invert, out))
        
        features_db = None
        chunk_size = 1000
        
        for i, image_n_mask in tqdm(enumerate(images_n_masks)):
            if image_n_mask is None:
                continue

            image, mask = image_n_mask
            # img = self.load_and_prepare_image(url, size)
            # self._debug_edges(image, mask, i, augment=augment); continue

            features = self.extract_from_image(image, sketch=sketch, augment=augment, mask=mask)

            if features_db is None:
                features_db = h5py.File(out, 'w')
                features_dataset = features_db.create_dataset('edgemac', (chunk_size, 512), dtype=features.dtype, maxshape=(None, 512))
                
            features_dataset[i] = features
            if (i+1) % chunk_size == 0:
                features_db.flush()
                features_dataset.resize((features_dataset.shape[0] + chunk_size, 512))
                
        features_db.flush()
        return features_dataset

    def prepare_image(self, im, size):
        if size:
            # Get aspect ratio and resize such as the largest side equals S
            im_size_hw = np.array(im.shape[0:2])
            ratio = float(size) / np.max(im_size_hw)
            new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
            im = cv2.resize(im, (new_size[1], new_size[0]))
        return im

    def load_and_prepare_image(self, fname, size):
        im = cv2.imread(fname)
        I = self.prepare_image(im, size)
        return I        

    def extract_from_pil(self, pil_img, sketch=False, size=None, augment=False, invert=False):
        pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
        img = self.prepare_image(img, size)
        return self.extract_from_image(img, sketch=sketch, augment=augment, invert=invert)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RMAC feature extractor')
    parser.add_argument('image_list', type=str, help='File containing the list of images URLs')
    parser.add_argument('output_db', type=str, help='Prefix of output HDF5 files')
    parser.add_argument('-s', '--sketch', action='store_true', help='Treat input as sketch')
    parser.add_argument('-a', '--augment', action='store_true', help='Use Multi-scale and Flip augmentation')
    parser.add_argument('-d', '--device_id', type=int, default=-1, help='Device index of the GPU to use, -1 for CPU')
    args = parser.parse_args()
    
    assert os.path.exists(args.image_list), "List file not found."
    
    urls = [line.rstrip() for line in open(args.image_list, 'rb')]
    n_images = len(urls)
    print('Found {} images.'.format(n_images))
    
    print('Loading the extractor...')
    extractor = EdgeMACExtractor(args.device_id)
    
    features_file = extractor.extract_from_urls(urls, '{}.h5'.format(args.output_db), sketch=args.sketch, augment=args.augment)
    print('Features saved:', features_file)
    
