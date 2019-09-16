# Model conversion from MatConvNet to Caffe

This folder contains the code used to convert the MatConvNet model released by the authors to a OpenCV + Caffe implementation.

## Requirements

You will need a MATLAB installation with MatConvNet and Caffe interfaces.
Clone (recursively) and install the two repos referred in this folder:

 - cnnimageretrieval
 - dagnn_caffe_deploy

Check the documentation of each repo for the exact versions of MatConvNet and Caffe to be used.
The original MatConvNet models are downloaded by the *cnnimageretrieval* repo automatically, check their documentation for more information.

Apply the patch to dagnn_caffe_deploy to support special layer types used in the EdgeMAC MatConvNet model:
```sh
cd conversion/dagnn_caffe_deploy
git apply ../dagnn_caffe_deploy_edgemac_support.patch
```
Once the environment is ready, the `edge2caffe.m` script will:

- extract parameters of the edge filtering step
- convert the edge feature extractor (cnn) to caffe (generate _.prototxt_ and _.caffemodel_ files)


