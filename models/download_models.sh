#!/bin/bash

function download_models() {
  # OpenCV Structured Edge Detection Model (Dollàr)
  EDGE_DET_MODEL="structured_edge_detection_model_opencv.yml.gz"
  EDGE_DET_MODEL_URL="https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz"
  if [ ! -f "${EDGE_DET_MODEL}" ]; then wget ${EDGE_DET_MODEL_URL} -o ${EDGE_DET_MODEL}; fi

  # Converted EdgeMAC CNN model
  EDGE_MAC_MODEL="retrievalSfM30k-edgemac-vgg.prototxt"
  EDGE_MAC_MODEL_URL=".../retrievalSfM30k-edgemac-vgg.prototxt"
  if [ ! -f "${EDGE_MAC_MODEL}" ]; then wget ${EDGE_MAC_MODEL_URL} -o ${EDGE_MAC_MODEL}; fi

  EDGE_MAC_MODEL_WEIGHTS="retrievalSfM30k-edgemac-vgg.caffemodel"
  EDGE_MAC_MODEL_WEIGHTS_URL="../retrievalSfM30k-edgemac-vgg.caffemodel"
  if [ ! -f "${EDGE_MAC_MODEL_WEIGHTS}" ]; then wget ${EDGE_MAC_MODEL_WEIGHTS_URL} -o ${EDGE_MAC_MODEL_WEIGHTS}; fi
}

download_models

