#!/bin/bash

DATA="/path/to/folder/with/images"
FEAT="/path/to/output/folder"

# features will be saves in the $FEAT folder

function prepare_list_files() {
  echo "Generating: list.txt"
  find ${DATA}/ -iname "*.png" | sort -V | sed -e "s|^${DATA}|imgs|" > list.txt
}

function extract_edgemac() {
  echo "Extracting EdgeMAC..."
  docker run --runtime=nvidia --rm -it \
    -v ${PWD}/list.txt:/code/list.txt \
    -v ${DATA}:/code/imgs \
    -v ${FEAT}:/features \
    -w /code \
    fabiocarrara/edgemac-extractor python extractor.py list.txt /features/edgemac_ -d 0 -a

}

prepare_list_files
extract_edgemac
