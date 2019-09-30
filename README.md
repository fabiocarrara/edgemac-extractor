# EdgeMAC Extractor

This repo provides a containerized EdgeMAC image descriptor extractor with REST API.
The implemented EdgeMAC descriptor is described in:

> Radenovic, F., Tolias, G. and Chum, O., 2018. Deep shape matching. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 751-767).

The Caffe models used here have been converted from the original MatConvNet models released by the authors (see the [conversion/](conversion/) folder for details about the conversion).

## Usage

Build and run the docker container to have a HTTP REST EdgeMAC extractor service that accepts URLs or image uploads and returns a JSON array containing the extracted features.
Check [service.py](service.py) for details about the API.

Check [usage.sh](usage.sh) if you want to use the extractor in batch/off-line mode.





