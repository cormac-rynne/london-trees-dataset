# London tree dataset

## Summary
This is a dataset of 132 images (1024x1024) of 13944 trees at 33 locations and around the London area. 

## Tree location data
Included is a csv file in the /data folder which contains the latitude and longitude coordinate of every tree.

## Functions included
* Download
* Tree count
* Convert JSONS to CSVs
* Example image with annotations
* Generate GT points map
* Generate GT gaussian map
* Normalisation (calculate dataset mean and stdev for RGB channels)
* Train test split
* Generate file summary
* Generate lat-lon of each tree

## Other files
Pytorch dataset configuration files as used in C^3

# How to get dataset

## Images

To build the images, you will need:
* A GCP account, and a Google Maps Services Static Map API Key (instructions on how to get these available 
* in the repo below)
* [GMapLoader python package](https://github.com/cormac-rynne/gmaploader)
* Use the 'Download images' section within the notebook to download the images
* Approximate cost is £1 - £1.50 in Google Maps API fees

## JSONS
[Available here (Google Drive Link)](https://drive.google.com/drive/folders/1YsdnD7_sKK98rDQVZ7zoyYmGOv0FVOXW?usp=sharing)

## Ground truth point x,y coordinate .csv files
These are provided in the repo under /gt_points

## Ground truth Map .h5 files
Use code 'Generate GT Map' provided in notebook

# How to change labels
Images were labeled using [labelMe](https://github.com/wkentaro/labelme). To alter the labels:
* Create an empty folder
* Move both the image and JSON for the image into this folder (filenames should be the same apart from the extension)
* use ```python -m labelme``` to open labelMe
* Use the 'Open Dir' function in label me to point to the directory with the images and JSONS in
* Images should appear in the bottom right window, select the one you want to change, labels will show