# CVEMP 2023

Code for the paper _Automatic Alignment of Multi-scale Aerial and Underwater Photogrammetric Point Clouds: a Case Study in the Maldivian Coral Reef_,
submitted at CVEMP 2023.

Code tested with python 3.10. 

Install the requirements with:
```
pip3 install -r requirements.txt
```

To extract the features with 3DSmoothNet, please consult the README. inside the `3DSmoothNet` folder, and then put the resulting .npz files in the `extracted_features` folder with the name "<pointcloud_name>_3dsmoothnet".

To run the registration algorithms, adjust the parameters in `registration.py` and run the script.

To obtain the error metrics run `check_gt.py`