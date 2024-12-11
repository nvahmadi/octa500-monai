# pip install h5py, tqdm

import os
import shutil
import re
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat   

# FOV: 3mm * 3mm * 2mm
# Volume: 304pixel * 304pixel * 640pixel

def natural_sort_key(s):
    """
    Helper function for natural sorting of filenames.
    Extracts numerical parts for sorting, ensuring 1.bmp < 2.bmp < 10.bmp.
    """
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def load_images_to_3d_array(directory, z_inverse=False):
    """
    Load BMP images from a directory and create a 3D array based on grayscale channel (channel 0).

    Parameters:
        directory (str): Path to the directory containing the BMP images.

    Returns:
        np.ndarray: A 3D NumPy array of shape (Height, Width, NumImages) with grayscale values.
    """
    # List all BMP files in the directory and sort them naturally
    files = sorted(
        (f for f in os.listdir(directory) if f.endswith('.bmp')),
        key=natural_sort_key
    )
    num_images = len(files)
    if num_images == 0:
        raise ValueError("No BMP images found in the directory.")
    
    # Open the first image to determine its size
    first_image = Image.open(os.path.join(directory, files[0]))
    width, height = first_image.size
    
    # Preallocate the 3D array
    volume = np.empty((height, width, num_images), dtype=np.uint8)
    print(f"{volume.__class__=}")
    
    # Load images and extract channel 0
    for i, file in enumerate(files):
        img = Image.open(os.path.join(directory, file))
        arr = np.array(img)  # Convert to NumPy array
        if z_inverse: 
            volume[:, :, num_images-1-i] = arr[:, :]
        else:
            volume[:, :, i] = arr[:, :]
    
    return volume

def load_volumes(subject_id=10301, basepath=r"C:\Projects\datasets\OCTA-500", fov="3mm", verbose=False, z_inverse=False):
    path_subject = os.path.join(basepath, "OCTA_"+fov, "__modality__",  str(subject_id))
    
    pn_oct  = os.path.join(path_subject.replace("__modality__", "OCT"))
    pn_octa = os.path.join(path_subject.replace("__modality__", "OCTA"))
    if verbose: 
        print(f"Loading OCT/OCTA images from folder: {pn_oct}")
    
    arr_oct = load_images_to_3d_array(pn_oct, z_inverse=z_inverse)
    arr_octa = load_images_to_3d_array(pn_octa, z_inverse=z_inverse)
    
    return arr_oct, arr_octa  

def load_gt_faz3d(subject_id=10301, 
                  basepath=r"C:\Projects\datasets\OCTA-500",
                  res="3mm",
                  verbose=False):
    ff_faz3d = os.path.join(basepath, "Label", "GT_FAZ3D",  str(subject_id)+".mat")
    if verbose:
        print("FAZ 3D file:")
        print(ff_faz3d)
    with h5py.File(ff_faz3d, 'r') as mat_data:
        if verbose:
            # List all variable names
            print("Variables in the file:")
            print(list(mat_data.keys()))
        
        # Access a specific variable (replace 'FAZlabel' with your variable's name)
        if 'FAZlabel' in mat_data:
            data = mat_data['FAZlabel'][:]  # Load the data into memory as a NumPy array
    data = np.transpose(data, (2, 1, 0))
    
    #node_faz3d = addVolumeFromArray(data, name=f"gt_faz3d_{subject_id}", nodeClassName="vtkMRMLLabelMapVolumeNode")
    #node_faz3d.SetSpacing(3.0/304.0, 3.0/304.0, 2.0/640)
    
    return node_faz3d

def load_gt_layers3d(subject_id=10301, 
                     basepath=r"C:\Projects\datasets\OCTA-500",
                     res="3mm",
                     verbose=False):
    ff_layers = os.path.join(basepath, "Label", "GT_Layers",  str(subject_id)+".mat")
    
    if verbose:
        print("Layers file:")
        print(ff_layers)
    
    mat_data = loadmat(ff_layers)
    if verbose:
        print(mat_data.keys())
    
    # Access a specific variable 
    if 'Layer' in mat_data:
        L = mat_data['Layer'][:]  # Load the data into memory as a NumPy array
    
    L3d = layers_to_3d(L)
    L3d = np.transpose(L3d, (2, 1, 0))
    #node_gt_layers3d = addVolumeFromArray(L3d, name=f"gt_layers3d_{subject_id}", nodeClassName="vtkMRMLLabelMapVolumeNode")
    #node_gt_layers3d.SetSpacing(3.0/304.0, 3.0/304.0, 2.0/640)
    
    return node_gt_layers3d

def layers_to_3d(L, index_5_to_bg=False):
    L = L.transpose(1, 2, 0)
    # Assuming V is your 3D volume of shape (X, Y, Z)
    # And L is your layer information of shape (X, Y, 6)
    #X, Y, Z = V.shape
    X, Y, Z = (304, 304, 640)
    V = np.empty((X, Y, Z), dtype=np.uint8)

    # Create a 3D grid of indices for the Z-axis
    z_indices = np.arange(Z).reshape(1, 1, Z)  # Shape (1, 1, Z)

    # Create a mask for each layer A-F
    layer_masks = []

    # Initialize start indices for the layers
    start_indices = np.zeros((X, Y), dtype=int)

    for i in range(6):  # Iterate over the 6 layers
        end_indices = L[:, :, i]  # End indices for the current layer
        # Create a mask for the current layer
        mask = (z_indices >= start_indices[:, :, None]) & (z_indices <= end_indices[:, :, None])
        layer_masks.append(mask)
        # Update start index for the next layer
        start_indices = end_indices + 1

    # Now, fill the volume `V` using the masks
    # You can assign any values you want for each layer. Here's an example:
    for i, mask in enumerate(layer_masks):
        V[mask] = i  # Assign a unique value for each layer (e.g., 1 for A, 2 for B, ...)
    # Optional: Set top layer (index 5) to background (index 0, like layer 0)
    if index_5_to_bg:
        V[V==5] = 0
    # V is now filled with values corresponding to the layers
    return V

#basepath = r"C:\Projects\datasets\OCTA-500"
if __name__ == "__main__":
    basepath = r"D:\Datasets\OCTA-500"
    subject_ids = list(range(10301, 10500+1))
    for idx, subject_id in enumerate(subject_ids):
        path_out = os.path.join(basepath, "OCTA500_MONAI_3mm", str(subject_id))
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if os.path.exists(os.path.join(path_out, "label_3D_Layers.nii.gz")):
            # this subject was already done
            continue
        
        # Load greyscale volumes
        nvol_oct, nvol_octa = load_volumes(subject_id=subject_id, basepath=basepath)
        # load groundtruth: FAZ 3D
        node_gt_faz3d = load_gt_faz3d(subject_id=subject_id, basepath=basepath)
        # load groundtruth: retinal layers
        node_gt_layers3d = load_gt_layers3d(subject_id=subject_id, basepath=basepath)
        
        saveNode(nvol_oct, os.path.join(path_out, "vol_OCT.nii.gz"))
        saveNode(nvol_octa, os.path.join(path_out, "vol_OCTA.nii.gz"))
        saveNode(node_gt_faz3d, os.path.join(path_out, "label_3D_FAZ.nii.gz"))
        saveNode(node_gt_layers3d, os.path.join(path_out, "label_3D_Layers.nii.gz"))
        
        su.closeScene()
        
        # copy 2D gt maps
        for tag_gt in ["GT_Artery", "GT_Capillary", "GT_CAVF", "GT_FAZ", "GT_LargeVessel", "GT_Vein"]:
            src_filepath = os.path.join(basepath, "Label", tag_gt, str(subject_id)+".bmp")
            tgt_filepath = os.path.join(path_out, "label_2D_" + tag_gt.replace("GT_", "") + ".bmp")
            shutil.copy(src_filepath, tgt_filepath)
        
        print(f"Finished subject {subject_id} ({idx+1} of {len(subject_ids)}).")






# Optional: invert z-axis
if False:
    M_flipZ = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    trf_flipZ = su.transformFrom4x4Matrix("z_flip", M_flipZ)
    for n in [nvol_oct, nvol_octa]:
        n.SetAndObserveTransformNodeID(trf_flipZ.GetID())

if False:
    # Set up transforms
    M_axisflip = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    M_IJKtoRAS = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    trf_axisflip = su.transformFrom4x4Matrix("axis_flip", M_axisflip)
    trf_IJKtoRAS = su.transformFrom4x4Matrix("IJKtoRAS", M_IJKtoRAS)
    trf_axisflip.SetAndObserveTransformNodeID(trf_IJKtoRAS.GetID())

