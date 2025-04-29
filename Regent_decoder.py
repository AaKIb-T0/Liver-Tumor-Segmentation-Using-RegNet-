# -*- coding: utf-8 -*-
"""
INFERENCE SCRIPT: Liver Tumor Segmentation with Pre-trained RegNet
------------------------------------------------------------------
Loads a pre-trained SegNetRegNet model and runs inference on specified
NIfTI volumes, saving the predicted segmentation masks.
"""

# ==========================================
# Cell 1: Install Necessary Libraries
# ==========================================
!pip install nibabel torch torchvision tqdm --quiet # Need torch, torchvision, nibabel, tqdm
print("Libraries checked/installed.")

# ==========================================
# Cell 2: Import Libraries
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models # Need this to define the model architecture again

import nibabel as nib
import os
import numpy as np
from PIL import Image
import time
import gc
import re
from tqdm.notebook import tqdm

print(f"PyTorch Version: {torch.__version__}")
print(f"Nibabel Version: {nib.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Cell 3: Configuration for Inference
# ==========================================
# --- USER MUST SET THESE ---
TEST_MODEL_PATH = '/content/regnet_lits_tumor_segmentation_trained.pth' # <--- PATH TO THE SAVED MODEL FROM TRAINING
TEST_DATA_DIR = '/content/lits_data/' # <--- DIRECTORY CONTAINING NIFTI VOLUMES TO TEST
PREDICTION_DIR = '/content/predictions/' # <--- WHERE TO SAVE OUTPUT MASKS
# List the IDs (numbers) of the volumes you want to test from TEST_DATA_DIR
VOLUME_IDS_TO_TEST = ['0', '45', '100'] # Example: test volume-0.nii, volume-45.nii etc.

# --- Parameters that MUST match the training configuration ---
RESIZE_SIZE = (128, 128) # Must be the same size used during training for model input
SLICE_AXIS = 2           # Must be the same axis used during training if relevant to preprocessing (here it's used for iterating)

# --- Inference Parameters ---
PREDICTION_THRESHOLD = 0.5 # Threshold for converting sigmoid output to binary mask

print("Inference Configuration set.")
print(f"Loading model from: {TEST_MODEL_PATH}")
print(f"Looking for test volumes in: {TEST_DATA_DIR}")
print(f"Saving predictions to: {PREDICTION_DIR}")
print(f"Using Resize Size: {RESIZE_SIZE}")
print(f"Using Slice Axis: {SLICE_AXIS}")


# ==========================================
# Cell 4: Define Model Architecture (MUST MATCH TRAINING)
# ==========================================
# --- Copy the EXACT SAME SegNetRegNet class definition from the training script ---
class SegNetRegNet(nn.Module):
    # (Keep the exact same Model class definition from the training script)
    # ... (rest of the class definition as provided previously) ...
    def __init__(self, num_classes=1):
        super(SegNetRegNet, self).__init__()
        try: # Use updated weights API if available (loading local weights, so pretrained doesn't matter here)
             weights = models.RegNet_Y_400MF_Weights.IMAGENET1K_V1
             full_backbone = models.regnet_y_400mf(weights=weights) # Load architecture, ignore weights initially
             print("Defined RegNet architecture using RegNet_Y_400MF_Weights.")
        except AttributeError: # Fallback for older torchvision
             full_backbone = models.regnet_y_400mf(pretrained=False) # Load architecture ONLY
             print("Defined RegNet architecture using legacy pretrained=False.")

        self.stem = full_backbone.stem
        self.trunk_output = full_backbone.trunk_output
        encoder_channels = 440
        # No need to print encoder channels again here, but keep definition
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, 256, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16, num_classes, kernel_size=1) )
    def forward(self, x):
        x = self.stem(x)
        features = self.trunk_output(x)
        segmentation_map = self.decoder(features)
        return segmentation_map
print("SegNetRegNet model class defined for inference.")

# ==========================================
# Cell 5: Load Model and Setup Inference
# ==========================================
inference_model = None
TEST_IMAGE_PATHS = []

# Find the actual file paths for the requested volume IDs
if os.path.isdir(TEST_DATA_DIR):
    all_vols_in_dir = os.listdir(TEST_DATA_DIR)
    for vol_id in VOLUME_IDS_TO_TEST:
        found = False
        # Construct potential filenames (handle .nii and .nii.gz)
        potential_filenames = [f'volume-{vol_id}.nii', f'volume-{vol_id}.nii.gz']
        for fname in potential_filenames:
            if fname in all_vols_in_dir:
                TEST_IMAGE_PATHS.append(os.path.join(TEST_DATA_DIR, fname))
                print(f"Found test image: {fname}")
                found = True
                break # Found one version, move to next ID
        if not found:
            print(f"Warning: Test volume for ID '{vol_id}' not found in {TEST_DATA_DIR}")
else:
    print(f"Error: Test data directory '{TEST_DATA_DIR}' not found.")

# Load model weights if the path exists
if os.path.exists(TEST_MODEL_PATH):
    try:
        print("\nLoading model architecture...")
        inference_model = SegNetRegNet(num_classes=1) # Instantiate the model structure
        print(f"Loading saved weights from {TEST_MODEL_PATH}...")
        inference_model.load_state_dict(torch.load(TEST_MODEL_PATH, map_location=device)) # Load the weights
        inference_model.to(device) # Move model to GPU/CPU
        inference_model.eval() # <<<--- SET TO EVALUATION MODE --->>>
        print("Model loaded successfully and set to evaluation mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        inference_model = None
else:
    print(f"Error: Model weights file not found at {TEST_MODEL_PATH}. Cannot run inference.")

# Create prediction directory
os.makedirs(PREDICTION_DIR, exist_ok=True)

# Define Transform for Test Images (MUST MATCH TRAINING - only need image part)
test_img_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE), # Use the same resize dimensions as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Same normalization
])
print("Inference setup complete.")


# ==========================================
# Cell 6: Run Inference Loop
# ==========================================
if inference_model is not None and TEST_IMAGE_PATHS:
    print(f"\n--- Starting Inference on {len(TEST_IMAGE_PATHS)} volumes ---")
    inference_start_time = time.time()

    with torch.no_grad(): # Disable gradient calculations
        for img_path in TEST_IMAGE_PATHS:
            print(f"\nProcessing: {os.path.basename(img_path)}...")
            processing_file_start_time = time.time()
            predicted_slices_list = []

            try:
                img_nii = nib.load(img_path)
                img_data_full = img_nii.get_fdata().astype(np.float32)
                original_shape = img_data_full.shape
                original_affine = img_nii.affine
                original_header = img_nii.header
                num_slices = original_shape[SLICE_AXIS]
                print(f"  Volume shape: {original_shape}, Slices ({SLICE_AXIS=}): {num_slices}")

                for slice_idx in tqdm(range(num_slices), desc=f"  Slices", leave=False):
                    # Extract, Preprocess, Predict, Post-process Slice
                    # (Keep the exact same slice processing logic from Cell 15 of the v5 script)
                    # ... (rest of the per-slice loop logic) ...
                    slice_obj = [slice(None)] * img_data_full.ndim
                    slice_obj[SLICE_AXIS] = slice_idx
                    img_slice = img_data_full[tuple(slice_obj)]

                    min_val, max_val = np.min(img_slice), np.max(img_slice)
                    if max_val > min_val: img_slice_norm = (img_slice - min_val) / (max_val - min_val)
                    else: img_slice_norm = np.zeros_like(img_slice)
                    img_pil = Image.fromarray(np.uint8(img_slice_norm * 255)).convert("RGB")
                    img_tensor = test_img_transform(img_pil).unsqueeze(0).to(device)

                    output = inference_model(img_tensor)
                    probs = torch.sigmoid(output)
                    pred_mask_tensor = (probs > PREDICTION_THRESHOLD).float()
                    original_slice_shape_hw = tuple(s for i, s in enumerate(original_shape) if i != SLICE_AXIS)
                    pred_mask_resized = F.interpolate(pred_mask_tensor, size=original_slice_shape_hw, mode='nearest')
                    pred_mask_np = pred_mask_resized.squeeze().cpu().numpy().astype(np.uint8)
                    predicted_slices_list.append(pred_mask_np)

                    del img_tensor, output, probs, pred_mask_tensor, pred_mask_resized, pred_mask_np

                # Stack Slices & Save Result
                if predicted_slices_list:
                    full_pred_mask = np.stack(predicted_slices_list, axis=SLICE_AXIS)
                    print(f"  Stacked prediction shape: {full_pred_mask.shape}")
                    pred_nii = nib.Nifti1Image(full_pred_mask.astype(np.uint8), original_affine, original_header)
                    base_name = os.path.basename(img_path)
                    safe_base_name = re.sub(r'\.(nii|nii\.gz)$', '', base_name)
                    output_filename = os.path.join(PREDICTION_DIR, f"prediction-{safe_base_name}.nii.gz")
                    nib.save(pred_nii, output_filename)
                    processing_file_end_time = time.time()
                    print(f"  Saved prediction to: {output_filename} (Duration: {processing_file_end_time - processing_file_start_time:.2f} sec)")
                else: print("  No slices processed.")

                del img_nii, img_data_full, predicted_slices_list, full_pred_mask, pred_nii
                gc.collect();
                if device.type == 'cuda': torch.cuda.empty_cache()

            except FileNotFoundError: print(f"Error: Test image not found: {img_path}. Skipping.")
            except Exception as e: print(f"Error processing {os.path.basename(img_path)}: {type(e).__name__}: {e}")

    inference_end_time = time.time()
    print(f"\n--- Inference Finished --- Total Duration: {(inference_end_time - inference_start_time)/60:.2f} minutes")
    print(f"Predictions saved in: {PREDICTION_DIR}")

else:
    if inference_model is None: print("\nInference skipped: Model not loaded.")
    if not TEST_IMAGE_PATHS: print("\nInference skipped: No valid test images found or specified.")

print("\n--- End of Inference Script ---")
