# -*- coding: utf-8 -*-
"""
Liver Tumor Segmentation with RegNet on LiTS Dataset (Colab - Kaggle API Direct) - v4.2 TRAINING SCRIPT (Forced Download)

Downloads LiTS data via Kaggle API (forcing download), consolidates files,
includes fixes for dtype errors, safe variable deletion, and optimized
DataLoader memory usage. Trains a segmentation model and saves it.
"""

# ==========================================
# Cell 1: Install Necessary Libraries
# ==========================================
!pip install nibabel kaggle --quiet
print("Libraries checked/installed.")

# ==========================================
# Cell 2: Import Libraries
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
# Correct import path for autocast/GradScaler may vary by PyTorch version
try:
    from torch.cuda.amp import GradScaler, autocast
    print("Using torch.cuda.amp for GradScaler/autocast.")
except ImportError:
    try:
        from torch.amp import GradScaler, autocast # Preferred for newer PyTorch >= 1.6
        print("Using torch.amp for GradScaler/autocast.")
    except ImportError:
        print("Warning: Could not import GradScaler/autocast. Mixed precision (AMP) might be disabled.")
        USE_AMP = False # Ensure AMP is disabled if import fails

import nibabel as nib
import os
import numpy as np
from PIL import Image
import time
import gc
import re
from google.colab import files

print(f"PyTorch Version: {torch.__version__}")
print(f"Nibabel Version: {nib.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Cell 3: Setup Kaggle API
# ==========================================
print("Please upload your Kaggle API token (`kaggle.json`)")
try:
    # Ensure kaggle.json doesn't exist from a previous run in the same session
    if os.path.exists('kaggle.json'):
        os.remove('kaggle.json')
    uploaded = files.upload()
    if 'kaggle.json' in uploaded:
        print("kaggle.json uploaded successfully.")
        !mkdir -p ~/.kaggle
        !cp kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json
        print("Kaggle API configured.")
    else:
         print("Upload failed or cancelled.")
except Exception as e:
    print(f"Error during Kaggle API setup: {e}")

# ==========================================
# Cell 4: Download Dataset via Kaggle API (Forced)   <<<--- MODIFIED HERE
# ==========================================
print("\nDownloading and unzipping the LiTS dataset from Kaggle...")
# Clean up potential leftover directories from previous runs first
!rm -rf /content/segmentations /content/volume_pt* /content/lits_data /content/*.zip # Also remove old zip
# Download command with --force flag added
!kaggle datasets download -d andrewmvd/liver-tumor-segmentation -p /content/ --unzip --force
print("Dataset download and unzip complete (forced).")

# ==========================================
# Cell 5: Consolidate Data Files
# ==========================================
print("\nConsolidating downloaded files into a single directory...")
# Create the target directory
!mkdir -p /content/lits_data # Use -p to avoid error if it somehow exists

# Check if source directories exist after forced unzip
# NOTE: The exact structure from Kaggle might change. Check '/content/' if these fail.
segmentations_exist = os.path.isdir("/content/segmentations")
volumes_exist = any(os.path.isdir(f"/content/volume_pt{i}") for i in range(1, 10)) # Check common range

if segmentations_exist:
    print("Moving segmentation files...")
    !mv /content/segmentations/*.nii* /content/lits_data/
else:
    print("Warning: /content/segmentations directory not found after download.")
    # Add a check for files directly in /content if needed
    # if ls /content/segmentation-*.nii* > /dev/null 2>&1; then ...

if volumes_exist:
    print("Moving volume files...")
    !mv /content/volume_pt*/*.nii* /content/lits_data/
else:
     print("Warning: /content/volume_pt* directories not found after download.")
     # Add a check for files directly in /content if needed
     # if ls /content/volume-*.nii* > /dev/null 2>&1; then ...

print("File consolidation attempt complete.")

# Verify consolidation
print("\nChecking contents of the new data folder '/content/lits_data/':")
if os.path.isdir("/content/lits_data"):
    !ls -lh /content/lits_data/ | head # Show first few files and sizes
    print("\nTotal number of NIfTI files in consolidation folder:")
    !ls /content/lits_data/*.nii* 2>/dev/null | wc -l # Count files, ignore error if none
else:
    print("Error: Consolidation directory '/content/lits_data' was not created or files failed to move.")

# ==========================================
# Cell 6: Configuration (DATA_DIR is now fixed)
# ==========================================

DATA_DIR = '/content/lits_data/'
MODEL_SAVE_PATH = '/content/regnet_lits_tumor_segmentation_trained.pth' # Keep the filename for the testing script
NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
RESIZE_SIZE = (128, 128)
NUM_WORKERS = 2 # Start with 2, reduce to 1 or 0 if RAM errors occur
USE_AMP = torch.cuda.is_available() and 'GradScaler' in globals()
TARGET_LABEL = 2
SLICE_AXIS = 2

print("Configuration set.")
print(f"Using data directory: {DATA_DIR}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")
if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR): # Also check if directory is empty
    print(f"--- CRITICAL WARNING ---")
    print(f"The data directory '{DATA_DIR}' does not exist OR IS EMPTY after consolidation attempt.")
    print(f"Check the output of Cell 4 and Cell 5 for download/move errors.")
    print(f"Training cannot proceed without data.")
    print(f"----------------------")
else:
    print(f"Directory '{DATA_DIR}' found and is not empty.")
print(f"Using TARGET_LABEL = {TARGET_LABEL}")
print(f"Using SLICE_AXIS = {SLICE_AXIS}")
print(f"Using NUM_WORKERS = {NUM_WORKERS}")
print(f"Using AMP: {USE_AMP}")

# ==========================================
# Cell 7: Dataset Class Definition (Memory Optimized)
# ==========================================
class LiverTumorDataset(Dataset):
    # (Dataset class definition remains unchanged)
    # ... (rest of the class definition as provided previously) ...
    def __init__(self, data_dir, transform=None, mask_transform=None, slice_axis=2, target_label=2):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.slice_axis = slice_axis
        self.target_label = target_label
        self.file_pairs = []
        if not os.path.isdir(self.data_dir):
             print(f"Error: Data directory '{self.data_dir}' not found during Dataset init.")
             return

        print(f"Scanning '{self.data_dir}' for image/segmentation pairs...")
        all_files = os.listdir(data_dir)
        found_masks = {}
        potential_images = {}
        # Make sure scan is case-insensitive if needed, but standard names are lowercase
        for filename in all_files:
            if not (filename.endswith('.nii') or filename.endswith('.nii.gz')): continue
            # Case-insensitive matching just in case: re.IGNORECASE
            mask_match = re.search(r'segmentation-(\d+)\.(nii|nii\.gz)', filename, re.IGNORECASE)
            if mask_match: found_masks[mask_match.group(1)] = os.path.join(self.data_dir, filename)
            img_match_vol = re.search(r'volume-(\d+)\.(nii|nii\.gz)', filename, re.IGNORECASE)
            if img_match_vol: potential_images[img_match_vol.group(1)] = os.path.join(self.data_dir, filename)

        print(f"Found {len(found_masks)} potential masks, {len(potential_images)} potential images.")
        missing_count = 0
        for file_id, mask_path in found_masks.items():
            if file_id in potential_images:
                self.file_pairs.append((potential_images[file_id], mask_path))
            else:
                if missing_count < 5: print(f"  - Warn: No matching image for mask ID {file_id}")
                elif missing_count == 5: print("  - (Further missing image warnings suppressed)")
                missing_count += 1
        if not self.file_pairs:
            print(f"\n--- ERROR: No valid image/mask pairs found in '{self.data_dir}'. ---")
            print(f"--- Please check file names in '{self.data_dir}' match 'volume-ID.nii' and 'segmentation-ID.nii' pattern. ---")
        else:
            print(f"Successfully found {len(self.file_pairs)} image/mask pairs.")

    def __len__(self): return len(self.file_pairs)
    def __getitem__(self, idx):
        # (__getitem__ logic remains unchanged)
        # ... (rest of the __getitem__ method as provided previously) ...
        if idx >= len(self.file_pairs): raise IndexError("Dataset index out of range.")
        image_path, mask_path = self.file_pairs[idx]
        try:
            img_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)
            current_slice_axis = self.slice_axis
            if not (0 <= current_slice_axis < img_nii.ndim): current_slice_axis = 2
            if current_slice_axis >= img_nii.ndim: return None, None

            slice_idx = img_nii.shape[current_slice_axis] // 2
            slice_obj = [slice(None)] * img_nii.ndim
            slice_obj[current_slice_axis] = slice_idx

            img_slice = img_nii.dataobj[tuple(slice_obj)].astype(np.float32)
            mask_slice = mask_nii.dataobj[tuple(slice_obj)].astype(np.int16)

            min_val, max_val = np.min(img_slice), np.max(img_slice)
            if max_val > min_val: img_slice = (img_slice - min_val) / (max_val - min_val)
            else: img_slice = np.zeros_like(img_slice)
            binary_mask_slice = (mask_slice == self.target_label).astype(np.float32)

            img_pil = Image.fromarray(np.uint8(img_slice * 255)).convert("RGB")
            mask_pil = Image.fromarray(np.uint8(binary_mask_slice * 255)).convert("L")

            img_tensor = self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil)
            mask_tensor = self.mask_transform(mask_pil) if self.mask_transform else transforms.ToTensor()(mask_pil)
            mask_tensor = mask_tensor.float()

            del img_nii, mask_nii, img_slice, mask_slice, binary_mask_slice, img_pil, mask_pil
            # gc.collect() # Collecting garbage frequently in getitem can slow things down
            return img_tensor, mask_tensor
        except Exception as e:
            print(f"Error processing pair idx={idx}, Img='{os.path.basename(image_path)}': {type(e).__name__}, {e}")
            return None, None
print("LiverTumorDataset class defined.")


# ==========================================
# Cell 8: Transforms and DataLoader Setup
# ==========================================
# (Remains unchanged)
img_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
mask_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),])
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
    if not batch: return torch.Tensor(), torch.Tensor()
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: print(f"Collation Error: {e}"); return torch.Tensor(), torch.Tensor()

print("\nCreating Dataset and DataLoader...")
dataset = None; dataloader = None
if os.path.isdir(DATA_DIR) and os.listdir(DATA_DIR): # Check directory exists AND is not empty
    try: dataset = LiverTumorDataset(DATA_DIR, img_transform, mask_transform, SLICE_AXIS, TARGET_LABEL)
    except Exception as e: print(f"Dataset creation failed: {e}")

    if dataset and len(dataset) > 0: # Check dataset object created AND found pairs
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=(device.type == 'cuda'), collate_fn=collate_fn,
                                persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2 if NUM_WORKERS > 0 else None)
        print(f"DataLoader created. Size: {len(dataset)}")
    elif dataset is None:
        print("Dataset object could not be created (check previous errors).")
    else: # dataset exists but len is 0
        print("Dataset created, but found 0 valid pairs (check file names/patterns in Cell 7).")
else:
    print(f"Dataset directory '{DATA_DIR}' not found or is empty. Cannot create DataLoader.")


# ==========================================
# Cell 9: Model Definition (SegNetRegNet)
# ==========================================
# (Remains unchanged)
class SegNetRegNet(nn.Module):
    # ... (rest of the class definition as provided previously) ...
    def __init__(self, num_classes=1):
        super(SegNetRegNet, self).__init__()
        try:
             weights = models.RegNet_Y_400MF_Weights.IMAGENET1K_V1
             full_backbone = models.regnet_y_400mf(weights=weights)
             print("Using RegNet_Y_400MF_Weights.IMAGENET1K_V1")
        except AttributeError:
             full_backbone = models.regnet_y_400mf(pretrained=True)
             print("Using legacy pretrained=True for RegNet")
        self.stem = full_backbone.stem
        self.trunk_output = full_backbone.trunk_output
        encoder_channels = 440
        print(f"Using known Encoder output channels (from trunk_output): {encoder_channels}")
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
print("CORRECTED SegNet based ---- > RegNet model class defined.")

# ==========================================
# Cell 10: Training Setup
# ==========================================
# (Remains unchanged)
model = None; optimizer = None; criterion = None; scaler = None
# Proceed only if dataloader was successfully created
if dataloader is not None:
    print("\nInitializing model, optimizer, loss, and scaler...")
    try:
        model = SegNetRegNet(num_classes=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        use_scaler = USE_AMP and device.type == 'cuda'
        if use_scaler:
            try: scaler = torch.amp.GradScaler('cuda', enabled=True)
            except (AttributeError, TypeError): scaler = GradScaler(enabled=True)
            print("Using AMP GradScaler.")
        else: scaler=None; print("AMP/Scaler disabled.")
        print("Training setup complete.")
    except Exception as e: print(f"Error during training setup: {e}"); model = None
else: print("\nSkipping training setup: DataLoader not available.")


# ==========================================
# Cell 11: Training Loop
# ==========================================
# (Remains unchanged)
training_successful = False
if model and optimizer and criterion and dataloader: # Check all prerequisites
    print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")
    start_time_total = time.time()
    autocast_device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0; batches_processed = 0; epoch_start_time = time.time()
        # Use tqdm for the dataloader iterator for progress bar per epoch
        # batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        batch_iterator = iter(dataloader) # Keep simple iterator if tqdm causes issues

        for i in range(len(dataloader)): # Iterate based on dataloader length
            images, masks = None, None; outputs, loss = None, None; batch = None
            try:
                batch = next(batch_iterator)
                # (Rest of the training loop logic remains the same)
                # ...
                images, masks = batch
                if images is None or masks is None or images.numel() == 0 or masks.numel() == 0: continue
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=autocast_device_type, dtype=torch.float16 if autocast_device_type=='cuda' else torch.bfloat16, enabled=USE_AMP):
                     outputs = model(images)
                     if outputs.shape[2:] != masks.shape[2:]:
                         outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                     loss = criterion(outputs, masks)
                if not torch.isfinite(loss):
                    print(f"Warn: Non-finite loss epoch {epoch+1}, batch {i+1}. Skip step."); continue
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item(); batches_processed += 1

            except StopIteration: print(f"Warn: Dataloader stopped early epoch {epoch+1}."); break
            except RuntimeError as e:
                 print(f"Runtime Error epoch {epoch+1}, batch {i+1}: {e}")
                 if "CUDA out of memory" in str(e): print("TIP: Reduce BATCH_SIZE."); raise e
                 elif "DataLoader worker" in str(e): print(f"TIP: Reduce NUM_WORKERS (current: {NUM_WORKERS})."); raise e
                 else: pass
            except Exception as e: print(f"General Error epoch {epoch+1}, batch {i+1}: {type(e).__name__}: {e}"); break
            finally: del images, masks, outputs, loss, batch

        epoch_duration = time.time() - epoch_start_time
        avg_loss = running_loss / batches_processed if batches_processed > 0 else 0
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Avg Loss: {avg_loss:.5f}, Batches: {batches_processed}/{len(dataloader)}, Duration: {epoch_duration:.2f} sec")
        gc.collect();
        if device.type == 'cuda': torch.cuda.empty_cache()

    end_time_total = time.time()
    print(f"\n--- Training Finished --- Total Duration: {(end_time_total - start_time_total)/60:.2f} minutes")
    training_successful = True

else: print("\nTraining loop skipped: prerequisites not met.")

# ==========================================
# Cell 12: Save Model Weights
# ==========================================
# (Remains unchanged)
if training_successful and model is not None:
    print(f"\nSaving model state dictionary to {MODEL_SAVE_PATH}...")
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model state_dict saved successfully to '{MODEL_SAVE_PATH}'.")
    except Exception as e: print(f"Error saving model: {e}")
elif model is not None: print("\nSkipping model saving: Training did not complete successfully.")
else: print("\nSkipping model saving: Model not initialized.")


# ==========================================
# Cell 13: Download the Saved Model (Optional)
# ==========================================
# (Remains unchanged)
if os.path.exists(MODEL_SAVE_PATH):
    print(f"\nAttempting to initiate download of '{os.path.basename(MODEL_SAVE_PATH)}'...")
    try:
        files.download(MODEL_SAVE_PATH)
        print("Model download initiated. Check your browser downloads.")
    except Exception as e:
        print(f"Could not initiate download via files.download: {e}")
        print(f"You can download the file manually from the Colab file browser on the left panel: '{MODEL_SAVE_PATH}'")
elif training_successful and model is not None:
     print(f"\nModel file '{MODEL_SAVE_PATH}' not found (saving might have failed). Cannot download.")
else:
     print("\nModel download skipped (training did not run or failed).")

print("\n--- End of Training Script ---")
