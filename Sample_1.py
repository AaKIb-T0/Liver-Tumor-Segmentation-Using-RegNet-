# ==========================================
# Cell 7: Visualize Results             <<<--- NEW CELL
# ==========================================
import matplotlib.pyplot as plt
import numpy as np # Make sure numpy is imported (usually done in Cell 2)
import nibabel as nib # Make sure nibabel is imported (usually done in Cell 2)
import os # Make sure os is imported (usually done in Cell 2)

print("\n--- Visualizing Sample Result ---")

# --- Configuration for Visualization ---
# Choose one of the volume IDs that was *successfully* processed in Cell 6
# Check the output of Cell 6 to see which predictions were saved.
if TEST_IMAGE_PATHS: # Check if any images were actually processed
    try:
        processed_base_name = os.path.basename(TEST_IMAGE_PATHS[0]) # Get the first processed image path
        match = re.search(r'volume-(\d+)', processed_base_name)
        if match:
             DEFAULT_VIS_ID = match.group(1)
             print(f"Automatically selecting ID '{DEFAULT_VIS_ID}' for visualization (first processed volume).")
        else:
             DEFAULT_VIS_ID = '0' # Fallback if regex fails
             print(f"Could not auto-detect ID from path, defaulting to '{DEFAULT_VIS_ID}'.")
    except IndexError:
        DEFAULT_VIS_ID = '0' # Fallback if TEST_IMAGE_PATHS was empty somehow after check
        print(f"No processed images found, defaulting visualization ID to '{DEFAULT_VIS_ID}'.")

else:
    DEFAULT_VIS_ID = '0'
    print("No images were processed in the previous step. Visualization attempt might fail.")


VOLUME_ID_TO_VISUALIZE = DEFAULT_VIS_ID # <<<--- Or manually set this ID, e.g., '45'
SLICE_INDEX_TO_VISUALIZE = None # Set to None to pick middle slice, or set an integer (e.g., 40)

# --- Construct File Paths (Handles .nii or .nii.gz for input) ---
original_vol_path = None
for ext in ['.nii', '.nii.gz']:
    path_check = os.path.join(TEST_DATA_DIR, f'volume-{VOLUME_ID_TO_VISUALIZE}{ext}')
    if os.path.exists(path_check):
        original_vol_path = path_check
        break

prediction_mask_path = os.path.join(PREDICTION_DIR, f'prediction-volume-{VOLUME_ID_TO_VISUALIZE}.nii.gz') # Prediction saved as .nii.gz

ground_truth_mask_path = None
for ext in ['.nii', '.nii.gz']:
     path_check = os.path.join(TEST_DATA_DIR, f'segmentation-{VOLUME_ID_TO_VISUALIZE}{ext}')
     if os.path.exists(path_check):
        ground_truth_mask_path = path_check
        break

# --- Check if files exist ---
original_exists = original_vol_path is not None and os.path.exists(original_vol_path)
prediction_exists = os.path.exists(prediction_mask_path)
ground_truth_exists = ground_truth_mask_path is not None and os.path.exists(ground_truth_mask_path)

if not (original_exists and prediction_exists):
    print(f"\nERROR: Cannot visualize.")
    if not original_exists: print(f" - Original volume not found for ID {VOLUME_ID_TO_VISUALIZE} in {TEST_DATA_DIR}")
    if not prediction_exists: print(f" - Prediction mask not found: {prediction_mask_path}")
    print(f" - Please check VOLUME_ID_TO_VISUALIZE and ensure inference ran successfully for this ID.")
else:
    print(f"\nVisualizing results for Volume ID: {VOLUME_ID_TO_VISUALIZE}")
    print(f" Original Volume: {os.path.basename(original_vol_path)}")
    print(f" Prediction Mask: {os.path.basename(prediction_mask_path)}")
    if ground_truth_exists:
        print(f" Ground Truth Mask: {os.path.basename(ground_truth_mask_path)}")
    else:
        print(f" Warning: Ground truth mask not found for ID {VOLUME_ID_TO_VISUALIZE}. Visualization will omit ground truth.")

    try:
        # Load the necessary volumes
        img_nii = nib.load(original_vol_path)
        pred_nii = nib.load(prediction_mask_path)
        gt_nii = nib.load(ground_truth_mask_path) if ground_truth_exists else None

        # Get data arrays
        img_data = img_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        # Ensure gt_data is loaded only if gt_nii exists
        gt_data = gt_nii.get_fdata() if gt_nii is not None else None

        # Determine slice index
        num_slices_in_vol = img_data.shape[SLICE_AXIS]
        if SLICE_INDEX_TO_VISUALIZE is None:
            slice_idx = num_slices_in_vol // 2 # Default to middle slice
        elif 0 <= SLICE_INDEX_TO_VISUALIZE < num_slices_in_vol:
            slice_idx = SLICE_INDEX_TO_VISUALIZE
        else:
            print(f"Warning: Invalid SLICE_INDEX_TO_VISUALIZE ({SLICE_INDEX_TO_VISUALIZE}). Using middle slice {num_slices_in_vol // 2}.")
            slice_idx = num_slices_in_vol // 2

        print(f" Displaying Slice Index: {slice_idx}")

        # Extract the specific slice using the SLICE_AXIS defined earlier
        slice_obj = [slice(None)] * img_nii.ndim
        slice_obj[SLICE_AXIS] = slice_idx

        img_slice = img_data[tuple(slice_obj)]
        pred_slice = pred_data[tuple(slice_obj)]
        # Extract ground truth slice only if gt_data was loaded
        gt_slice = gt_data[tuple(slice_obj)] if gt_data is not None else None
        # Get the specific tumor mask from GT using TARGET_LABEL (handle case where gt_slice is None)
        gt_tumor_slice = (gt_slice == TARGET_LABEL).astype(np.uint8) if gt_slice is not None else None

        # --- Plotting ---
        # Determine number of plots needed based on ground truth availability
        has_gt = gt_tumor_slice is not None
        num_base_plots = 3 if has_gt else 2  # Img, Pred, (GT)
        num_overlay_plots = 2 if has_gt else 1 # PredOverlay, (GTOverlay)
        total_plots = num_base_plots + num_overlay_plots

        plt.figure(figsize=(5 * total_plots, 5.5)) # Adjust figure size for titles
        plot_index = 1 # Counter for subplot position

        # 1. Original Image Slice
        plt.subplot(1, total_plots, plot_index)
        plt.imshow(img_slice.T, cmap='gray', origin='lower') # Use .T for correct orientation
        plt.title(f'Original Slice {slice_idx}')
        plt.axis('off')
        plot_index += 1

        # 2. Predicted Mask Slice (Tumor = 1 after thresholding)
        plt.subplot(1, total_plots, plot_index)
        plt.imshow(pred_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=1) # Display binary mask
        plt.title(f'Predicted Mask')
        plt.axis('off')
        plot_index += 1

        # 3. Ground Truth Mask Slice (if available)
        if has_gt:
            plt.subplot(1, total_plots, plot_index)
            plt.imshow(gt_tumor_slice.T, cmap='hot', origin='lower', vmin=0, vmax=1) # Display binary GT mask
            plt.title(f'Ground Truth (Label={TARGET_LABEL})')
            plt.axis('off')
            plot_index += 1

        # 4. Overlay: Prediction on Original
        plt.subplot(1, total_plots, plot_index)
        plt.imshow(img_slice.T, cmap='gray', origin='lower')
        # Overlay the prediction mask where it's not zero (value is 1)
        plt.imshow(np.ma.masked_where(pred_slice == 0, pred_slice).T, cmap='viridis', alpha=0.5, origin='lower')
        plt.title(f'Prediction Overlay')
        plt.axis('off')
        plot_index += 1

        # 5. Overlay: Ground Truth on Original (if available)
        if has_gt:
             plt.subplot(1, total_plots, plot_index)
             plt.imshow(img_slice.T, cmap='gray', origin='lower')
             # Overlay the ground truth tumor mask where it's not zero (value is 1)
             plt.imshow(np.ma.masked_where(gt_tumor_slice == 0, gt_tumor_slice).T, cmap='hot', alpha=0.5, origin='lower')
             plt.title(f'Ground Truth Overlay')
             plt.axis('off')
             plot_index += 1

        plt.suptitle(f"Visualization for Volume ID {VOLUME_ID_TO_VISUALIZE}, Slice {slice_idx}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    except FileNotFoundError as e:
        print(f"\nError during visualization loading: {e}")
        print("Please ensure the specified volume ID and corresponding files exist.")
    except Exception as e:
        print(f"An error occurred during visualization plotting: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

print("\n--- End of Visualization ---")
