# ==========================================
# Cell 7: Visualize Results (Expanded)    <<<--- REVISED CELL
# ==========================================
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import re
from skimage.measure import find_contours # For boundary plotting

print("\n--- Visualizing Sample Result (Expanded) ---")

# --- Configuration for Visualization ---
# (Same logic as before to find a default ID)
if 'TEST_IMAGE_PATHS' in locals() and TEST_IMAGE_PATHS: # Check if variable exists and is not empty
    try:
        processed_base_name = os.path.basename(TEST_IMAGE_PATHS[0])
        match = re.search(r'volume-(\d+)', processed_base_name)
        if match:
             DEFAULT_VIS_ID = match.group(1)
             print(f"Automatically selecting ID '{DEFAULT_VIS_ID}' for visualization.")
        else:
             DEFAULT_VIS_ID = '0'; print(f"Could not auto-detect ID, defaulting to '{DEFAULT_VIS_ID}'.")
    except IndexError:
        DEFAULT_VIS_ID = '0'; print(f"No processed images found, defaulting visualization ID to '{DEFAULT_VIS_ID}'.")
else:
    DEFAULT_VIS_ID = '0'
    print("Warning: TEST_IMAGE_PATHS not found or empty. Visualization attempt might fail.")


VOLUME_ID_TO_VISUALIZE = DEFAULT_VIS_ID # <<<--- Or manually set this ID, e.g., '45'
SLICE_INDEX_TO_VISUALIZE = None # None for middle slice, or set an integer

# --- Construct File Paths ---
# (Same file path logic as before)
original_vol_path = None
for ext in ['.nii', '.nii.gz']:
    path_check = os.path.join(TEST_DATA_DIR, f'volume-{VOLUME_ID_TO_VISUALIZE}{ext}')
    if os.path.exists(path_check): original_vol_path = path_check; break
prediction_mask_path = os.path.join(PREDICTION_DIR, f'prediction-volume-{VOLUME_ID_TO_VISUALIZE}.nii.gz')
ground_truth_mask_path = None
for ext in ['.nii', '.nii.gz']:
     path_check = os.path.join(TEST_DATA_DIR, f'segmentation-{VOLUME_ID_TO_VISUALIZE}{ext}')
     if os.path.exists(path_check): ground_truth_mask_path = path_check; break

# --- Check if files exist ---
original_exists = original_vol_path is not None and os.path.exists(original_vol_path)
prediction_exists = os.path.exists(prediction_mask_path)
ground_truth_exists = ground_truth_mask_path is not None and os.path.exists(ground_truth_mask_path)

# --- Load Data ---
if not (original_exists and prediction_exists):
    print(f"\nERROR: Cannot visualize. Missing essential files.")
    # ... (error messages as before) ...
else:
    print(f"\nVisualizing results for Volume ID: {VOLUME_ID_TO_VISUALIZE}")
    # ... (print file paths as before) ...

    try:
        img_nii = nib.load(original_vol_path)
        pred_nii = nib.load(prediction_mask_path)
        gt_nii = nib.load(ground_truth_mask_path) if ground_truth_exists else None

        img_data = img_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        gt_data = gt_nii.get_fdata() if gt_nii is not None else None

        # Determine slice index
        num_slices_in_vol = img_data.shape[SLICE_AXIS]
        if SLICE_INDEX_TO_VISUALIZE is None: slice_idx = num_slices_in_vol // 2
        elif 0 <= SLICE_INDEX_TO_VISUALIZE < num_slices_in_vol: slice_idx = SLICE_INDEX_TO_VISUALIZE
        else: slice_idx = num_slices_in_vol // 2; print(f"Warning: Invalid slice index. Using middle slice {slice_idx}.")
        print(f" Displaying Slice Index: {slice_idx}")

        # Extract the specific slice
        slice_obj = [slice(None)] * img_nii.ndim; slice_obj[SLICE_AXIS] = slice_idx
        img_slice = img_data[tuple(slice_obj)]
        pred_slice = pred_data[tuple(slice_obj)].astype(np.uint8) # Ensure uint8
        gt_slice = gt_data[tuple(slice_obj)] if gt_data is not None else None

        # Prepare specific masks needed
        gt_tumor_slice = (gt_slice == TARGET_LABEL).astype(np.uint8) if gt_slice is not None else None
        LIVER_LABEL = 1 # Standard LiTS label for liver
        gt_liver_slice = (gt_slice == LIVER_LABEL).astype(np.uint8) if gt_slice is not None else None

        # Check availability for conditional plotting
        has_gt_tumor = gt_tumor_slice is not None and np.any(gt_tumor_slice) # Check if GT tumor mask exists and is not empty
        has_gt_liver = gt_liver_slice is not None and np.any(gt_liver_slice) # Check if GT liver mask exists and is not empty

        print(f" Ground Truth Tumor Mask available: {has_gt_tumor}")
        print(f" Ground Truth Liver Mask available: {has_gt_liver}")

        # --- Plotting ---
        # Dynamically determine the number of plots
        plot_list = ['Original', 'Prediction'] # Always show these
        if has_gt_tumor: plot_list.extend(['GT Tumor', 'Pred Overlay', 'GT Overlay', 'Boundaries', 'Difference'])
        else: plot_list.extend(['Pred Overlay']) # Show prediction overlay even without GT
        if has_gt_liver: plot_list.extend(['Liver Context'])

        num_plots = len(plot_list)
        plt.figure(figsize=(5 * num_plots, 6)) # Adjust size

        plot_idx = 1
        # Plot Original
        if 'Original' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            ax.imshow(img_slice.T, cmap='gray', origin='lower')
            ax.set_title(f'Original Slice {slice_idx}'); ax.axis('off')

        # Plot Prediction Mask
        if 'Prediction' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            ax.imshow(pred_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
            ax.set_title(f'Predicted Mask'); ax.axis('off')

        # Plot GT Tumor Mask
        if 'GT Tumor' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            ax.imshow(gt_tumor_slice.T, cmap='hot', origin='lower', vmin=0, vmax=1)
            ax.set_title(f'GT Tumor (Label {TARGET_LABEL})'); ax.axis('off')

        # Plot Prediction Overlay
        if 'Pred Overlay' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            ax.imshow(img_slice.T, cmap='gray', origin='lower')
            ax.imshow(np.ma.masked_where(pred_slice == 0, pred_slice).T, cmap='viridis', alpha=0.6, origin='lower')
            ax.set_title(f'Prediction Overlay'); ax.axis('off')

        # Plot GT Overlay
        if 'GT Overlay' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            ax.imshow(img_slice.T, cmap='gray', origin='lower')
            ax.imshow(np.ma.masked_where(gt_tumor_slice == 0, gt_tumor_slice).T, cmap='hot', alpha=0.6, origin='lower')
            ax.set_title(f'GT Tumor Overlay'); ax.axis('off')

        # Plot Boundaries
        if 'Boundaries' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            ax.imshow(img_slice.T, cmap='gray', origin='lower')
            # Find and plot prediction contours
            pred_contours = find_contours(pred_slice, 0.5) # Find boundary for value > 0.5
            for contour in pred_contours:
                 ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='cyan', label='Prediction')
            # Find and plot GT contours
            gt_contours = find_contours(gt_tumor_slice, 0.5)
            for contour in gt_contours:
                 ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red', label='Ground Truth')
            ax.set_title('Boundaries'); ax.axis('off')
            # Add legend if contours were found
            handles, labels = ax.get_legend_handles_labels()
            if handles: # Only show legend if labels were added
                 # Create unique legend entries
                 by_label = dict(zip(labels, handles))
                 ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')


        # Plot Difference Map
        if 'Difference' in plot_list:
             ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
             # Difference: 1 = FN (GT only), -1 = FP (Pred only), 0 = TP/TN
             difference = gt_tumor_slice.astype(np.int8) - pred_slice.astype(np.int8)
             # Use a diverging colormap centered at 0
             # FN (1) = Red, TP/TN (0) = White, FP (-1) = Blue
             cmap_diff = plt.cm.coolwarm_r # Reversed coolwarm: Red=Hot(High), Blue=Cold(Low)
             img_diff = ax.imshow(difference.T, cmap=cmap_diff, origin='lower', vmin=-1, vmax=1)
             plt.colorbar(img_diff, ax=ax, ticks=[-1, 0, 1], label='Difference (GT - Pred)')
             ax.set_title('Difference Map\n(Red:FN, Blue:FP)'); ax.axis('off')


        # Plot Liver Context
        if 'Liver Context' in plot_list:
            ax = plt.subplot(1, num_plots, plot_idx); plot_idx+=1
            # Show original image masked by the liver
            img_liver_context = np.ma.masked_where(gt_liver_slice == 0, img_slice)
            ax.imshow(img_liver_context.T, cmap='gray', origin='lower')
            # Overlay Prediction within liver
            pred_in_liver = np.ma.masked_where(pred_slice == 0, pred_slice)
            ax.imshow(pred_in_liver.T, cmap='viridis', alpha=0.7, origin='lower')
            # Overlay GT Tumor within liver
            gt_tumor_in_liver = np.ma.masked_where(gt_tumor_slice == 0, gt_tumor_slice)
            ax.imshow(gt_tumor_in_liver.T, cmap='hot', alpha=0.5, origin='lower')
            # Optional: Add liver contour
            liver_contours = find_contours(gt_liver_slice, 0.5)
            for contour in liver_contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='yellow', linestyle=':')
            ax.set_title('Tumor in Liver Context\n(Pred:Green, GT:Red)'); ax.axis('off')


        plt.suptitle(f"Visualization for Volume ID {VOLUME_ID_TO_VISUALIZE}, Slice {slice_idx}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout further
        plt.show()

    except FileNotFoundError as e:
        print(f"\nError during visualization loading: {e}")
    except Exception as e:
        print(f"An error occurred during visualization plotting: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

print("\n--- End of Visualization ---")
