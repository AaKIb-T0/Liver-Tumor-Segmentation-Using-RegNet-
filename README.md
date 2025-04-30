# Liver-Tumor-Segmentation-Using-RegNet-
Deep learning project using RegNet for precise liver tumor segmentation on CT scans. Combines medical imaging and AI to detect tumors with accuracy and efficiency.

## Dataset
- Liver Tumor Segmentation dataset from Kaggle
- Accessed via Kaggle API in Colab

## Model
- RegNet architecture used as backbone
- Trained on preprocessed CT scan slices
Project: Automated Liver Tumor Segmentation using Deep Learning
This repository contains the code and documentation for a deep learning project focused on segmenting liver tumors from abdominal CT scans. The goal was to develop and train a model capable of automatically identifying and outlining tumor regions within NIfTI-formatted medical images.
1. Project Objective
To implement and evaluate a deep learning pipeline for semantic segmentation of liver tumors (specifically targeting Label 2 in the LiTS dataset convention) from 3D CT volumes.
2. Dataset
Source: LiTS - Liver Tumor Segmentation Challenge dataset hosted on Kaggle.
Format: NIfTI (.nii / .nii.gz) containing 3D volumetric CT scans (volume-ID.nii) and corresponding segmentation masks (segmentation-ID.nii).
Acquisition: The dataset was downloaded and extracted directly within the Google Colab environment using the Kaggle API.
Handling: Downloaded files (often split into parts on Kaggle) were consolidated into a single data directory (/content/lits_data/) for easier access.
3. Methodology & Model Architecture
Approach: A 2D slice-based segmentation approach was adopted for computational and memory efficiency, particularly suited for environments like Google Colab. The model processes individual axial slices.
Core Model: A custom segmentation architecture was implemented:
Encoder (Backbone): Utilized a RegNetY-400MF model pre-trained on ImageNet. This acts as an efficient feature extractor, leveraging transfer learning.
Decoder (Upsampling Path): Employed a series of ConvTranspose2d layers combined with Conv2d layers and ReLU activations to progressively upsample the features extracted by the encoder back to the original input slice resolution, generating the final segmentation map.
Libraries: Implemented using Python with the PyTorch deep learning framework.

4 Implementation Details
Data Loading: A custom PyTorch Dataset class was created to:
Identify corresponding volume/segmentation file pairs.
Load only the required 2D slice from the 3D NIfTI files during data fetching (__getitem__) to significantly reduce RAM usage (memory optimization).
Handle potential file loading errors gracefully.
Preprocessing:
Per-slice Min-Max intensity normalization (scaling pixel values to [0, 1]).
Resizing input slices to a fixed size (e.g., 128x128 pixels).
Applying ImageNet standard normalization (mean/std dev) as expected by the pre-trained RegNet encoder.
Resizing segmentation masks using nearest-neighbor interpolation.
Training:
Loss Function: Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
Optimizer: Adam.
Techniques: Automatic Mixed Precision (AMP) was utilized for potential speed improvements and reduced memory usage on compatible GPUs.
Environment: Training was performed on Google Colab, leveraging GPU acceleration.
Output: The trained model weights (state dictionary) were saved to a .pth file.
Inference:
The saved model weights were loaded into the defined architecture (set to evaluation mode).
Test 3D volumes were processed slice-by-slice. Each slice underwent the same preprocessing as during training.
Model predictions (logits) were converted to probabilities using Sigmoid, then thresholded (at 0.5) to create binary masks.
Predicted 2D slices were stacked to reconstruct the full 3D segmentation mask.
Results were saved as NIfTI files (.nii.gz), preserving the original image's spatial information (affine matrix and header).
5. Evaluation & Visualization
Primary Evaluation: Qualitative assessment through visualization.
Visualization Tools: Implemented using Matplotlib to display:
Original CT slice.
Predicted binary segmentation mask.
Ground Truth segmentation mask (tumor label extracted).
Overlays of the prediction/ground truth masks on the original slice.
Boundary contours of the prediction vs. ground truth overlaid on the original slice.
Difference map highlighting False Positives (FP) and False Negatives (FN).
Tumor prediction within the context of the segmented liver (if liver mask available in GT).
6. Project Structure & Usage
This project is structured into two main workflows, ideally run sequentially:
Training (Training_Notebook.ipynb): Handles data download, preprocessing setup, model definition, training loop execution, and saving the trained model weights (regnet_lits_tumor_segmentation_trained.pth).
Inference & Visualization (Inference_Visualization_Notebook.ipynb): Loads the pre-trained weights, defines test volumes, performs inference slice-by-slice, saves prediction masks, and generates the visualizations described above.
(Refer to the Setup and Usage sections below for detailed instructions on running the notebooks).
7. Status
The project successfully demonstrates:
Training of a RegNet-based segmentation model on the LiTS dataset within a resource-constrained environment (Colab).
Inference pipeline to generate 3D segmentation masks from the trained 2D model.
A suite of visualizations for qualitative assessment of the segmentation results.
