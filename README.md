
# Image Classification - Street View House Number (SVHN) using a Custom CNN

<div style="max-width: 700px; line-height: 1.5; text-align: justify;">

This repository contains the project for multiclass image classification of digits (0-9) from the Stanford Street View House Number (SVHN) dataset. The project showcases a complete machine learning pipeline, from data acquisition and preprocessing to model training and evaluation using a custom Convolutional Neural Network (CNN).

The accompanying PDF document, `MAPrinsloo - SVHN Image Classification CNN.pdf`, provides a detailed explanation of the project's methodology, analysis, and results.

## Project Overview

The primary goal of this project is to develop and fine-tune a custom CNN capable of accurately classifying house numbers. This includes:

*   **Data Sourcing:** Utilizing the Stanford SVHN (cropped_digits) dataset.
*   **Exploratory Data Analysis (EDA):** Initial data exploration using Apache Spark.
*   **Image Preprocessing:** Implementation of a custom pipeline for image quality filtering, resizing, and sharpening.
*   **Model Development:** Creation and training of custom CNN architectures.
*   **Performance Evaluation:** Assessment of model performance using various metrics.

## Repository Contents

*   `MAPrinsloo_SVHN-CNN.ipynb`: The main Jupyter Notebook containing the full project pipeline. This file includes executed code cells and their outputs, showcasing the workflow.
*   `MAPrinsloo - SVHN Image Classification CNN.pdf`: The supporting project document detailing the methodology, analysis, and interpretation of results.
*   `environment.tf-gpu-wsl.yml`: The Conda environment file used during the production of this project.
*   `SaveRawImagesToDisk.py`: A Python script used to decode the raw parquet dataset files and save them as individual PNG images.
*   `Fine_Tune_Sharpening_Docs/`: Directory that contains text files documenting the quantitative results of the sharpening parameter fine-tuning.
*   `CustomCNN_model_0_0/`: Found in the release assets - contains the saved model checkpoints and TensorBoard logs for Model Attempt 1 (trained on unsharpened images).
*   `CustomCNN_model_15_2/`: Found in the release assets - contains the saved model checkpoints and TensorBoard logs for Model Attempt 2 (trained on sharpened images).

**Note on Large Files:**
Pre-trained model files (`.h5`) are often too large for direct storage within a standard Git repository (GitHub's limit is 100MB per file). These models will be provided as **release assets** on the [Releases page](https://github.com/MAPrinsloo/SVHN_CNN_Image_Classificaiton/releases) of this repository **shortly after the initial project release**. Once available, you can download the zipped model folders from there and extract them into the project's root directory to use them with the notebook. In the meantime, you can regenerate the models by running the notebook's training cells.

## Getting Started

Follow these steps to set up the project environment and run the analysis:

### 1. Clone the Repository

```bash
git clone https://github.com/MAPrinsloo/SVHN_CNN_Image_Classificaiton
cd SVHN_CNN_Image_Classificaiton
```

### 2. Set Up the Conda Environment

Ensure you have Miniconda or Anaconda installed.
```bash
conda env create -f environment.tf-gpu-wsl.yml
conda activate tf-gpu-wsl
```

### 3. Download the Dataset

The raw SVHN `cropped_digits` dataset is required. Download the parquet files (`extra-00000-of-00002.parquet` and `extra-00001-of-00002.parquet`) from [Hugging Face - ufldl-stanford/svhn](https://huggingface.co/datasets/ufldl-stanford/svhn).
Place these two `.parquet` files directly into the **root directory** of your cloned project.

### 4. Prepare Raw Images

Execute the `SaveRawImagesToDisk.py` script. This will process the downloaded parquet files and save individual PNG images into a new directory named `saved_images_by_class/` within your project root.
```bash
python SaveRawImagesToDisk.py
```

### 5. Run the Jupyter Notebook

Launch Jupyter Notebook and open `MAPrinsloo_SVHN-CNN.ipynb`.
```bash
jupyter notebook
```
Execute all cells in the notebook sequentially. This will perform the image quality filtering, preprocessing, data splitting, model training, and evaluation for both attempts. The notebook includes saved outputs from a complete execution.

### 6. (Optional) View TensorBoard Logs

To visualize the training progress and model architectures, use TensorBoard. Open a new terminal (while your `tf-gpu-wsl` environment is active) and run:
```bash
# For Model Attempt 1 (unsharpened data)
tensorboard --logdir CustomCNN_model_0_0/logs

# For Model Attempt 2 (sharpened data)
tensorboard --logdir CustomCNN_model_15_2/logs
```
Access TensorBoard by navigating to the URL provided in the terminal output.

## Results Overview

The project demonstrates the successful classification of the SVHN digits. A discussion can be found in the accompanying PDF document, `MAPrinsloo - SVHN Image Classification CNN.pdf`, regarding the results of the following models:

*   **Model Attempt 1 (Upscaled Only):** Achieved a test accuracy of approximately 94.7%.
*   **Model Attempt 2 (Upscaled & Sharpened):** Achieved a test accuracy of approximately 94.8%, showing a marginal improvement and better handling of visually ambiguous digits.

Classification reports and confusion matrices are available within the executed notebook and the supporting PDF document.

## Author & Contact

*   **Author:** Matthew Prinsloo
*   **LinkedIn:** [https://www.linkedin.com/in/matthew-prinsloo-b01b262b7/](https://www.linkedin.com/in/matthew-prinsloo-b01b262b7/)

</div>

---
