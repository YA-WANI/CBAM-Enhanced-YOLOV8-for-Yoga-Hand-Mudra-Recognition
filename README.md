# CBAM-Enhanced-YOLOV8-for-Yoga-Hand-Mudra-Recognition
This project implements a CBAM (Convolutional Block Attention Module)-enhanced YOLOv8 model for the task of hand mudra detection. The entire experiment, from data preparation to visualization, has been modularized into a series of Python scripts for clarity and ease of use.

**Project Structure**
**data_preparation.py**: This script is responsible for preparing and splitting the dataset into training, validation, and testing sets. It also generates the data.yaml configuration file required by Ultralytics.

**cbam_enhancement.py**: Defines the custom CBAM layer and registers it with the Ultralytics framework. It also includes the logic for building the dynamic YOLOv8 model architecture YAML file with the CBAM layer integrated into the backbone.

**training.py**: The main script for training the CBAM-enhanced YOLOv8 model using the specified hyperparameters. It utilizes the model definition from cbam_enhancement.py and the data configuration from data.yaml.

**testing.py**: This script is used to evaluate the performance of the trained model on a held-out test dataset, providing key metrics like mAP.

**visualization.py**: Provides a powerful tool to visualize the model's performance. It generates and overlays attention heatmaps from the CBAM layer onto images, helping you understand where the model is focusing. It also displays the final detections.

**wandb_sweep.py**: This file contains the configuration and logic for performing a Weights & Biases (W&B) sweep to find the optimal hyperparameters for the model.

**Prerequisites**
Before running the scripts, ensure you have the necessary libraries installed. It is highly recommended to use a virtual environment.

pip install torch torchvision torchaudio ultralytics wandb opencv-python matplotlib numpy scikit-learn

**Usage**
**Step 1: Prepare the Dataset**
First, you need to prepare your data. Make sure your dataset is in the correct YOLO format (images and corresponding .txt label files).

Run the data preparation script to split the data and generate the data.yaml file.

python data_preparation.py

**Step 2: Train the Model**
Next, you will train the model. This script will automatically generate the model YAML file with the CBAM layer and start the training process.

python training.py

**Step 3: Test the Model**
Once training is complete, you can evaluate the model's performance on the test set.

python testing.py

**Step 4: Visualize Results**
This script will generate attention heatmaps and prediction overlays for a set of unseen images, providing valuable insights into the model's focus. Note: You may need to update the model_cbam path and unseen_images_dir in the script.

python visualization.py

**Step 5 (Optional): Hyperparameter Tuning**
To find the best hyperparameters, you can use the Weights & Biases sweep functionality. First, set up the sweep:

python wandb_sweep.py

This will output a command to run the sweep agent. Copy and paste that command into your terminal to start the tuning process.

wandb agent [SWEEP_ID]

**Customization**
Paths: You will need to update the file paths for your dataset in data_preparation.py and visualization.py.

Hyperparameters: All training hyperparameters, such as epochs, batch size, and learning rate, can be adjusted in training.py.

Model Variants: The training.py script is set to train the yolov8n (nano) variant by default. You can change this to s, m, l, or x by modifying the variants list.

CBAM Layer Position: The target_layer in visualization.py is hardcoded to index 8. If you change the model architecture, you may need to adjust this index to point to the correct CBAM layer.
