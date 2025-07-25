# Model Directory Overview

This directory contains all the files and scripts needed to prepare, convert, and understand the MobileNet model for use with the rv1103/rv1106 platform.

## Contents

- [`mobilenetv2-12.onnx`](./mobilenetv2-12.onnx): The original ONNX format MobileNetV2 model.
- [`convert_tflite.py`](./convert_tflite.py): Script to convert the ONNX model to TensorFlow Lite (TFLite) format.
- [`convert_rknn.py`](./convert_rknn.py): Script to convert the TFLite model to RKNN format for use on Rockchip NPU.
- [`create_calibration_dataset.py`](./create_calibration_dataset.py): Script to create a calibration dataset for quantizing the model.
- [`download_model.sh`](./download_model.sh): Shell script to download the original model file(s).
- [`synset.txt`](./synset.txt): List of class labels for the model's output.
- [`bell.jpg`](./bell.jpg): Example image for testing or demonstration.
- [`README_conversion.md`](./README_conversion.md): Detailed documentation about the model conversion process.

## How to Use Each Script

### 1. Download the Model
If you do not already have the ONNX model, run:
```sh
./download_model.sh
```
This will download the required model file(s) into this directory.

### 2. TFLITE model
Where you get the .tflite model (in this case it is a pre-trained model by Keras)
```sh
python3 tflite_model.py
```
This will generate a `.tflite`.

### 3. Create Calibration Dataset (Optional, for Quantization)
If you need to quantize the model, you may need a calibration dataset. Generate it with:
```sh
python3 create_calibration_dataset.py
```
This will create a set of images or data used for calibration during quantization.

### 4. Convert TFLite to RKNN
To convert the TFLite model to RKNN format for the Rockchip NPU, run:
```sh
python3 convert_rknn.py
```
This will produce a `.rknn` file ready for deployment.

### 5. Other Files
- [`synset.txt`](./synset.txt): Used by inference scripts to map model outputs to human-readable class names.
- [`bell.jpg`](./bell.jpg): Example image for testing the model or conversion scripts.
- [`README_conversion.md`](./README_conversion.md): For more detailed, step-by-step conversion instructions and troubleshooting.

---

**Tip:** Always check the comments and documentation inside each script for additional options or requirements.
