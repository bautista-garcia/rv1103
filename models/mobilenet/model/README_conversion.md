# MobileNet Model Conversion Guide

This guide explains how to convert MobileNet models from PyTorch (.pt) or ONNX (.onnx) format to RKNN format for deployment on Rockchip NPU platforms.

## Prerequisites

- Python 3.6+
- RKNN Toolkit 2
- PyTorch (for .pt model conversion)
- ONNX (for .onnx model conversion)

## Quick Start

### 1. Basic ONNX Conversion

```bash
cd mobilenet/model
python3 convert.py --model mobilenetv2-12.onnx --target rv1103
```

This will create `mobilenet.rknn` in the current directory.

### 2. PyTorch Model Conversion

```bash
python3 convert.py --model mobilenet_v2.pt --target rv1103 --output mobilenet_v2.rknn
```

### 3. Float Precision Conversion

```bash
python3 convert.py --model mobilenetv2-12.onnx --target rv1103 --dtype fp
```

## Command Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--model` | Input model path (.pt or .onnx) | - | Yes |
| `--target` | Target platform | rv1103 | No |
| `--output` | Output RKNN model path | mobilenet.rknn | No |
| `--dtype` | Data type: i8 (quantized) or fp (float) | i8 | No |
| `--dataset` | Calibration dataset path | ../data/dataset.txt | No |

## Supported Platforms

- rv1103 (default)
- rv1106
- rk3588
- rk3568
- rk3566
- rk3562
- rk3576
- rv1126b
- rv1109
- rv1126
- rk1808
- rk3399pro

## Calibration Dataset

For quantized models (dtype=i8), a calibration dataset is required. The dataset should be a text file with one image path per line.

### Creating a Calibration Dataset

Use the provided script to create a calibration dataset from a directory of images:

```bash
python3 create_calibration_dataset.py --image_dir /path/to/images --output ../data/dataset.txt --num_images 100
```

### Manual Dataset Creation

Create a text file with image paths:

```bash
echo "/path/to/image1.jpg" > ../data/dataset.txt
echo "/path/to/image2.jpg" >> ../data/dataset.txt
# ... add more images
```

## Model Preprocessing

The conversion script automatically configures ImageNet preprocessing:

- **Mean values**: [123.675, 116.28, 103.53]
- **Std values**: [58.395, 57.12, 57.375]
- **Input size**: 224x224x3 RGB
- **Output**: 1000 classes

## Usage Examples

### Example 1: Convert ONNX with Custom Output
```bash
python3 convert.py \
    --model mobilenetv2-12.onnx \
    --target rk3588 \
    --output mobilenet_rk3588.rknn
```

### Example 2: Convert PyTorch with Float Precision
```bash
python3 convert.py \
    --model mobilenet_v2.pt \
    --target rv1106 \
    --dtype fp \
    --output mobilenet_float.rknn
```

### Example 3: Convert with Custom Calibration Dataset
```bash
python3 convert.py \
    --model mobilenetv2-12.onnx \
    --target rv1103 \
    --dataset ../data/custom_dataset.txt
```

## Error Handling

The script includes comprehensive error handling:

- **File validation**: Checks if input model and dataset files exist
- **Format validation**: Ensures model is .onnx or .pt format
- **Platform validation**: Warns about unsupported platforms
- **Conversion errors**: Provides detailed error messages

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Check the model path is correct
   - Ensure the file exists and is readable

2. **"Calibration dataset not found"**
   - Create a calibration dataset using the provided script
   - Or use `--dtype fp` for float precision (no calibration needed)

3. **"Unsupported model format"**
   - Ensure the model file has .onnx or .pt extension
   - Check that the model file is not corrupted

4. **Conversion fails**
   - Check RKNN Toolkit installation
   - Verify target platform compatibility
   - Try with float precision first: `--dtype fp`

### Performance Tips

- Use 100-1000 images for calibration dataset
- Ensure calibration images are representative of your use case
- For production, use quantized models (i8) for better performance
- For development/testing, use float precision (fp) for easier debugging

## Integration with Build System

The converted RKNN model can be used with the existing C++ demos:

```bash
# Build the demo
cd ../cpp
./build-linux.sh -t rv1103 -a armhf -d mobilenet

# Run inference
./rknn_mobilenet_demo model/mobilenet.rknn model/bell.jpg
```

## File Structure

```
mobilenet/
├── model/
│   ├── convert.py                    # Main conversion script
│   ├── create_calibration_dataset.py # Dataset creation helper
│   ├── mobilenetv2-12.onnx          # Pre-trained ONNX model
│   ├── mobilenet.rknn               # Converted RKNN model (output)
│   └── README_conversion.md         # This file
├── data/
│   └── dataset.txt                  # Calibration dataset
└── cpp/                             # C++ inference demo
```

## Advanced Configuration

### Custom Preprocessing

If your model uses different preprocessing, modify the `configure_model` function in `convert.py`:

```python
# For normalized input (0-1 range)
rknn.config(
    mean_values=[[0.485, 0.456, 0.406]], 
    std_values=[[0.229, 0.224, 0.225]],
    target_platform=target_platform
)
```

### Quantization Options

For advanced quantization settings, modify the configuration:

```python
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]], 
    std_values=[[58.395, 57.12, 57.375]],
    target_platform=target_platform,
    quantized_dtype='asymmetric_quantized-8',
    quantized_algorithm='kl_divergence'  # Alternative to 'normal'
)
```

## Support

For issues with the conversion process:

1. Check the error messages for specific issues
2. Verify all prerequisites are installed
3. Test with the provided sample model first
4. Ensure target platform compatibility 