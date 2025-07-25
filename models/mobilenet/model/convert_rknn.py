#!/usr/bin/env python3
import sys
import os
import argparse
from rknn.api import RKNN

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MobileNet model to RKNN format')
    parser.add_argument('--model', required=True, help='Input model path (.pt or .onnx)')
    parser.add_argument('--target', default='rv1103', help='Target platform (default: rv1103)')
    parser.add_argument('--output', default='mobilenet.rknn', help='Output RKNN model path')
    parser.add_argument('--dtype', choices=['i8', 'fp'], default='i8', help='Data type (i8 for quantization, fp for float)')
    parser.add_argument('--dataset', default='../data/dataset.txt', help='Calibration dataset path')
    return parser.parse_args()

def validate_inputs(args):
    """Validate input parameters and files"""
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    if args.dtype == 'i8' and not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Calibration dataset not found: {args.dataset}")
    
    supported_platforms = ['rv1103', 'rv1106', 'rk3588', 'rk3568', 'rk3566', 'rk3562', 'rk3576', 'rv1126b', 'rv1109', 'rv1126', 'rk1808', 'rk3399pro']
    if args.target not in supported_platforms:
        print(f"Warning: Target platform {args.target} may not be fully supported")
    
    # Validate model format
    if not (args.model.endswith('.onnx') or args.model.endswith('.pt')):
        raise ValueError("Unsupported model format. Use .onnx or .pt files")

def load_model(rknn, model_path):
    """Load model based on file extension"""
    if model_path.endswith('.onnx'):
        print('--> Load ONNX model')
        rknn.load_onnx(model=model_path)
    elif model_path.endswith('.pt'):
        print('--> Load PyTorch model')
        rknn.load_pytorch(model=model_path, input_size_list=[[1, 3, 224, 224]])
    else:
        raise ValueError("Unsupported model format. Use .onnx or .pt files")
    print('done')

def configure_model(rknn, target_platform, dtype):
    """Configure RKNN model with appropriate settings"""
    print('--> Config')
    
    # ImageNet preprocessing: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    # For uint8 input: mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]]
    # For normalized input (0-1): mean_values=[[0.485, 0.456, 0.406]], std_values=[[0.229, 0.224, 0.225]]
    
    if dtype == 'i8':
        # Quantization configuration
        rknn.config(
            mean_values=[[123.675, 116.28, 103.53]], 
            std_values=[[58.395, 57.12, 57.375]],
            target_platform=target_platform,
            quantized_dtype='asymmetric_quantized-8',
            quantized_algorithm='normal'
        )
    else:
        # Float configuration
        rknn.config(
            mean_values=[[123.675, 116.28, 103.53]], 
            std_values=[[58.395, 57.12, 57.375]],
            target_platform=target_platform
        )
    print('done')

def convert_model(args):
    """Main conversion function"""
    rknn = RKNN()
    
    try:
        # Configure model
        configure_model(rknn, args.target, args.dtype)
        
        # Load model
        load_model(rknn, args.model)
        
        # Build model
        print('--> Build')
        if args.dtype == 'i8':
            rknn.build(do_quantization=True, dataset=args.dataset)
        else:
            rknn.build(do_quantization=False)
        print('done')
        
        # Export model
        print('--> Export')
        rknn.export_rknn(args.output)
        print('done')
        
        print(f"Model successfully converted to: {args.output}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return -1
    finally:
        rknn.release()
    
    return 0

def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Perform conversion
        ret = convert_model(args)
        
        if ret == 0:
            print("Conversion completed successfully!")
        else:
            print("Conversion failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 