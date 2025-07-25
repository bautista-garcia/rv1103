#!/usr/bin/env python3
import os
import glob
import argparse
import requests
import urllib.request
from urllib.parse import urlparse
import time
import random

def download_image(url, save_path, timeout=10):
    """Download a single image from URL"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download image
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def download_images_from_urls(urls, output_dir, num_images=None):
    """Download images from a list of URLs"""
    if num_images:
        urls = urls[:num_images]
    
    downloaded_files = []
    
    for i, url in enumerate(urls):
        # Generate filename from URL or use index
        filename = f"calibration_{i:04d}.jpg"
        save_path = os.path.join(output_dir, filename)
        
        print(f"Downloading {i+1}/{len(urls)}: {url}")
        
        if download_image(url, save_path):
            downloaded_files.append(save_path)
        
        # Small delay to be respectful to servers
        time.sleep(0.1)
    
    return downloaded_files

def get_imagenet_sample_urls(num_images=100):
    """Get sample ImageNet image URLs for calibration"""
    # Sample ImageNet URLs (these are example URLs - in practice you'd use a proper dataset)
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/1200px-Eopsaltria_australis_-_Mogo_Campground.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Red_Apple.jpg/1200px-Red_Apple.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/1200px-Google_2015_logo.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Arduino_Logo.svg/1200px-Arduino_Logo.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Plus_symbol.svg/1200px-Plus_symbol.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Logo_TVRI_2000.svg/1200px-Logo_TVRI_2000.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Smiley.svg/1200px-Smiley.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Wikipedia-logo-en-big.png/1200px-Wikipedia-logo-en-big.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Wikimedia-logo.svg/1200px-Wikimedia-logo.svg.png"
    ]
    
    # Repeat the sample URLs to reach the desired number
    repeated_urls = []
    while len(repeated_urls) < num_images:
        repeated_urls.extend(sample_urls)
    
    return repeated_urls[:num_images]

def get_random_images_from_unsplash(num_images=100):
    """Get random images from Unsplash (requires API key for production use)"""
    # For demo purposes, using a simple approach
    # In production, you'd use the Unsplash API with proper authentication
    
    # Sample Unsplash image IDs (these are example IDs)
    unsplash_ids = [
        "photo-1506905925346-21bda4d32df4",  # Nature
        "photo-1441974231531-c6227db76b6e",  # Forest
        "photo-1506905925346-21bda4d32df4",  # Mountain
        "photo-1441974231531-c6227db76b6e",  # Lake
        "photo-1506905925346-21bda4d32df4",  # Sunset
    ]
    
    urls = []
    for i in range(num_images):
        # Use a simple image service for demo
        url = f"https://picsum.photos/224/224?random={i}"
        urls.append(url)
    
    return urls

def create_calibration_dataset_from_source(source_type, output_dir, output_file, num_images=100):
    """Create calibration dataset by downloading images from a source"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image URLs based on source type
    if source_type == "imagenet":
        urls = get_imagenet_sample_urls(num_images)
    elif source_type == "unsplash":
        urls = get_random_images_from_unsplash(num_images)
    elif source_type == "picsum":
        # Use Lorem Picsum for random images
        urls = [f"https://picsum.photos/224/224?random={i}" for i in range(num_images)]
    else:
        raise ValueError(f"Unknown source type: {source_type}")
    
    print(f"Downloading {len(urls)} images from {source_type}...")
    
    # Download images
    downloaded_files = download_images_from_urls(urls, output_dir, num_images)
    
    if not downloaded_files:
        raise ValueError("No images were successfully downloaded")
    
    # Create dataset.txt file with local paths
    with open(output_file, 'w') as f:
        for img_path in downloaded_files:
            # Use relative path from the dataset.txt location
            relative_path = os.path.relpath(img_path, os.path.dirname(output_file))
            f.write(f"{relative_path}\n")
    
    print(f"Created calibration dataset with {len(downloaded_files)} images")
    print(f"Dataset saved to: {output_file}")
    print(f"Images saved to: {output_dir}")
    
    return len(downloaded_files)

def create_calibration_dataset_from_directory(image_dir, output_file, num_images=100):
    """Create calibration dataset from existing image directory (original functionality)"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Collect all image files
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {image_dir}")
    
    # Take first num_images
    image_files = image_files[:num_images]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write dataset file
    with open(output_file, 'w') as f:
        for img_path in image_files:
            f.write(f"{img_path}\n")
    
    print(f"Created calibration dataset with {len(image_files)} images")
    print(f"Dataset saved to: {output_file}")
    
    return len(image_files)

def main():
    parser = argparse.ArgumentParser(description='Create calibration dataset by downloading images or from directory')
    parser.add_argument('--source', choices=['imagenet', 'unsplash', 'picsum', 'directory'], 
                       default='picsum', help='Source type for images (default: picsum)')
    parser.add_argument('--image_dir', help='Directory containing images (for source=directory)')
    parser.add_argument('--output', default='../data/dataset.txt', help='Output dataset file path')
    parser.add_argument('--output_dir', default='../data/calibration_images', help='Output directory for downloaded images')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to include (default: 100)')
    
    args = parser.parse_args()
    
    try:
        if args.source == 'directory':
            if not args.image_dir:
                raise ValueError("--image_dir is required when source=directory")
            num_created = create_calibration_dataset_from_directory(args.image_dir, args.output, args.num_images)
        else:
            num_created = create_calibration_dataset_from_source(args.source, args.output_dir, args.output, args.num_images)
        
        print(f"Successfully created calibration dataset with {num_created} images")
        
    except Exception as e:
        print(f"Error creating calibration dataset: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 