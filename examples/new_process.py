#!/usr/bin/env python3
"""
2D zarr processing script with command line interface
"""

import zarr
import numpy as np
import torch
import argparse
import json
from pathlib import Path

def create_multiscale_metadata(shape, scale=[8, 8, 8], translation=[0, 0, 0], crop_name="crop", class_name="mito"):
    """Create OME-NGFF multiscale metadata"""
    return {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"}
                ],
                "coordinateTransformations": [
                    {"scale": [1.0, 1.0, 1.0], "type": "scale"}
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {"scale": scale, "type": "scale"},
                            {"translation": translation, "type": "translation"}
                        ],
                        "path": "s0"
                    }
                ],
                "name": f"/{crop_name}/{class_name}",
                "version": "0.4"
            }
        ]
    }

def process_2d_zarr(crops, input_base, output_base, class_name="mito", threshold=0.5):
    """Process 2D zarr prediction files"""
    
    print(f"Starting 2D zarr processing...")
    print(f"Input base: {input_base}")
    print(f"Output base: {output_base}")
    print(f"Crops: {crops}")
    print(f"Class: {class_name}")
    print(f"Threshold: {threshold}")
    print(f"Output values: 0 and 1 (binary)")
    
    # Ensure output directory exists
    output_base.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for crop in crops:
        crop_name = f'crop{crop}'
        print(f'\nProcessing {crop_name}...')
        
        # Input and output paths
        input_path = input_base / crop_name / class_name / 's0'
        output_crop_path = output_base / crop_name
        output_class_path = output_crop_path / class_name
        
        # Check if input exists
        if not input_path.exists():
            print(f'   Input path does not exist: {input_path}')
            continue
        
        try:
            # Load input zarr array
            z_input = zarr.open(str(input_path), mode='r')
            print(f'   Input shape: {z_input.shape}')
            print(f'   Input dtype: {z_input.dtype}')
            print(f'   Input chunks: {z_input.chunks}')
            
            # Create the class group with metadata
            class_group = zarr.open(str(output_class_path), mode='w')
            
            # Create output zarr at s0 level within the group
            z_output = class_group.create_dataset(
                's0',
                shape=z_input.shape, 
                dtype=np.uint8, 
                chunks=z_input.chunks
            )
            
            print(f'   Processing {z_input.shape[0]} slices...')
            
            # Process slice by slice (2D processing on each slice)
            for i in range(z_input.shape[0]):
                # Load slice
                data_slice = z_input[i]
                
                # Convert to torch tensor
                tensor = torch.from_numpy(data_slice).float()
                
                # Apply sigmoid and threshold to get binary values
                probabilities = torch.sigmoid(tensor)
                binary_mask = (probabilities > threshold).numpy().astype(np.uint8)
                
                # Save processed slice (always 0 or 1)
                z_output[i] = binary_mask
                
                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == z_input.shape[0]:
                    print(f'   Progress: {i + 1}/{z_input.shape[0]} slices')
            
            # Read original multiscale metadata
            original_class_path = input_base / crop_name / class_name
            try:
                original_zarr = zarr.open(str(original_class_path), mode='r')
                original_attrs = original_zarr.attrs.asdict()
                
                if 'multiscales' in original_attrs:
                    # Extract original scale and translation
                    original_ms = original_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations']
                    scale = None
                    translation = None
                    
                    for transform in original_ms:
                        if transform['type'] == 'scale':
                            scale = transform['scale']
                        elif transform['type'] == 'translation':
                            translation = transform['translation']
                    
                    if scale is None:
                        scale = [8, 8, 8]
                    if translation is None:
                        translation = [0, 0, 0]
                else:
                    scale = [8, 8, 8]
                    translation = [0, 0, 0]
            except:
                scale = [8, 8, 8]
                translation = [0, 0, 0]
            
            # Add multiscale metadata to the class group
            metadata = create_multiscale_metadata(
                shape=z_output.shape,
                scale=scale,
                translation=translation,
                crop_name=crop_name,
                class_name=class_name
            )
            class_group.attrs.update(metadata)
            
            print(f'   Completed {crop_name}')
            print(f'   Output saved to: {output_class_path}')
            processed_count += 1
            
        except Exception as e:
            print(f'   Error processing {crop_name}: {e}')
            import traceback
            traceback.print_exc()
    
    print(f'\nProcessed {processed_count}/{len(crops)} crops successfully!')
    print(f'Results saved to: {output_base}')

def main():
    parser = argparse.ArgumentParser(description='Process 2D zarr prediction files')
    parser.add_argument('-c', '--crops', required=True, 
                        help='Comma-separated list of crop numbers (e.g., 234,236,237)')
    parser.add_argument('-d', '--dataset', required=True,
                        help='Dataset name (e.g., jrc_cos7-1a)')
    parser.add_argument('-i', '--input-base', 
                        default='/home/wghong22/cellmap-segmentation-challenge/data/predictions',
                        help='Input base directory (default: predictions dir)')
    parser.add_argument('-o', '--output-base',
                        default='/home/wghong22/cellmap-segmentation-challenge/data/processed',
                        help='Output base directory (default: processed dir)')
    parser.add_argument('-C', '--class-name', default='mito',
                        help='Class name to process (default: mito)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Threshold for binary mask (default: 0.5)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    
    args = parser.parse_args()
    
    # Parse crops
    crops = [crop.strip() for crop in args.crops.split(',')]
    
    # Construct full paths using dataset name
    input_base = Path(args.input_base) / f"{args.dataset}.zarr"
    output_base = Path(args.output_base) / f"{args.dataset}.zarr"
    
    # Check if output exists and handle overwrite
    if output_base.exists() and not args.overwrite:
        response = input(f"Output directory {output_base} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run processing
    process_2d_zarr(
        crops=crops,
        input_base=input_base,
        output_base=output_base,
        class_name=args.class_name,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()