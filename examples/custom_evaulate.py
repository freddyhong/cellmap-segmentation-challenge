import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm
from upath import UPath

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from cellmap_segmentation_challenge.evaluate import (
    INSTANCE_CLASSES,
    TRUTH_PATH,
    combine_scores,
    match_crop_space,
    score_instance,
    score_semantic,
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def get_crop_metadata_from_zarr(truth_label_path: str):
    try:
        group = zarr.open(truth_label_path, mode='r')
        attrs = group.attrs.asdict()
        if 'multiscales' in attrs:
            meta = attrs['multiscales'][0]
            transform = meta['datasets'][0]['coordinateTransformations']
            scale = next(t['scale'] for t in transform if t['type'] == 'scale')
            translation = next(t['translation'] for t in transform if t['type'] == 'translation')
            shape = zarr.open(str(Path(truth_label_path) / meta['datasets'][0]['path']), mode='r').shape
            return {'voxel_size': scale, 'shape': shape, 'translation': translation}
        else:
            # Fallback for simpler Zarr formats
            voxel_size = attrs.get('voxel_size') or attrs.get('resolution') or attrs.get('scale')
            shape = group.shape
            translation = attrs.get('translation') or attrs.get('offset')
            if voxel_size and shape and translation is not None:
                return {'voxel_size': voxel_size, 'shape': shape, 'translation': translation}
        raise ValueError("Could not determine metadata from Zarr attributes.")
    except Exception as e:
        logging.error(f"Error reading metadata from {truth_label_path}: {e}")
        return None

def score_label_fixed(pred_label_path, label_name, crop_name, truth_path, instance_classes):
    logging.info(f"Scoring {label_name} for {crop_name}...")
    truth_label_path = str(Path(truth_path) / crop_name / label_name)
    
    metadata = get_crop_metadata_from_zarr(truth_label_path)
    if not metadata:
        logging.error(f"Failed to get metadata for {truth_label_path}. Skipping.")
        return crop_name, label_name, {}

    pred_label = match_crop_space(
        pred_label_path, label_name, metadata['voxel_size'], metadata['shape'], metadata['translation']
    )
    truth_label = match_crop_space(
        truth_label_path, label_name, metadata['voxel_size'], metadata['shape'], metadata['translation']
    )

    mask_path = Path(truth_path) / crop_name / f"{label_name}_mask"
    if mask_path.exists():
        logging.info(f"Applying mask for {label_name}...")
        mask = zarr.open(str(mask_path), mode='r')[:]
        pred_label *= mask
        truth_label *= mask

    if label_name in instance_classes:
        results = score_instance(pred_label, truth_label, metadata['voxel_size'])
    else:
        results = score_semantic(pred_label, truth_label)
        
    results["num_voxels"] = int(np.prod(truth_label.shape))
    results["voxel_size"] = metadata['voxel_size']
    results["is_missing"] = False
    return crop_name, label_name, results

def get_evaluation_args_fixed(volumes, submission_path, truth_path, instance_classes):
    arglist = []
    submission_path = UPath(submission_path)
    truth_path = UPath(truth_path)
    for volume in volumes:
        pred_vol_path = submission_path / volume
        truth_vol_path = truth_path / volume
        try:
            pred_labels = list(zarr.open(str(pred_vol_path), mode='r').keys())
            truth_labels = [k for k in zarr.open(str(truth_vol_path), mode='r').keys() if "_mask" not in k]
            matching_labels = list(set(pred_labels) & set(truth_labels))
            logging.info(f"Found matching labels for {volume}: {matching_labels}")
            for label in matching_labels:
                arglist.append((str(pred_vol_path / label), label, volume, str(truth_path), instance_classes))
        except Exception as e:
            logging.error(f"Could not process volume {volume}. Error: {e}")
    return arglist

def score_local_submission(submission_dir: str, result_file: str = None):
    submission_path = Path(submission_dir)
    pred_volumes = [d.name for d in submission_path.glob("crop*") if d.is_dir()]
    if not pred_volumes:
        raise ValueError(f"No 'crop*' directories found in {submission_path.resolve()}.")

    logging.info(f"Found prediction volumes: {pred_volumes}")
    evaluation_args = get_evaluation_args_fixed(pred_volumes, submission_path, TRUTH_PATH, INSTANCE_CLASSES)

    if not evaluation_args:
        logging.error("No matching labels found to evaluate.")
        return

    scores = {}
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(score_label_fixed, *args) for args in evaluation_args]
        results = [future.result() for future in tqdm(as_completed(futures), desc="Scoring labels", total=len(futures))]

    for crop_name, label_name, result in results:
        if not result:
            continue
        if crop_name not in scores: scores[crop_name] = {}
        scores[crop_name][label_name] = result

    final_scores = combine_scores(scores, include_missing=False)
    if result_file:
        with open(result_file, "w") as f: json.dump(final_scores, f, indent=4)
        logging.info(f"Evaluation results saved to {result_file}")
    else:
        print("\n--- Evaluation Results ---\n", json.dumps(final_scores, indent=4))

    logging.info(f"Evaluation complete. Overall Score: {final_scores.get('overall_score', 'N/A'):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate local predictions for the CellMap Challenge.")
    parser.add_argument("submission_dir", help="Path to the directory with structured predictions.")
    parser.add_argument("--result_file", "-o", help="Optional path to save JSON results.")
    args = parser.parse_args()
    score_local_submission(args.submission_dir, args.result_file)