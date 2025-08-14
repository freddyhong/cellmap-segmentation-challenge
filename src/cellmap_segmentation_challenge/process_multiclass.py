from functools import partial
import os
from glob import glob
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from cellmap_data import CellMapDatasetWriter, CellMapImage
import torch
from tqdm import tqdm
from upath import UPath

from .config import PREDICTIONS_PATH, PROCESSED_PATH
from .utils import load_safe_config, fetch_test_crop_manifest
from .utils.datasplit import get_formatted_fields

# This is the new, correct _process function for process.py
def get_channel_indices(model_classes: list[str], sub_classes: list[str]) -> list[int]:
    """Finds the channel indices for a list of sub-classes."""
    return [model_classes.index(sc) for sc in sub_classes if sc in model_classes]

def _process(
    dataset_writer_kwargs: dict[str, Any],
    process_func: Callable,
    batch_size: int,
    target_class: str,
    class_mapping: dict[str, list[str]],
    model_output_classes: list[str],
) -> None:
    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)
    
    for batch in tqdm(dataset_writer.loader(batch_size=batch_size), dynamic_ncols=True):
        # batch['input'] is the raw multi-channel tensor from your prediction
        multi_channel_probs = batch["input"]

        # 1. Identify which channels to combine for the current target class
        sub_classes = class_mapping.get(target_class, [target_class])
        channel_indices = get_channel_indices(model_output_classes, sub_classes)

        if not channel_indices:
            print(f"Warning: No channels found for target class '{target_class}'. Skipping.")
            continue

        # 2. Sum the probabilities of the relevant channels
        # Shape: [B, C, Z, Y, X] -> [B, Z, Y, X]
        combined_probs = torch.sum(multi_channel_probs[:, channel_indices, ...], dim=1)

        # 3. Pass the combined tensor to your original process_func
        # The process_func now receives a single-channel probability map
        outputs = process_func({"input": combined_probs})
        
        dataset_writer[batch["idx"]] = {"output": outputs}


def process_multiclass(
    config_path: str | UPath,
    crops: str = "test",
    input_path: str = PREDICTIONS_PATH,
    output_path: str = PROCESSED_PATH,
    overwrite: bool = False,
    device: Optional[str | torch.device] = None,
    max_workers: int = os.cpu_count(),
) -> None:
    """
    Process and save arrays using an arbitrary process function defined in a config python file.

    Parameters
    ----------
    config_path : str | UPath
        The path to the python file containing the process function and other configurations. The script should specify the process function as `process_func`; `input_array_info` and `target_array_info` corresponding to the chunk sizes and scales for the input and output datasets, respectively; `batch_size`; `classes`; and any other required configurations.
        The process function should take an array as input and return an array as output.
    crops: str, optional
        A comma-separated list of crop numbers to process, or "test" to process the entire test set. Default is "test".
    input_path: str, optional
        The path to the data to process, formatted as a string with a placeholders for the crop number, dataset, and label. Default is PREDICTIONS_PATH set in `cellmap-segmentation/config.py`.
    output_path: str, optional
        The path to save the processed output to, formatted as a string with a placeholders for the crop number, dataset, and label. Default is PROCESSED_PATH set in `cellmap-segmentation/config.py`.
    overwrite: bool, optional
        Whether to overwrite the output dataset if it already exists. Default is False.
    device: str | torch.device, optional
        The device to use for processing the data. Default is to use that specified in the config. If not specified, then defaults to "cuda" if available, then "mps", otherwise "cpu".
    max_workers: int, optional
        The maximum number of workers to use for processing the data. Default is the number of CPUs on the system.

    """
    config = load_safe_config(config_path)
    process_func = config.process_func
    
    # --- NEW: Load variables from your updated config ---
    target_classes = config.target_classes
    model_output_classes = config.model_output_classes
    class_mapping = config.class_mapping
    # ---

    batch_size = getattr(config, "batch_size", 8)
    input_array_info = config.input_array_info
    target_array_info = config.target_array_info

    if device is None:
        if hasattr(config, "device"):
            device = config.device
        elif torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}
    assert (
        input_arrays is not None and target_arrays is not None
    ), "No array info provided"

    # Get the crops to predict on
    if crops == "test":
        test_crops = fetch_test_crop_manifest()
        crop_list = list(set([c.id for c in test_crops]))
    else:
        crop_list = crops.split(",")

    crop_paths = []
    for i, crop in enumerate(crop_list):
        if (isinstance(crop, str) and crop.isnumeric()) or isinstance(crop, int):
            crop = f"crop{crop}"
            crop_list[i] = crop  # type: ignore

        crop_paths.extend(
            glob(input_path.format(dataset="*", crop=crop).rstrip(os.path.sep))
        )

    crop_dict = {}
    for crop, path in zip(crop_list, crop_paths):
        dataset = get_formatted_fields(path, input_path, ["{dataset}"])["dataset"]
        crop_dict[crop] = [
            input_path.format(
                crop=crop,
                dataset=dataset,
            ),
            output_path.format(
                crop=crop,
                dataset=dataset,
            ),
        ]

    dataset_writers_and_params = []
    for crop, (in_path, out_path) in crop_dict.items():
        input_image = CellMapImage(
            in_path,
            target_class=None,
            target_scale=input_array_info["scale"],
            target_voxel_shape=input_array_info["shape"]
        )
        # The key "output" must match the name of your target_array
        bounds = {"output": input_image.bounding_box}
        for label in target_classes:
            class_out_path = str(UPath(out_path) / label)

            # The input_images and target_bounds logic for finding crop boundaries
            # can be simplified if the entire crop is processed. Let's assume
            # the writer is created for the whole dataset pointed to by `in_path`.
            # Note: CellMapImage `raw_path` should point to the multi-channel prediction Zarr.
            
            # Create a dictionary of parameters for each processing job
            dataset_writers_and_params.append({
                "writer_kwargs": {
                    "raw_path": in_path,  # This is the multi-channel prediction
                    "target_path": class_out_path,
                    "classes": [label], # Used for metadata, output is single class
                    "input_arrays": {"input": input_array_info},
                    "target_arrays": {"output": target_array_info},
                    "overwrite": overwrite,
                    "device": device,
                    "target_bounds": bounds,
                    # You might need to add target_bounds back here if you process sub-regions
                },
                "params": {
                    "target_class": label,
                    "class_mapping": class_mapping,
                    "model_output_classes": model_output_classes,
                }
            })

    executor = ThreadPoolExecutor(max_workers)

    futures = []
    for job in dataset_writers_and_params:
        partial_process = partial(
            _process,
            process_func=process_func,
            batch_size=batch_size,
            **job["params"],
        )
        futures.append(executor.submit(partial_process, job["writer_kwargs"]))

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing..."):
        future.result()