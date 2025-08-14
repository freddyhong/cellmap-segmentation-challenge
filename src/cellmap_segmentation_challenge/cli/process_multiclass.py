# In src/cellmap_segmentation_challenge/cli/process_multiclass.py

import os
import click

# Import config paths and the specific function this CLI will run
from cellmap_segmentation_challenge.config import PREDICTIONS_PATH, PROCESSED_PATH
from cellmap_segmentation_challenge.process_multiclass import process_multiclass


@click.command
@click.argument(
    "config_path",
    type=click.Path(exists=True),
)
@click.option(
    "--crops",
    "-c",
    default="test",
    help="Crops to process. Default: 'test'.",
)
@click.option(
    "--input-path",
    "-i",
    default=PREDICTIONS_PATH,
    help=f"Path to input data. Default: {PREDICTIONS_PATH}.",
)
@click.option(
    "--output-path",
    "-o",
    default=PROCESSED_PATH,
    help=f"Path to save output. Default: {PROCESSED_PATH}.",
)
@click.option(
    "--overwrite",
    "-O",
    is_flag=True,
    default=False,
    help="Overwrite existing data.",
)
@click.option(
    "--device",
    "-d",
    default=None,
    help="Device to use for processing.",
)
@click.option(
    "--max-workers",
    "-w",
    default=os.cpu_count(),
    help=f"Max workers for processing. Default: {os.cpu_count()}.",
)
def process_multiclass_cli(config_path, crops, input_path, output_path, overwrite, device, max_workers):
    process_multiclass(
        config_path=config_path,
        crops=crops,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        device=device,
        max_workers=max_workers,
    )