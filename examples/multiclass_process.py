import zarr
import numpy as np
import shutil
import click
from pathlib import Path
from glob import glob

# CLASS_MAPPING remains the same
CLASS_MAPPING = {
    "cell": [
        "pm", "mito_mem", "mito_lum", "mito_ribo", "golgi_mem", "golgi_lum",
        "ves_mem", "ves_lum", "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
        "ld_mem", "ld_lum", "er_mem", "er_lum", "eres_mem", "eres_lum",
        "ne_mem", "ne_lum", "np_out", "np_in", "hchrom", "nhchrom", "echrom", "nechrom", 
        "nucpl", "nucleo", "mt_out", "cent", "cent_dapp", "cent_sdapp", "ribo", "cyto",
        "mt_in", "vim", "glyco", "perox_mem", "perox_lum", 
        "isg_mem", "isg_lum", "isg_ins", "actin", "tbar"
    ],
    "nuc": [
        "ne_mem", "ne_lum", "np_out", "np_in", "hchrom", "nhchrom", "echrom",
        "nechrom", "nucpl", "nucleo"
    ],
    "er": [
        "er_mem", "er_lum", "eres_mem", "eres_lum", "ne_mem", "ne_lum",
        "np_out", "np_in"
    ],
    "endo": ["endo_lum", "endo_mem"],
    "lyso": ["lyso_lum", "lyso_mem"],
    "mito": ["mito_lum", "mito_mem", "mito_ribo"],
    "ves": ["ves_lum", "ves_mem"],
    "perox": ["perox_mem", "perox_lum"],
    "ld": ["ld_mem", "ld_lum"],
    "cent_all": ["cent", "cent_dapp", "cent_sdapp"],
    "chlor": ["chlor_mem", "chlor_lum", "chlor_sg"],
    "chrom": ["hchrom", "nhchrom", "echrom", "nechrom"],
    "er_mem_all": ["er_mem", "eres_mem", "ne_mem"],
    "eres": ["eres_mem", "eres_lum"],
    "golgi": ["golgi_mem", "golgi_lum"],
    "isg": ["isg_mem", "isg_lum", "isg_ins"],
    "mt": ["mt_in", "mt_out"],
    "ne": ["ne_mem", "ne_lum", "np_out", "np_in"],
    "ne_mem_all": ["ne_mem", "np_out", "np_in"],
    "np": ["np_out", "np_in"],
    "vac": ["vac_mem", "vac_lum"],
    "yolk": ["yolk_mem", "yolk_lum"],
}


@click.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--output-dir', '-o', required=True, type=click.Path(file_okay=False))
@click.option('--crop', '-c', required=True)
@click.option(
    '--classes',
    default="all",
)
def combine_masks(input_dir, output_dir, crop, classes):

    # Find the specific crop directory to process
    crop_path_pattern = str(Path(input_dir) / f"crop{crop}")
    found_paths = glob(crop_path_pattern)
    if not found_paths:
        return
    source_path = Path(found_paths[0])

    # Determine which classes to process
    if classes.lower() == "all":
        process_map = CLASS_MAPPING
    else:
        requested = classes.split(',')
        process_map = {k: v for k, v in CLASS_MAPPING.items() if k in requested}

    # Loop through all the defined classes
    for target_class, source_components in process_map.items():
        print(f"\n--- Processing target class: {target_class} ---")

        all_source_paths = [source_path / comp / "s0" for comp in source_components]

        existing_source_paths = [p for p in all_source_paths if p.exists()]
        
        if len(existing_source_paths) < len(all_source_paths):
            print(f" Found only {len(existing_source_paths)} of {len(all_source_paths)} masks for '{target_class}'. Proceeding with the ones found.")
            missing_paths = set(all_source_paths) - set(existing_source_paths)
            for p in missing_paths:
                print(f"  - Missing: {p}")

        print(f"Loading {len(existing_source_paths)} source masks...")
        list_of_arrays = [zarr.open(str(p), mode='r')[:] for p in existing_source_paths]

        if len(list_of_arrays) == 1:
            combined_array = list_of_arrays[0]
        else:
            print("Combining masks")
            stacked_arrays = np.stack(list_of_arrays, axis=0)
            combined_array = np.sum(stacked_arrays, axis=0)

        source_group_path_for_metadata = existing_source_paths[0].parent
        output_group_path = Path(output_dir) / source_path.name / target_class

        if output_group_path.exists():
            shutil.rmtree(output_group_path)
        shutil.copytree(source_group_path_for_metadata, output_group_path)

        zarr_out_group = zarr.open(str(output_group_path), mode='r+')
        zarr_out_group['s0'][:] = combined_array
        print(f"✅ Saved combined mask to {output_group_path}")

if __name__ == '__main__':
    combine_masks()