"""QC visual generation and metric serialization."""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from aind_pophys_converter.utils.image_utils import add_border_and_label
from aind_pophys_converter.utils.metadata_utils import (
    build_fov_qc_metric,
    build_vasculature_qc_metric,
)


def write_fov_metric(
    unique_id: str,
    raw_tif_path: Path,
    intended_depth: int,
    targeted_structure_id: str,
    closest_z: float,
    merged_path: Path,
    output_dir: Path,
) -> None:
    """Write a single FOV parent-child matching QC metric JSON.

    Parameters
    ----------
    unique_id : str
        Identifier string like 'VISp_200'.
    raw_tif_path : Path
        Path to the parent depth snapshot TIFF.
    intended_depth : int
        Intended imaging depth in microns.
    targeted_structure_id : str
        CCF structure acronym.
    closest_z : float
        Closest child z-plane in microns.
    merged_path : Path
        Path to the saved merged PNG.
    output_dir : Path
        Directory to write the metric JSON.
    """
    metric = build_fov_qc_metric(
        unique_id=unique_id,
        raw_tif_path=raw_tif_path,
        intended_depth=intended_depth,
        targeted_structure_id=targeted_structure_id,
        closest_z=closest_z,
        merged_path=merged_path,
    )
    metric_out_path = output_dir / f"{unique_id}_fov_metric.json"
    with open(metric_out_path, "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
    logging.info(f"Saved QC metric -> {metric_out_path}")


def write_vasculature_metric(
    vasculature_output_fp: Path,
    output_dir: Path,
) -> None:
    """Write the vasculature window clarity QC metric JSON.

    Parameters
    ----------
    vasculature_output_fp : Path
        Path to the saved vasculature PNG.
    output_dir : Path
        Directory to write the metric JSON.
    """
    metric = build_vasculature_qc_metric(vasculature_output_fp)
    metric_out_path = output_dir / "vasculature_metric.json"
    with open(metric_out_path, "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
    logging.info(f"Saved QC metric -> {metric_out_path}")


def write_avg_depth_slices(splitter, output_dir: Path) -> None:
    """Write per-ROI averaged-depth slices as float32 TIF files.

    Parameters
    ----------
    splitter : AvgImageTiffSplitter
        Initialized splitter with a loaded averaged-depth TIFF.
    output_dir : Path
        Directory to write the per-z-value TIF files.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for roi_idx, z_int in splitter.roi_z_int_manifest:
        z_value = splitter._z_from_int(z_int)
        tif_path = output_dir / f"{z_value:.1f}.tif"
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp_tif:
            tmp_path = Path(tmp_tif.name)
            splitter.write_output_file(
                i_roi=roi_idx, z_value=z_value, output_path=tmp_path
            )
            img_array = tifffile.imread(tmp_path)
        tifffile.imwrite(tif_path, img_array.astype(np.float32))
        logging.info(f"Saved TIF: {tif_path}")


def pair_depth_tifs_with_avg_depth_pngs(
    pophys_dir: Path,
    avg_slice_dir: Path,
    platform_fp: Path,
    output_dir: Path,
) -> None:
    """Pair parent depth TIFs with child averaged-depth PNGs and write QC.

    For each imaging plane defined in platform.json, locate the parent depth
    TIFF by intended_depth and targeted_structure_id, find the closest child
    averaged-depth TIF by abs(scanimage_scanfield_z), merge them side-by-side
    with borders and labels, and save the result. Also writes a per-plane QC
    metric JSON.

    Parameters
    ----------
    pophys_dir : Path
        Directory containing parent depth TIFFs named
        <timestamp>_<intended_depth>_<targeted_structure_id>_depth.tif.
    avg_slice_dir : Path
        Directory containing child averaged-depth float32 TIF files named by
        z-value (e.g. -276.0.tif), written by write_avg_depth_slices.
    platform_fp : Path
        Path to the session platform.json.
    output_dir : Path
        Directory to save merged PNGs and QC metric JSONs.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(platform_fp) as f:
        platform_json = json.load(f)

    imaging_planes = [
        plane
        for group in platform_json.get("imaging_plane_groups", [])
        for plane in group.get("imaging_planes", [])
    ]

    if not imaging_planes:
        logging.info(
            "No imaging planes found in platform.json;"
            " skipping depth pairing."
        )
        return

    child_tifs = list(avg_slice_dir.glob("*.tif"))
    z_to_tif = {abs(float(p.stem)): p for p in child_tifs}

    if not z_to_tif:
        logging.info("No averaged-depth TIFs found; skipping depth pairing.")
        return

    for plane in imaging_planes:
        intended_depth = plane["intended_depth"]
        targeted_structure_id = plane["targeted_structure_id"]
        scanfield_z = plane["scanimage_scanfield_z"]

        parent_tifs = list(
            pophys_dir.glob(
                f"*_{intended_depth}_{targeted_structure_id}_depth.tif"
            )
        )
        if len(parent_tifs) == 0:
            logging.warning(
                f"No parent TIF found for "
                f"intended_depth={intended_depth}, "
                f"targeted_structure_id={targeted_structure_id}; skipping."
            )
            continue
        if len(parent_tifs) > 1:
            logging.warning(
                f"Multiple parent TIFs found for "
                f"intended_depth={intended_depth}, "
                f"targeted_structure_id={targeted_structure_id}; "
                f"using first: {parent_tifs[0].name}"
            )
        raw_tif_path = parent_tifs[0]

        closest_z = min(
            z_to_tif.keys(), key=lambda z: abs(z - abs(scanfield_z))
        )
        child_tif_path = z_to_tif[closest_z]

        with tifffile.TiffFile(raw_tif_path) as tif:
            parent_array = tif.asarray().astype(np.float64)
        with tifffile.TiffFile(child_tif_path) as tif:
            child_array = tif.asarray().astype(np.float64)

        def _to_uint8(arr: np.ndarray) -> np.ndarray:
            """Normalize array to uint8 using 5th/95th percentiles."""
            lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
            if hi > lo:
                return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(
                    np.uint8
                )
            return np.zeros_like(arr, dtype=np.uint8)

        p_lo = np.percentile(parent_array, 5)
        p_hi = np.percentile(parent_array, 95)
        c_lo = np.percentile(child_array, 5)
        c_hi = np.percentile(child_array, 95)
        logging.info(
            f"Parent normalization: p5={p_lo:.1f}, p95={p_hi:.1f}"
            f" (dtype={parent_array.dtype})"
        )
        logging.info(
            f"Child  normalization: p5={c_lo:.1f}, p95={c_hi:.1f}"
            f" (dtype={child_array.dtype})"
        )

        raw_img = Image.fromarray(_to_uint8(parent_array))
        avg_img = Image.fromarray(_to_uint8(child_array))
        raw_img = add_border_and_label(raw_img, "Parent")
        avg_img = add_border_and_label(avg_img, "Child")

        total_width = raw_img.width + avg_img.width
        max_height = max(raw_img.height, avg_img.height)
        merged = Image.new("RGB", (total_width, max_height), color="white")
        merged.paste(raw_img, (0, 0))
        merged.paste(avg_img, (raw_img.width, 0))

        merged_path = output_dir / f"{raw_tif_path.stem}_merged.png"
        merged.save(merged_path)
        logging.info(
            f"Saved merged image for {raw_tif_path.name} -> {merged_path}"
        )

        try:
            child_tif_path.unlink()
            logging.info(
                f"Deleted original averaged TIF -> {child_tif_path}"
            )
        except Exception as e:
            logging.warning(f"Failed to delete {child_tif_path}: {e}")

        unique_id = f"{targeted_structure_id}_{intended_depth}"
        write_fov_metric(
            unique_id=unique_id,
            raw_tif_path=raw_tif_path,
            intended_depth=intended_depth,
            targeted_structure_id=targeted_structure_id,
            closest_z=closest_z,
            merged_path=merged_path,
            output_dir=output_dir,
        )


def create_vasculature(pophys_dir: Path, output_dir: Path) -> None:
    """Create a vasculature PNG and write a window-clarity QC metric.

    Parameters
    ----------
    pophys_dir : Path
        Directory containing the vasculature TIFF file.
    output_dir : Path
        Parent output directory; vasculature/ subdirectory is created here.
    """
    vasculature_fp = next(pophys_dir.glob("*_vasculature.tif"), None)
    if not vasculature_fp:
        logging.info("No vasculature TIFF found; skipping.")
        return
    vasculature_output_dir = output_dir / "vasculature"
    vasculature_output_dir.mkdir(exist_ok=True, parents=True)
    vasculature_output_fp = vasculature_output_dir / "vasculature.png"
    with Image.open(vasculature_fp) as im:
        im.save(vasculature_output_fp)
    logging.info(f"Saved vasculature image -> {vasculature_output_fp}")
    write_vasculature_metric(vasculature_output_fp, vasculature_output_dir)
