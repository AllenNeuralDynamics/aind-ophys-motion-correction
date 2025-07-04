import argparse
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from datetime import datetime as dt
from functools import lru_cache, partial
from glob import glob
from itertools import product
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import time
from typing import Callable, List, Optional, Tuple, Union

import cv2
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import suite2p
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_data_schema_models.process_names import ProcessName
from aind_log_utils.log import setup_logging
from aind_ophys_utils.array_utils import normalize_array
from aind_ophys_utils.summary_images import mean_image
from aind_ophys_utils.video_utils import (
    downsample_array,
    downsample_h5_video,
    encode_video,
)
from aind_qcportal_schema.metric_value import DropdownMetric
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
from pydantic_settings import BaseSettings
from ScanImageTiffReader import ScanImageTiffReader
from scipy.ndimage import median_filter
from scipy.stats import sigmaclip
from suite2p.registration.nonrigid import make_blocks
from suite2p.registration.register import pick_initial_reference, register_frames
from suite2p.registration.rigid import (
    apply_masks,
    compute_masks,
    phasecorr,
    phasecorr_reference,
    shift_frame,
)
from sync_dataset import Sync
from typing import Annotated
from pydantic import Field

mpl.use("Agg")


class MotionCorrectionSettings(BaseSettings, cli_parse_args=True):
    """Settings for Suite2P motion correction.

    This class defines all configuration parameters for motion correction using Suite2P.
    Values can be provided via constructor, environment variables, or .env file.
    """

    input_dir: Path = Field(
        default=Path("/data/"), description="File or directory where h5 file is stored"
    )
    output_dir: Path = Field(default=Path("/results/"), description="Output directory")
    debug: bool = Field(default=False, description="Run with partial dataset")
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description="Directory into which to write temporary files produced by Suite2P",
    )
    data_type: str = Field(
        default="h5", description="Processing h5 (default) or TIFF timeseries"
    )
    do_registration: bool = Field(
        default="true",
        description="whether to register data (2 forces re-registration)",
    )
    batch_size: int = Field(default=500, description="Number of frames per batch")
    align_by_chan: int = Field(
        default=1,
        description="when multi-channel, you can align by non-functional channel "
        "(1-based)",
    )
    maxregshift: float = Field(
        default=0.1,
        description="max allowed registration shift, as a fraction of "
        "frame max(width and height). This will be ignored if force_refImg is set to True",
    )
    force_refImg: bool = Field(
        default=True, description="Force the use of an external reference image"
    )
    nonrigid: bool = Field(
        default=True, description="Whether to use non-rigid registration"
    )
    block_size: list = Field(default_factory=lambda: [128, 128],
        description="Block size for non-rigid registration.")
    snr_thresh: float = Field(
        default=1.2,
        description="if any nonrigid block is below this threshold, it gets smoothed "
        "until above this threshold. 1.0 results in no smoothing",
    )
    maxregshiftNR: int = Field(
        default=5,
        description="maximum pixel shift allowed for nonrigid, relative to rigid",
    )
    outlier_detrend_window: float = Field(
        default=3.0,
        description="For outlier rejection in the xoff/yoff outputs of suite2p, the offsets are first de-trended "
        "with a median filter of this duration [seconds]. "
        "This value is ~30 or 90 samples in size for 11 and 31 Hz sampling rates respectively.",
    )
    outlier_maxregshift: float = Field(
        default=0.05,
        description="Units [fraction FOV dim]. After median-filter detrending, outliers more than this value are "
        "clipped to this value in x and y offset, independently. "
        "This is similar to Suite2P's internal maxregshift, but allows for low-frequency drift. "
        "Default value of 0.05 is typically clipping outliers to 512 * 0.05 = 25 pixels above or below the median trend.",
    )
    clip_negative: bool = Field(
        default=False,
        description="Whether or not to clip negative pixel values in output. Because the pixel values "
        "in the raw movies are set by the current coming off a photomultiplier tube, there can "
        "be pixels with negative values (current has a sign), possibly due to noise in the rig. "
        "Some segmentation algorithms cannot handle negative values in the movie, so we have this "
        "option to artificially set those pixels to zero.",
    )
    max_reference_iterations: int = Field(
        default=8,
        description="Maximum number of iterations for creating a reference image",
    )
    auto_remove_empty_frames: bool = Field(
        default=True,
        description="Automatically detect empty noise frames at the start and end of the movie. Overrides values set in "
        "trim_frames_start and trim_frames_end. Some movies arrive with otherwise quality data but contain a set of "
        "frames that are empty and contain pure noise. When processed, these frames tend to receive "
        "large random shifts that throw off motion border calculation. Turning on this setting automatically "
        "detects these frames before processing and removes them from reference image creation, automated smoothing "
        "parameter searches, and finally the motion border calculation. The frames are still written however any "
        "shift estimated is removed and their shift is set to 0 to avoid large motion borders.",
    )
    trim_frames_start: int = Field(
        default=0,
        description="Number of frames to remove from the start of the movie if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    trim_frames_end: int = Field(
        default=0,
        description="Number of frames to remove from the end of the movie if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    do_optimize_motion_params: bool = Field(
        default=False,
        description="Do a search for best parameters of smooth_sigma and smooth_sigma_time. Adds significant runtime cost to "
        "motion correction and should only be run once per experiment with the resulting parameters being stored "
        "for later use.",
    )
    smooth_sigma_time: int = Field(
        default=0,
        description="gaussian smoothing in time. If do_optimize_motion_params is set, this will be overridden",
    )
    smooth_sigma: float = Field(
        default=1.15,
        description="~1 good for 2P recordings, recommend 3-5 for 1P recordings. If do_optimize_motion_params is set, this will be overridden",
    )
    use_ave_image_as_reference: bool = Field(
        default=False,
        description="Only available if `do_optimize_motion_params` is set. After the a best set of smoothing parameters is found, "
        "use the resulting average image as the reference for the full registration. This can be used as two step "
        "registration by setting by setting smooth_sigma_min=smooth_sigma_max and "
        "smooth_sigma_time_min=smooth_sigma_time_max and steps=1.",
    )
    # Additional parameters that were hardcoded in the original code
    movie_lower_quantile: float = Field(
        default=0.1,
        description="Lower quantile threshold for avg projection histogram adjustment of movie",
    )
    movie_upper_quantile: float = Field(
        default=0.999,
        description="Upper quantile threshold for avg projection histogram adjustment of movie",
    )
    preview_frame_bin_seconds: float = Field(
        default=2.0,
        description="Before creating the webm, the movies will be averaged into bins of this many seconds",
    )
    preview_playback_factor: float = Field(
        default=10.0,
        description="The preview movie will playback at this factor times real-time",
    )
    n_batches: int = Field(
        default=20,
        description="Number of batches to load from the movie for smoothing parameter testing. Batches are evenly spaced throughout the movie.",
    )
    smooth_sigma_min: float = Field(
        default=0.65,
        description="Minimum value of the parameter search for smooth_sigma",
    )
    smooth_sigma_max: float = Field(
        default=2.15,
        description="Maximum value of the parameter search for smooth_sigma",
    )
    smooth_sigma_steps: int = Field(
        default=4,
        description="Number of steps to grid between smooth_sigma and smooth_sigma_max",
    )
    smooth_sigma_time_min: float = Field(
        default=0,
        description="Minimum value of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_max: float = Field(
        default=6,
        description="Maximum value of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_steps: int = Field(
        default=7,
        description="Number of steps to grid between smooth_sigma and smooth_sigma_time_max. Large values will add significant time to motion correction",
    )

    class Config:
        env_prefix = "MOTION_CORRECTION_"
        case_sensitive = False
        env_file = ".env"


def is_S3(file_path: str):
    """Test if a file is in a S3 bucket
    Parameters
    ----------
    file_path : str
        Location of the file.
    """
    return "s3fs" in subprocess.check_output(
        "df " + file_path + "| sed -n '2 p'", shell=True, text=True
    )


def h5py_byteorder_name(h5py_file: h5py.File, h5py_key: str) -> Tuple[str, str]:
    """Get the byteorder and name of the dataset in the h5py file.

    Parameters
    ----------
    h5py_file : h5py.File
        h5py file object
    h5py_key : str
        key to the dataset

    Returns
    -------
    str
        byteorder of the dataset
    str
        name of the dataset
    """
    with h5py.File(h5py_file, "r") as f:
        byteorder = f[h5py_key].dtype.byteorder
        name = f[h5py_key].dtype.name
    return byteorder, name


def tiff_byteorder_name(tiff_file: Path) -> Tuple[str, str]:
    """Get the byteorder and name of the dataset in the tiff file.

    Parameters
    ----------
    tiff_file : Path
        Location of the tiff file

    Returns
    -------
    str
        byteorder of the dataset
    str
        name of the dataset
    """
    with ScanImageTiffReader(tiff_file) as reader:
        byteorder = reader.data().dtype.byteorder
        name = reader.data().dtype.name
    return byteorder, name


def h5py_to_numpy(
    h5py_file: str,
    h5py_key: str,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Converts a h5py dataset to a numpy array

    Parameters
    ----------
    h5py_file: str
        h5py file path
    h5py_key : str
        key to the dataset
    trim_frames_start : int
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_end : int
        Number of frames to disregard from the end of the movie. Default 0.
    Returns
    -------
    np.ndarray
        numpy array
    """
    with h5py.File(h5py_file, "r") as f:
        n_frames = f[h5py_key].shape[0]
        if trim_frames_start > 0 or trim_frames_end > 0:
            return f[h5py_key][trim_frames_start : n_frames - trim_frames_end]
        else:
            return f[h5py_key][:]


@lru_cache(maxsize=None)
def _tiff_to_numpy(tiff_file: Path) -> np.ndarray:
    with ScanImageTiffReader(tiff_file) as reader:
        return reader.data()


def tiff_to_numpy(
    tiff_list: List[Path], trim_frames_start: int = 0, trim_frames_end: int = 0
) -> np.ndarray:
    """
    Converts a list of TIFF files to a single numpy array, with optional frame trimming.

    Parameters
    ----------
    tiff_list : List[str]
        List of TIFF file paths to process
    trim_frames_start : int, optional
        Number of frames to remove from the start (default: 0)
    trim_frames_end : int, optional
        Number of frames to remove from the end (default: 0)

    Returns
    -------
    np.ndarray
        Combined array of all TIFF data with specified trimming

    Raises
    ------
    ValueError
        If trim values exceed total number of frames or are negative
    """
    if trim_frames_start < 0 or trim_frames_end < 0:
        raise ValueError("Trim values must be non-negative")

    def get_total_frames(tiff_files: List[Path]) -> int:
        """Calculate total number of frames across all TIFF files."""
        total = 0
        for tiff in tiff_files:
            with ScanImageTiffReader(tiff) as reader:
                total += reader.shape()[0]
        return total

    # Validate trim parameters
    total_frames = get_total_frames(tiff_list)
    if trim_frames_start + trim_frames_end >= total_frames:
        raise ValueError(
            f"Invalid trim values: start ({trim_frames_start}) + end ({trim_frames_end}) "
            f"must be less than total frames ({total_frames})"
        )

    # Initialize variables for frame counting
    processed_frames = 0
    arrays_to_stack = []

    for tiff_path in tiff_list:
        with ScanImageTiffReader(tiff_path) as reader:
            current_frames = reader.shape()[0]
        start_idx = max(0, trim_frames_start - processed_frames)
        end_idx = current_frames

        if processed_frames + current_frames > total_frames - trim_frames_end:
            end_idx = total_frames - trim_frames_end - processed_frames

        if start_idx < end_idx:  # Only process if there are frames to include
            data = np.array(_tiff_to_numpy(tiff_path)[start_idx:end_idx])
            arrays_to_stack.append(data)

        processed_frames += current_frames

        # Break if we've processed all needed frames
        if processed_frames >= total_frames - trim_frames_end:
            break

    # Stack all arrays along the appropriate axis
    if not arrays_to_stack:
        raise ValueError("No frames remained after trimming")

    return (
        np.concatenate(arrays_to_stack, axis=0)
        if len(arrays_to_stack) > 1
        else arrays_to_stack[0]
    )


def load_initial_frames(
    file_path: Union[str, list],
    h5py_key: str,
    n_frames: int,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Load a subset of frames from the hdf5 data specified by file_path.

    Only loads frames between trim_frames_start and n_frames - trim_frames_end
    from the movie. If both are 0, load frames from the full movie.

    Parameters
    ----------
    file_path : str
        Location of the raw ophys, HDF5 data to load.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_frames : int
        Number of frames to load from the input HDF5 data.

    Returns
    -------
    frames : array-like, (n_frames, nrows, ncols)
        Frames selected from the input raw data linearly spaced in index of the
        time axis. If n_frames > tot_frames, a number of frames equal to
        tot_frames is returned.
    """
    if isinstance(file_path, str):
        array = h5py_to_numpy(file_path, h5py_key, trim_frames_start, trim_frames_end)
    elif isinstance(file_path, list):
        array = tiff_to_numpy(file_path, trim_frames_start, trim_frames_end)
    else:
        raise ValueError("File type not supported")
    # Total number of frames in the movie.
    tot_frames = array.shape[0]
    requested_frames = np.linspace(
        0, tot_frames, 1 + min(n_frames, tot_frames), dtype=int
    )[:-1]
    frames = array[requested_frames]
    return frames


def combine_images_with_individual_titles(
    image1_path: Path, image2_path: Path, output_path: Path, title1: str, title2: str
) -> None:
    """Combine two images side-by-side with padding and titles above each image.

    Parameters
    ----------
    image1_path : Path
        Path to the first image.
    image2_path : Path
        Path to the second image.
    output_path : Path
        Path to save the combined image.
    title1 : str
         Title text for the first image.
    title2 - str
        Title text for the second image.

    Returns
    -------
    None
    """
    # Open both images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure both images have the same height
    max_height = max(img1.height, img2.height)
    img1 = img1.resize((img1.width, max_height), Image.Resampling.LANCZOS)
    img2 = img2.resize((img2.width, max_height), Image.Resampling.LANCZOS)

    # Set padding and title height
    padding = 20
    title_height = 50  # Space for the titles

    # Calculate dimensions of the combined image
    combined_width = (
        img1.width + img2.width + padding * 3
    )  # Padding between and around images
    combined_height = max_height + padding * 2 + title_height

    # Create a new blank image with padding and room for the titles
    combined_image = Image.new(
        "RGB", (combined_width, combined_height), (255, 255, 255)
    )

    # Draw the titles
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype(
            "arial.ttf", 24
        )  # You can replace with a path to your desired font
    except IOError:
        font = (
            ImageFont.load_default()
        )  # Fallback to default font; may not match expected size

    # Title 1: Above the second image
    bbox1 = draw.textbbox((0, 0), title1, font=font)
    text_width1 = bbox1[2] - bbox1[0]
    text_y1 = padding
    text_x1 = padding + (img1.width - text_width1) // 2
    draw.text((text_x1, text_y1), title1, fill="black", font=font)

    # Title 2: Above the first image
    bbox2 = draw.textbbox((0, 0), title2, font=font)
    text_width2 = bbox2[2] - bbox2[0]
    text_x2 = padding * 2 + img2.width + (img2.width - text_width2) // 2
    text_y2 = padding
    draw.text((text_x2, text_y2), title2, fill="black", font=font)

    # Paste images into the new image
    img1_x = padding
    img2_x = img1_x + img1.width + padding
    img_y = padding + title_height
    combined_image.paste(img1, (img1_x, img_y))
    combined_image.paste(img2, (img2_x, img_y))

    # Save the result
    combined_image.save(output_path)


def serialize_registration_summary_qcmetric() -> None:
    """Serialize the registration summary QCMetric

    This function does not take any parameters.

    QCMetric is named 'registration_summary_metric.json' and is
    saved to the same directory as *_registration_summary.png.
    Ex: '/results/<unique_id>/motion_correction/'
    """

    file_path = next(output_dir.rglob("*_registration_summary.png"))

    # Remove '/results' from file_path
    reference_filepath = Path(*file_path.parts[2:])
    unique_id = reference_filepath.parts[0]

    metric = QCMetric(
        name=f"{unique_id} Registration Summary",
        description="Review the registration summary plot to ensure that the motion correction is accurate and sufficient.",
        reference=str(reference_filepath),
        status_history=[
            QCStatus(
                evaluator="Pending review", timestamp=dt.now(), status=Status.PENDING
            )
        ],
        value=DropdownMetric(
            value=[],
            options=[
                "Motion correction successful",
                "No motion correction applied",
                "Motion correction failed",
                "Motion correction partially successful",
            ],
            status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL],
        ),
    )

    with open(
        Path(file_path.parent) / f"{unique_id}_registration_summary_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def serialize_fov_quality_qcmetric() -> None:
    """Serialize the FOV Quality QCMetric

    This function does not take any parameters.

    QCMetric is named 'fov_quality_metric.json' and is
    saved to the same directory as *_maximum_projection.png.
    Ex: '/results/<unique_id>/motion_correction/'
    """

    avg_projection_file_path = next(output_dir.rglob("*_average_projection.png"))
    max_projection_file_path = next(output_dir.rglob("*_maximum_projection.png"))

    file_path = Path(str(max_projection_file_path).replace("maximum", "combined"))

    combine_images_with_individual_titles(
        avg_projection_file_path,
        max_projection_file_path,
        file_path,
        title1="Average Projection",
        title2="Maximum Projection",
    )

    # Remove /results from file_path
    reference_filepath = Path(*file_path.parts[2:])
    unique_id = reference_filepath.parts[0]

    metric = QCMetric(
        name=f"{unique_id} FOV Quality",
        description="Review the avg. and max. projections to ensure that the FOV quality is sufficient.",
        reference=str(reference_filepath),
        status_history=[
            QCStatus(
                evaluator="Pending review", timestamp=dt.now(), status=Status.PENDING
            )
        ],
        value=DropdownMetric(
            value=["Quality is sufficient"],
            options=[
                "Quality is sufficient",
                "Timeseries shuffled between planes",
                "Field of view associated with incorrect area and/or depth",
                "Paired plane cross talk: Extreme",
                "Paired plane cross-talk: Moderate",
            ],
            status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL, Status.FAIL],
        ),
    )

    with open(
        Path(file_path.parent) / f"{unique_id}_fov_quality_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def compute_residual_optical_flow(
    reg_pc: Union[h5py.Dataset, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the residual optical flow from the registration principal
    components.

    Parameters
    ----------
    reg_pc : Union[h5py.Dataset, np.ndarray]
        Registration principal components.

    Returns
    -------
    residual optical flow : np.ndarray
    average and max shifts : np.ndarray
    """

    regPC = reg_pc
    flows = np.zeros(regPC.shape[1:] + (2,), np.float32)
    for i in range(len(flows)):
        pclow, pchigh = regPC[:, i]
        flows[i] = cv2.calcOpticalFlowFarneback(
            pclow,
            pchigh,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=100,
            iterations=15,
            poly_n=5,
            poly_sigma=1.2 / 5,
            flags=0,
        )
    flows_norm = np.sqrt(np.sum(flows**2, -1))
    farnebackDX = np.transpose([flows_norm.mean((1, 2)), flows_norm.max((1, 2))])
    return flows, farnebackDX


def compute_crispness(
    mov_raw: Union[h5py.Dataset, np.ndarray], mov_corr: Union[h5py.Dataset, np.ndarray]
) -> List[float]:
    """Compute the crispness of the raw and corrected movie.

    Parameters
    ----------
    mov_raw : Union[h5py.Dataset, np.ndarray]
        Raw movie data.
    mov_corr : Union[h5py.Dataset, np.ndarray]
        Corrected movie data.

    Returns
    -------
    crispness of mean image : List[float]
    """
    return [
        np.sqrt(np.sum(np.array(np.gradient(mean_image(m))) ** 2))
        for m in (mov_raw, mov_corr)
    ]


def compute_reference(
    input_frames: np.ndarray,
    niter: int,
    maxregshift: float,
    smooth_sigma: float,
    smooth_sigma_time: float,
    mask_slope_factor: float = 3,
) -> np.ndarray:
    """Computes a stacked reference image from the input frames.

    Modified version of Suite2P's compute_reference function with no updating
    of input frames. Picks initial reference then iteratively aligns frames to
    create reference. This code does not reproduce the pre-processing suite2p
    does to data from 1Photon scopes. As such, if processing 1Photon data, the
    user should use the suite2p reference image creation.

    Parameters
    ----------
    input_frames : array-like, (n_frames, nrows, ncols)
        Set of frames to create a reference from.
    niter : int
        Number of iterations to perform when creating the reference image.
    maxregshift : float
        Maximum shift allowed as a fraction of the image width or height, which
        ever is longer.
    smooth_sigma : float
        Width of the Gaussian used to smooth the phase correlation between the
        reference and the frame with which it is being registered.
    smooth_sigma_time : float
        Width of the Gaussian used to smooth between multiple frames by before
        phase correlation.
    mask_slope_factor : int
        Factor to multiply ``smooth_sigma`` by when creating masks for the
        reference image during suite2p phase correlation. These masks down
        weight edges of the image. The default used in suite2p, where this
        method is adapted from, is 3.

    Returns
    -------
    refImg : array-like, (nrows, ncols)
        Reference image created from the input data.
    """
    # Get the dtype of the input frames to properly cast the final reference
    # image as the same type.
    frames_dtype = input_frames.dtype
    # Get initial reference image from suite2p.
    frames = remove_extrema_frames(input_frames)
    ref_image = pick_initial_reference(frames)

    # Determine how much to pad our frames by before shifting to prevent
    # wraps.
    pad_y = int(np.ceil(maxregshift * ref_image.shape[0]))
    pad_x = int(np.ceil(maxregshift * ref_image.shape[1]))

    for idx in range(niter):
        # Compute the number of frames to select in creating the reference
        # image. At most we select half to the input frames.
        nmax = int(frames.shape[0] * (1.0 + idx) / (2 * niter))

        # rigid Suite2P phase registration.
        ymax, xmax, cmax = phasecorr(
            data=apply_masks(
                frames,
                *compute_masks(
                    refImg=ref_image,
                    maskSlope=mask_slope_factor * smooth_sigma,
                ),
            ),
            cfRefImg=phasecorr_reference(
                refImg=ref_image,
                smooth_sigma=smooth_sigma,
            ),
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
        )

        # Find the indexes of the frames that are the most correlated and
        # select the first nmax.
        isort = np.argsort(-cmax)[:nmax]

        # Copy the most correlated frames so we don't shift the original data.
        # We pad this data to prevent wraps from showing up in the reference
        # image. We pad with NaN values to enable us to use nanmean and only
        # average those pixels that contain data in the average.
        max_corr_frames = np.pad(
            array=frames[isort].astype(float),
            pad_width=((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
            constant_values=np.nan,
        )
        max_corr_xmax = xmax[isort]
        max_corr_ymax = ymax[isort]
        # Apply shift to the copy of the frames.
        for frame, dy, dx in zip(max_corr_frames, max_corr_ymax, max_corr_xmax):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Create a new reference image from the highest correlated data.
        with warnings.catch_warnings():
            # Assuming the motion correction went well, there should be a lot
            # of empty values in the padded area around the frames. We suppress
            # warnings for these "Empty Slices" as they are expected.
            warnings.filterwarnings("ignore", "Mean of empty slice")
            ref_image = np.nanmean(max_corr_frames, axis=0)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-max_corr_ymax.mean())),
            dx=int(np.round(-max_corr_xmax.mean())),
        )
        # Clip the reference image back down to the original size and remove
        # any NaNs remaining. Throw warning if a NaN is found.
        ref_image = ref_image[pad_y:-pad_y, pad_x:-pad_x]
        if np.any(np.isnan(ref_image)):
            # NaNs can sometimes be left over from the image padding during the
            # first few iterations before the reference image has converged.
            # If there are still NaNs left after the final iteration, we
            # throw the following warning.
            if idx + 1 == niter:
                logging.warning(
                    f"Warning: {np.isnan(ref_image).sum()} NaN pixels were "
                    "found in the reference image on the final iteration. "
                    "Likely the image quality is low and shifting frames "
                    "failed. Setting NaN values to the image mean."
                )
            ref_image = np.nan_to_num(ref_image, nan=np.nanmean(ref_image), copy=False)
        ref_image = ref_image.astype(frames_dtype)

    return ref_image


def remove_extrema_frames(input_frames: np.ndarray, n_sigma: float = 3) -> np.ndarray:
    """Remove frames with extremum mean values from the frames used in
    reference image processing/creation.

    Likely these are empty frames of pure noise or very high intensity frames
    relative to mean.

    Parameters
    ----------
    input_frames : numpy.ndarray, (N, M, K)
        Set of frames to trim.
    n_sigma : float, optional
        Number of standard deviations to above which to clip. Default is 3
        which was found to remove all empty frames while preserving most
        frames.

    Returns
    -------
    trimmed_frames : numpy.ndarray, (N, M, K)
        Set of frames with the extremum frames removed.
    """
    frame_means = np.mean(input_frames, axis=(1, 2))
    _, low_cut, high_cut = sigmaclip(frame_means, low=n_sigma, high=n_sigma)
    trimmed_frames = input_frames[
        np.logical_and(frame_means > low_cut, frame_means < high_cut)
    ]
    return trimmed_frames


def optimize_motion_parameters(
    initial_frames: np.ndarray,
    smooth_sigmas: np.array,
    smooth_sigma_times: np.array,
    suite2p_args: dict,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    logger: Optional[Callable] = None,
) -> dict:
    """Loop over a range of parameters and select the best set from the
    max acutance of the final, average image.

    Parameters
    ----------
    initial_frames : numpy.ndarray, (N, M, K)
        Smaller subset of frames to create a reference image from.
    smooth_sigmas : numpy.ndarray, (N,)
        Array of suite2p smooth sigma values to attempt. Number of iterations
        will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    smooth_sigma_times : numpy.ndarray, (N,)
        Array of suite2p smooth sigma time values to attempt. Number of
        iterations will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    suite2p_args : dict
        A dictionary of suite2p configs containing at minimum:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Processing a large number of frames at once
        will likely result in running out of memory, hence processing in
        batches. Total returned size isn_batches * suit2p_args['batch_size'].
    logger : Optional[Callable]
        Function to print to stdout or a log.

    Returns
    -------
    best_result : dict
        A dict containing the final results of the search:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``ref_image``
            Reference Image created with the settings yielding the highest
            image acutance (numpy.ndarray, (N, M))
        ``acutance``
            Acutance of ``best_image``. (float)
        ``smooth_sigma``
            Value of ``smooth_sigma`` found to yield the best acutance (float).
        ``smooth_sigma_time``
            Value of ``smooth_sigma_time`` found to yield the best acutance
            (float).
    """
    best_results = {
        "acutance": 1e-16,
        "ave_image": np.array([]),
        "ref_image": np.array([]),
        "smooth_sigma": -1,
        "smooth_sigma_time": -1,
    }
    logger("Starting search for best smoothing parameters...")
    sub_frames = load_representative_sub_frames(
        suite2p_args["h5py"],
        suite2p_args["h5py_key"],
        trim_frames_start,
        trim_frames_end,
        n_batches=n_batches,
        batch_size=suite2p_args["batch_size"],
    )
    start_time = time()
    for param_spatial, param_time in product(smooth_sigmas, smooth_sigma_times):
        current_args = suite2p_args.copy()
        current_args["smooth_sigma"] = param_spatial
        current_args["smooth_sigma_time"] = param_time

        if logger:
            logger(
                f'\tTrying: smooth_sigma={current_args["smooth_sigma"]}, '
                f'smooth_sigma_time={current_args["smooth_sigma_time"]}'
            )

        ref_image = compute_reference(
            initial_frames,
            8,
            current_args["maxregshift"],
            current_args["smooth_sigma"],
            current_args["smooth_sigma_time"],
        )
        image_results = create_ave_image(
            ref_image,
            sub_frames.copy(),
            current_args,
            batch_size=suite2p_args["batch_size"],
        )
        ave_image = image_results["ave_image"]
        # Compute the acutance ignoring the motion border. Sharp motion
        # borders can potentially get rewarded with high acutance.
        current_acu = compute_acutance(
            ave_image,
            image_results["min_y"],
            image_results["max_y"],
            image_results["min_x"],
            image_results["max_x"],
        )

        if current_acu > best_results["acutance"]:
            best_results["acutance"] = current_acu
            best_results["ave_image"] = ave_image
            best_results["ref_image"] = ref_image
            best_results["smooth_sigma"] = current_args["smooth_sigma"]
            best_results["smooth_sigma_time"] = current_args["smooth_sigma_time"]
        if logger:
            logger(f"\t\tResulting acutance={current_acu:.4f}")
    if logger:
        logger(
            f"Found best motion parameters in {time() - start_time:.0f} "
            f'seconds, with image acutance={best_results["acutance"]:.4f}, '
            f'for parameters: smooth_sigma={best_results["smooth_sigma"]}, '
            f'smooth_sigma_time={best_results["smooth_sigma_time"]}'
        )
    return best_results


def load_representative_sub_frames(
    h5py_name,
    h5py_key,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    batch_size: int = 500,
):
    """Load a subset of frames spanning the full movie.

    Parameters
    ----------
    h5py_name : str
        Path to the h5 file to load frames from.
    h5py_key : str
        Name of the h5 dataset containing the movie.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Total returned size is
        n_batches * batch_size.
    batch_size : int, optional
        Number of frames to process at once. Total returned size is
        n_batches * batch_size.

    Returns
    -------
    """
    output_frames = []
    frame_fracts = np.arange(0, 1, 1 / n_batches)
    with h5py.File(h5py_name, "r") as h5_file:
        dataset = h5_file[h5py_key]
        total_frames = dataset.shape[0] - trim_frames_start - trim_frames_end
        if total_frames < n_batches * batch_size:
            return dataset[:]
        for percent_start in frame_fracts:
            frame_start = int(percent_start * total_frames + trim_frames_start)
            output_frames.append(dataset[frame_start : frame_start + batch_size])
    return np.concatenate(output_frames)


def create_ave_image(
    ref_image: np.ndarray,
    input_frames: np.ndarray,
    suite2p_args: dict,
    batch_size: int = 500,
) -> dict:
    """Run suite2p image motion correction over a full movie.

    Parameters
    ----------
    ref_image : numpy.ndarray, (N, M)
        Reference image to correlate with movie frames.
    input_frames : numpy.ndarray, (L, N, M)
        Frames to motion correct and compute average image/acutance of.
    suite2p_args : dict
        Dictionary of suite2p args containing:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
        ``"smooth_sigma"``
            Spatial Gaussian smoothing parameter used by suite2p to smooth
            frames before correlation. (float).
        ``"smooth_sigma_time"``
            Time Gaussian smoothing of frames to apply before correlation.
            (float).
    batch_size : int, optional
        Number of frames to process at once.

    Returns
    -------
    ave_image_dict : dict
        A dict containing the average image and motion border values:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``min_y``
            Minimum y allowed value in image array. Below this is motion
            border.
        ``max_y``
            Maximum y allowed value in image array. Above this is motion
            border.
        ``min_x``
            Minimum x allowed value in image array. Below this is motion
            border.
        ``max_x``
            Maximum x allowed value in image array. Above this is motion
            border.
    """
    ave_frame = np.zeros((ref_image.shape[0], ref_image.shape[1]))
    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0
    tot_frames = input_frames.shape[0]
    add_modify_required_parameters(suite2p_args)
    for start_idx in np.arange(0, tot_frames, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > tot_frames:
            end_idx = tot_frames
        frames = input_frames[start_idx:end_idx]
        frames, dy, dx, _, _, _, _ = register_frames(
            refAndMasks=ref_image, frames=frames, ops=suite2p_args
        )
        min_y = min(min_y, dy.min())
        max_y = max(max_y, dy.max())
        min_x = min(min_x, dx.min())
        max_x = max(max_x, dx.max())
        ave_frame += frames.sum(axis=0) / tot_frames

    return {
        "ave_image": ave_frame,
        "min_y": int(np.fabs(min_y)),
        "max_y": int(max_y),
        "min_x": int(np.fabs(min_x)),
        "max_x": int(max_x),
    }


def add_modify_required_parameters(suite2p_args: dict):
    """Check that minimum parameters needed by suite2p registration are
    available. If not add them to the suite2p_args dict.

    Additionally, make sure that nonrigid is set to false as are gridsearch
    of parameters above is not setup to use nonrigid.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary with potentially missing values.
    """
    if suite2p_args.get("1Preg") is None:
        suite2p_args["1Preg"] = False
    if suite2p_args.get("bidiphase") is None:
        suite2p_args["bidiphase"] = False
    if suite2p_args.get("nonrigid") is None:
        suite2p_args["nonrigid"] = False
    if suite2p_args.get("norm_frames") is None:
        suite2p_args["norm_frames"] = True
    # Don't use nonrigid for parameter search.
    suite2p_args["nonrigid"] = False


def compute_acutance(
    image: np.ndarray,
    min_cut_y: int = 0,
    max_cut_y: int = 0,
    min_cut_x: int = 0,
    max_cut_x: int = 0,
) -> float:
    """Compute the acutance (sharpness) of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute acutance of.
    min_cut_y : int
        Number of pixels to cut from the beginning of the y axis.
    max_cut_y : int
        Number of pixels to cut from the end of the y axis.
    min_cut_x : int
        Number of pixels to cut from the beginning of the x axis.
    max_cut_x : int
        Number of pixels to cut from the end of the x axis.

    Returns
    -------
    acutance : float
        Acutance of the image.
    """
    im_max_y, im_max_x = image.shape

    cut_image = image[
        min_cut_y : im_max_y - max_cut_y, min_cut_x : im_max_x - max_cut_x
    ]
    grady, gradx = np.gradient(cut_image)
    return (grady**2 + gradx**2).mean()


def check_and_warn_on_datatype(
    filepath: Union[Path, list],
    logger: Callable,
    filetype: str = "h5",
    h5py_key: str = "",
):
    """Suite2p assumes int16 types throughout code. Check that the input
    data is type int16 else throw a warning.

    Parameters
    ----------
    filepath : Union[Path, list]
        Path to the HDF5 containing the data.
    logger : Callable
        Logger to output logger warning to.
    filetype : str
        Type of file to check. Default is "h5".
    h5py_key : str
        Name of the dataset to check. Default is "".

    """
    if filetype == "h5":
        byteorder, name = h5py_byteorder_name(filepath, h5py_key)
    elif filetype == "tiff":
        byteorder, name = tiff_byteorder_name(filepath)
    else:
        raise ValueError("File type not supported")
    if byteorder == ">":
        logger(
            "Data byteorder is big-endian which may cause issues in "
            "suite2p. This may result in a crash or unexpected "
            "results."
        )
    if name != "int16":
        logger(
            f"Data type is {name} and not int16. Suite2p "
            "assumes int16 data as input and throughout codebase. "
            "Non-int16 data may result in unexpected results or "
            "crashes."
        )


def _mean_of_batch(i, array):
    return array[i : i + 1000].mean(axis=(1, 2))


def find_movie_start_end_empty_frames(
    filepath: Union[str, list[str]],
    h5py_key: str = "",
    n_sigma: float = 5,
    logger: Optional[Callable] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[int, int]:
    """Load a movie from HDF5 and find frames at the start and end of the
    movie that are empty or pure noise and 5 sigma discrepant from the
    average frame.

    If a non-contiguous set of frames is found, the code will return 0 for
    that half of the movie and throw a warning about the quality of the data.

    Parameters
    ----------
    h5py_name : str
        Name of the HDF5 file to load from.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_sigma : float
        Number of standard deviations beyond which a frame is considered an
        outlier and "empty".
    logger : Optional[Callable]
        Function to print warning messages to.
    n_jobs: Optional[int]
        The number of jobs to run in parallel.

    Returns
    -------
    trim_frames : Tuple[int, int]
        Tuple of the number of frames to cut from the start and end of the
        movie as (n_trim_start, n_trim_end).
    """
    if isinstance(filepath, str):
        array = h5py_to_numpy(filepath, h5py_key)
    else:
        array = tiff_to_numpy(filepath)
    # Find the midpoint of the movie.
    n_frames = array.shape[0]
    midpoint = n_frames // 2
    # We discover empty or extrema frames by comparing the mean of each frames
    # to the mean of the full movie.
    if n_jobs == 1 or n_frames < 2000:
        means = array[:].mean(axis=(1, 2))
    else:
        means = np.concatenate(
            ThreadPool(n_jobs).starmap(
                _mean_of_batch,
                product(range(0, n_frames, 1000), [array]),
            )
        )
    mean_of_frames = means.mean()

    # Compute a robust standard deviation that is not sensitive to the
    # outliers we are attempting to find.
    quart_low, quart_high = np.percentile(means, [25, 75])
    # Convert the inner quartile range to an estimate of the standard deviation
    # 0.6745 is the converting factor between the inner quartile and a
    # traditional standard deviation.
    std_est = (quart_high - quart_low) / (2 * 0.6745)

    # Get the indexes of the frames that are found to be n_sigma deviating.
    start_idxs = np.sort(
        np.argwhere(means[:midpoint] < mean_of_frames - n_sigma * std_est)
    ).flatten()
    end_idxs = (
        np.sort(
            np.argwhere(means[midpoint:] < mean_of_frames - n_sigma * std_est)
        ).flatten()
        + midpoint
    )

    # Get the total number of these frames.
    lowside = len(start_idxs)
    highside = len(end_idxs)

    # Check to make sure that the indexes found were only from the start/end
    # of the movie. If not, throw a warning and reset the number of frames
    # found to zero.
    if not np.array_equal(start_idxs, np.arange(0, lowside, dtype=start_idxs.dtype)):
        lowside = 0
        if logger is not None:
            logger(
                f"{n_sigma} sigma discrepant frames found outside the "
                "beginning of the movie. Please inspect the movie for data "
                "quality. Not trimming frames from the movie beginning."
            )
    if not np.array_equal(
        end_idxs,
        np.arange(n_frames - highside, n_frames, dtype=end_idxs.dtype),
    ):
        highside = 0
        if logger is not None:
            logger(
                f"{n_sigma} sigma discrepant frames found outside the end "
                "of the movie. Please inspect the movie for data quality. "
                "Not trimming frames from the movie end."
            )

    return (lowside, highside)


def reset_frame_shift(
    frames: np.ndarray,
    dy_array: np.ndarray,
    dx_array: np.ndarray,
    trim_frames_start: int,
    trim_frames_end: int,
):
    """Reset the frames of a movie and their shifts.

    Shifts the frame back to its original location and resets the shifts for
    those frames to (0, 0). Frames, dy_array, and dx_array are edited in
    place.

    Parameters
    ----------
    frames : numpy.ndarray, (N, M, K)
        Full movie to reset frames in.
    dy_array : numpy.ndarray, (N,)
        Array of shifts in the y direction for each frame of the movie.
    dx_array : numpy.ndarray, (N,)
        Array of shifts in the x direction for each frame of the movie.
    trim_frames_start : int
        Number of frames at the start of the movie that were identified as
        empty or pure noise.
    trim_frames_end : int
        Number of frames at the end of the movie that were identified as
        empty or pure noise.
    """
    for idx in range(trim_frames_start):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0

    for idx in range(frames.shape[0] - trim_frames_end, frames.shape[0]):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0


def projection_process(data: np.ndarray, projection: str = "max") -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError('projection can be "max" or "avg" not ' f"{projection}")
    return normalize_array(proj)


def identify_and_clip_outliers(
    data: np.ndarray, med_filter_size: int, thresh: int
) -> Tuple[np.ndarray, np.ndarray]:
    """given data, identify the indices of outliers
    based on median filter detrending, and a threshold

    Parameters
    ----------
    data: np.ndarray
        1D array of samples
    med_filter_size: int
        the number of samples for 'size' in
        scipy.ndimage.filters.median_filter
    thresh: int
        multipled by the noise estimate to establish a threshold, above
        which, samples will be marked as outliers.

    Returns
    -------
    data: np.ndarry
        1D array of samples, clipped to threshold around median-filtered data
    indices: np.ndarray
        the indices where clipping took place

    """
    data_filtered = median_filter(data, med_filter_size, mode="nearest")
    detrended = data - data_filtered
    indices = np.argwhere(np.abs(detrended) > thresh).flatten()
    data[indices] = np.clip(
        data[indices], data_filtered[indices] - thresh, data_filtered[indices] + thresh
    )
    return data, indices


def make_output_directory(output_dir: Path, plane: str = "") -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    plane: str
        plane number

    Returns
    -------
    output_dir: Path
        output directory
    """
    if not plane:
        output_dir = output_dir / "motion_correction"
    else:
        output_dir = output_dir / plane / "motion_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_frame_rate_platform_json(input_dir: str) -> float:
    """Get the frame rate from the platform json file.
    Platform json will need to get copied to each data directory throughout the pipeline

    Parameters
    ----------
    input_dir: str
        directory where file is located

    Returns
    -------
    frame_rate: float
        frame rate
    """
    try:
        try:
            platform_directory = os.path.dirname(os.path.dirname(input_dir))
            platform_json = glob.glob(f"{platform_directory}/*platform.json")
        except IndexError:
            raise IndexError
        with open(platform_json) as f:
            data = json.load(f)
        frame_rate = data["imaging_plane_groups"][0]["acquisition_framerate_Hz"]
        return frame_rate
    except IndexError as exc:
        raise Exception(f"Error: {exc}")


def write_data_process(
    metadata: dict,
    raw_movie: Union[str, Path],
    motion_corrected_movie: Union[str, Path],
    output_dir: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    raw_movie: str
        path to raw movies
    motion_corrected_movie: str
        path to motion corrected movies
    """
    if isinstance(raw_movie, Path):
        raw_movie = str(raw_movie)
    if isinstance(motion_corrected_movie, Path):
        motion_corrected_movie = str(motion_corrected_movie)
    data_proc = DataProcess(
        name=ProcessName.VIDEO_MOTION_CORRECTION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=str(raw_movie),
        output_location=str(motion_corrected_movie),
        code_url=(
            "https://github.com/AllenNeuralDynamics/"
            "aind-ophys-motion-correction/tree/main/code"
        ),
        parameters=metadata,
    )
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    with open(
        output_dir / f"{unique_id}_motion_correction_data_process.json", "w"
    ) as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def check_trim_frames(data):
    """Make sure that if the user sets auto_remove_empty_frames
    and timing frames is already requested, raise an error.
    """
    if data["auto_remove_empty_frames"] and (
        data["trim_frames_start"] > 0 or data["trim_frames_end"] > 0
    ):
        msg = (
            "Requested auto_remove_empty_frames but "
            "trim_frames_start > 0 or trim_frames_end > 0. Please "
            "either request auto_remove_empty_frames or manually set "
            "trim_frames_start/trim_frames_end if number of frames to "
            "trim is known."
        )
        raise ValueError(msg)
    return data


def make_png(
    max_proj_path: Path, avg_proj_path: Path, summary_df: pd.DataFrame, dst_path: Path
):
    """ """
    xo = np.abs(summary_df["x"]).max()
    yo = np.abs(summary_df["y"]).max()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 4)
    mx_ax = fig.add_subplot(gs[0:2, 0:2])
    av_ax = fig.add_subplot(gs[0:2, 2:4])
    xyax = fig.add_subplot(gs[2, :])
    corrax = fig.add_subplot(gs[3, :])

    for ax, im_path in zip([mx_ax, av_ax], [max_proj_path, avg_proj_path]):
        with Image.open(im_path) as im:
            ax.imshow(im, cmap="gray")
            sz = im.size
        ax.axvline(xo, color="r", linestyle="--")
        ax.axvline(sz[0] - xo, color="r", linestyle="--")
        ax.axhline(yo, color="g", linestyle="--")
        ax.axhline(sz[1] - yo, color="g", linestyle="--")
        ax.set_title(f"{im_path.parent}\n{im_path.name}", fontsize=8)

    xyax.plot(summary_df["x"], linewidth=0.5, color="r", label="xoff")
    xyax.axhline(xo, color="r", linestyle="--")
    xyax.axhline(-xo, color="r", linestyle="--")
    xyax.plot(summary_df["y"], color="g", linewidth=0.5, alpha=0.5, label="yoff")
    xyax.axhline(yo, color="g", linestyle="--")
    xyax.axhline(-yo, color="g", linestyle="--")
    xyax.legend(loc=0)
    xyax.set_ylabel("correction offset [pixels]")

    corrax.plot(summary_df["correlation"], color="k", linewidth=0.5, label="corrXY")
    corrax.set_xlabel("frame index")
    corrax.set_ylabel("correlation peak value")
    corrax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(dst_path)

    return dst_path


def make_nonrigid_png(
    output_path: Path, avg_proj_path: Path, summary_df: pd.DataFrame, dst_path: Path
):
    """ """
    nonrigid_y = np.array(list(map(eval, summary_df["nonrigid_y"])), dtype=np.float32)
    nonrigid_x = np.array(list(map(eval, summary_df["nonrigid_x"])), dtype=np.float32)
    nonrigid_corr = np.array(
        list(map(eval, summary_df["nonrigid_corr"])), dtype=np.float32
    )
    ops = json.loads(h5py.File(output_path)["metadata"][()].decode())["suite2p_args"]
    with Image.open(avg_proj_path) as im:
        Ly, Lx = im.size
    yblock, xblock = make_blocks(Ly=Ly, Lx=Lx, block_size=ops["block_size"])[:2]
    nblocks = len(xblock)

    fig = plt.figure(figsize=(22, 3 * nblocks))
    gs = fig.add_gridspec(25 * nblocks, 6)
    for i in range(nblocks):
        av_ax = fig.add_subplot(gs[25 * i : 25 * i + 20, 0])
        xyax = fig.add_subplot(gs[25 * i : 25 * i + 10, 1:])
        corrax = fig.add_subplot(gs[25 * i + 10 : 25 * i + 20, 1:])

        with Image.open(avg_proj_path) as im:
            av_ax.imshow(im, cmap="gray")
            sz = im.size
            av_ax.set_ylim(0, sz[0])
            av_ax.set_xlim(0, sz[1])
        for x in xblock[i]:
            av_ax.vlines(x, *yblock[i], color="r", linestyle="--")
        for y in yblock[i]:
            av_ax.hlines(y, *xblock[i], color="g", linestyle="--")

        xyax.plot(nonrigid_x[:, i], linewidth=0.5, color="r", label="xoff")
        xyax.plot(nonrigid_y[:, i], color="g", linewidth=0.5, alpha=0.5, label="yoff")
        if i == 0:
            xyax.legend(loc=0)
        xyax.set_xticks([])
        xyax.set_xlim(0, nonrigid_x.shape[0])
        xyax.set_ylabel("offset [pixels]")

        corrax.plot(nonrigid_corr[:, i], color="k", linewidth=0.5, label="corrXY")
        corrax.set_xlim(0, nonrigid_x.shape[0])
        corrax.set_xlabel("frame index")
        corrax.set_ylabel("correlation")
        if i == 0:
            corrax.legend(loc=0)
    fig.savefig(dst_path, bbox_inches="tight")

    return dst_path


def downsample_normalize(
    movie_path: Path,
    frame_rate: float,
    bin_size: float,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    """reads in a movie (nframes x nrows x ncols), downsamples,
    creates an average projection, and normalizes according to
    quantiles in that projection.

    Parameters
    ----------
    movie_path: Path
        path to an h5 file, containing an (nframes x nrows x ncol) dataset
        named 'data'
    frame_rate: float
        frame rate of the movie specified by 'movie_path'
    bin_size: float
        desired duration in seconds of a downsampled bin, i.e. the reciprocal
        of the desired downsampled frame rate.
    lower_quantile: float
        arg supplied to `np.quantile()` to determine lower cutoff value from
        avg projection for normalization.
    upper_quantile: float
        arg supplied to `np.quantile()` to determine upper cutoff value from
        avg projection for normalization.

    Returns
    -------
    ds: np.ndarray
        a downsampled and normalized array

    Notes
    -----
    This strategy was satisfactory in the labeling app for maintaining
    consistent visibility.

    """
    if isinstance(movie_path, Path):
        ds = downsample_h5_video(
            movie_path, input_fps=frame_rate, output_fps=1.0 / bin_size
        )
    else:
        ds = downsample_array(
            movie_path, input_fps=frame_rate, output_fps=1.0 / bin_size
        )
    avg_projection = ds.mean(axis=0)
    lower_cutoff, upper_cutoff = np.quantile(
        avg_projection.flatten(), (lower_quantile, upper_quantile)
    )
    ds = normalize_array(ds, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff)
    return ds


def flow_png(output_path: Path, dst_path: str, iPC: int = 0):
    with h5py.File(output_path) as f:
        regPC = f["reg_metrics/regPC"]
        tPC = f["reg_metrics/tPC"]
        flows = f["reg_metrics/farnebackROF"]
        flow_ds = np.array(
            [
                cv2.resize(flows[iPC, :, :, a], dsize=None, fx=0.1, fy=0.1)
                for a in (0, 1)
            ]
        )
        flow_ds_norm = np.sqrt(np.sum(flow_ds**2, 0))
        # redo Suite2p's PCA-based frame selection
        n_frames, Ly, Lx = f["data"].shape
        nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames)
        inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
        nlowhigh = np.minimum(300, int(n_frames / 2))
        isort = np.argsort(tPC, axis=0)

        for k in (0, 1):
            f, a = plt.subplots(2, figsize=(5, 6))
            a[0].set_position([0.08, 0.92, 0.88, 0.08])
            a[0].hist(
                np.sort(inds[isort[-nlowhigh:, iPC] if k else isort[:nlowhigh, iPC]]),
                50,
            )
            a[0].set_title(
                "averaged frames for " + ("$PC_{high}$" if k else "$PC_{low}$")
            )
            a[1].set_position([0, 0, 1, 0.9])
            vmin = np.min(regPC[1 if k else 0, iPC])
            vmax = 5 * np.median(regPC[1 if k else 0, iPC]) - 4 * vmin
            a[1].imshow(regPC[1 if k else 0, iPC], cmap="gray", vmin=vmin, vmax=vmax)
            a[1].axis("off")
            plt.savefig(
                dst_path + (f"_PC{iPC}low.png", f"_PC{iPC}high.png")[k],
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
        f, a = plt.subplots(2, figsize=(5, 6))
        a[0].set_position([0.06, 0.95, 0.9, 0.05])
        a[1].set_position([0, 0, 1, 0.9])
        im = a[1].quiver(
            *flow_ds[:, ::-1], flow_ds_norm[::-1]
        )  # imshow puts origin [0,0] in upper left
        a[1].axis("off")
        plt.colorbar(im, cax=a[0], location="bottom")
        a[0].set_title("residual optical flow")
        plt.savefig(
            dst_path + f"_PC{iPC}rof.png", format="png", dpi=300, bbox_inches="tight"
        )


def get_frame_rate_from_sync(sync_file, platform_data) -> float:
    """Calculate frame rate from sync file
    Parameters
    ----------
    sync_file: str
        path to sync file
    platform_data: dict
        platform data from platform.json
    Returns
    -------
    frame_rate_hz: float
        frame rate in Hz
    """
    labels = ["vsync_2p", "2p_vsync"]  # older versions of sync may 2p_vsync label
    imaging_groups = len(
        platform_data["imaging_plane_groups"]
    )  # Number of imaging plane groups for frequency calculation
    frame_rate_hz = None
    for i in labels:
        logging.info(f"Pulling framerate from sync file: {sync_file}")
        sync_data = Sync(sync_file)
        try:
            rising_edges = sync_data.get_rising_edges(i, units="seconds")
            image_freq = 1 / (np.mean(np.diff(rising_edges)))
            frame_rate_hz = image_freq / imaging_groups

        except ValueError:
            pass
    sync_data.close()
    if not frame_rate_hz:
        raise ValueError(
            f"Frame rate no acquired, line labels: {sync_data.line_labels}"
        )
    return frame_rate_hz


def multiplane_motion_correction(data_dir: Path, output_dir: Path, debug: bool = False):
    """Process multiplane data for suite2p parameters

    Parameters
    ----------
    data_dir: Path
        path to h5 file
    output_dir: Path
        output directory
    debug: bool
        run in debug mode
    Returns
    -------
    h5_file: Path
        path to h5 file
    output_dir: Path
        output directory
    frame_rate_hz: float
        frame rate in Hz
    """
    pattern = re.compile(r"^V.*\d+$")
    matching_files = [d for d in data_dir.rglob("*.txt") if pattern.match(d.stem)]
    if len(matching_files) > 0:
        with open(matching_files[0], "r") as f:
            h5_file = f.read()
        h5_file = next(data_dir.rglob(h5_file))
        unique_id = h5_file.stem
    else:
        h5_dir = [i for i in data_dir.rglob("*V*") if i.is_dir()][0]
        unique_id = h5_dir.name
        h5_file = [i for i in h5_dir.glob(f"{h5_dir.name}.h5")][0]
    logging.info("Found raw time series to process %s", h5_file)
    session_fp = next(data_dir.rglob("session.json"), "")
    if not session_fp:
        raise f"Could not locate session.json in {session_fp}"
    with open(session_fp) as f:
        session_data = json.load(f)
    output_dir = make_output_directory(output_dir, unique_id)
    try:
        frame_rate_hz = float(
            session_data["data_streams"][0]["ophys_fovs"][0]["frame_rate"]
        )
    except KeyError:
        logging.warning(
            "Frame rate not found in session.json, pulling from platform.json"
        )
        platform_json = next(data_dir.rglob("*platform.json"))
        with open(platform_json, "r") as j:
            platform_data = json.load(j)
        sync_file = [i for i in data_dir.rglob("platform_data['sync_file']")][0]
        frame_rate_hz = get_frame_rate_from_sync(sync_file, platform_data)
    if debug:
        logging.info(f"Running in debug mode....")
        raw_data = h5py.File(h5_file, "r")
        frames_6min = int(360 * float(frame_rate_hz))
        trimmed_data = raw_data["data"][:frames_6min]
        raw_data.close()
        trimmed_fn = Path("../scratch") / f"{unique_id}.h5"
        with h5py.File(trimmed_fn, "w") as f:
            f.create_dataset("data", data=trimmed_data)
        h5_file = trimmed_fn
    return h5_file, output_dir, frame_rate_hz


def update_suite2p_args_reference_image(
    suite2p_args: dict, args: dict, reference_image_fp=None
):
    """Update the suite2p_args dictionary with the reference image.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary.
    args : dict
        Dictionary of arguments from the command line.
    reference_image_fp : Path
        Path to the reference image to use. Default is None.

    Returns
    -------
    suite2p_args : dict
        Updated suite2p_args dictionary.
    args : dict
        Updated args dictionary.
    """
    # Use our own version of compute_reference to create the initial
    # reference image used by suite2p.
    logger.info(
        f'Loading {suite2p_args["nimg_init"]} frames ' "for reference image creation."
    )
    if reference_image_fp:
        initial_frames = load_initial_frames(
            file_path=reference_image_fp,
            h5py_key=suite2p_args["h5py_key"],
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
        )

    else:
        if suite2p_args.get("h5py", None):
            file_path = suite2p_args["h5py"]
            h5py_key = suite2p_args["h5py_key"]
        else:
            file_path = suite2p_args["tiff_list"]
            h5py_key = None
        initial_frames = load_initial_frames(
            file_path=file_path,
            h5py_key=h5py_key,
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
        )

    if args["do_optimize_motion_params"]:
        logger.info("Attempting to optimize registration parameters Using:")
        logger.info(
            "\tsmooth_sigma range: "
            f'{args["smooth_sigma_min"]} - '
            f'{args["smooth_sigma_max"]}, '
            f'steps: {args["smooth_sigma_steps"]}'
        )
        logger.info(
            "\tsmooth_sigma_time range: "
            f'{args["smooth_sigma_time_min"]} - '
            f'{args["smooth_sigma_time_max"]}, '
            f'steps: {args["smooth_sigma_time_steps"]}'
        )

        # Create linear spaced arrays for the range of smooth
        # parameters to try.
        smooth_sigmas = np.linspace(
            args["smooth_sigma_min"],
            args["smooth_sigma_max"],
            args["smooth_sigma_steps"],
        )
        smooth_sigma_times = np.linspace(
            args["smooth_sigma_time_min"],
            args["smooth_sigma_time_max"],
            args["smooth_sigma_time_steps"],
        )

        optimize_result = optimize_motion_parameters(
            initial_frames=initial_frames,
            smooth_sigmas=smooth_sigmas,
            smooth_sigma_times=smooth_sigma_times,
            suite2p_args=suite2p_args,
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
            n_batches=args["n_batches"],
            logger=logger.info,
        )
        if args["use_ave_image_as_reference"]:
            suite2p_args["refImg"] = optimize_result["ave_image"]
        else:
            suite2p_args["refImg"] = optimize_result["ref_image"]
        suite2p_args["smooth_sigma"] = optimize_result["smooth_sigma"]
        suite2p_args["smooth_sigma_time"] = optimize_result["smooth_sigma_time"]
    else:
        # Create the initial reference image and store it in the
        # suite2p_args dictionary. 8 iterations is the current default
        # in suite2p.
        tic = -time()
        logger.info("Creating custom reference image...")
        suite2p_args["refImg"] = compute_reference(
            input_frames=initial_frames,
            niter=args["max_reference_iterations"],
            maxregshift=suite2p_args["maxregshift"],
            smooth_sigma=suite2p_args["smooth_sigma"],
            smooth_sigma_time=suite2p_args["smooth_sigma_time"],
        )
        tic += time()
        logger.info(f"took {tic}s")
    return suite2p_args, args


def generate_single_plane_reference(fp: Path, session) -> Path:
    """Generate virtual movies for Bergamo data

    Parameters
    ----------
    fp: Path
        path to h5 file
    session: dict
        session metadata
    Returns
    -------
    Path
        path to reference image
    """
    with h5py.File(fp, "r") as f:
        # take the first bci epoch to save out reference image TODO
        tiff_stems = json.loads(f["epoch_locations"][:][0])
        bci_epochs = [
            i
            for i in session["stimulus_epochs"]
            if i["stimulus_name"] == "single neuron BCI conditioning"
        ]
        bci_epoch_loc = [i["output_parameters"]["tiff_stem"] for i in bci_epochs][0]
        frame_length = tiff_stems[bci_epoch_loc][1] - tiff_stems[bci_epoch_loc][0]
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(
            shape=(frame_length, *f["data"].shape[1:]), dtype=f["data"].dtype
        )
        layout[0:frame_length] = vsource[
            tiff_stems[bci_epoch_loc][0] : tiff_stems[bci_epoch_loc][1]
        ]

        with h5py.File("../scratch/reference_image.h5", "w") as ref:

            ref.create_virtual_dataset("data", layout)
    return Path("../scratch/reference_image.h5")


def singleplane_motion_correction(
    h5_file: Path, output_dir: Path, session, unique_id: str, debug: bool = False
):
    """Process single plane data for suite2p parameters

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    output_dir: Path
        output directory
    session: dict
        session metadata
    unique_id: str
        experiment id from data description
    debug: bool

    Returns
    -------
    h5_file: str
        path to h5 file
    output_dir: Path
        output directory
    reference_image_fp: str
        path to reference image
    """
    if not h5_file.is_file():
        h5_file = [f for f in h5_file.rglob("*.h5") if unique_id in str(f)][0]
    output_dir = make_output_directory(output_dir, unique_id)
    reference_image_fp = generate_single_plane_reference(h5_file, session)
    if debug:
        stem = h5_file.stem
        debug_file = Path("../scratch") / f"{stem}_debug.h5"
        with h5py.File(h5_file, "r") as f:
            data = f["data"][:5000]
            trial_locations = f["trial_locations"][()]
            epoch_filenames = f["epoch_locations"][()]
        with h5py.File(debug_file, "a") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("trial_locations", data=trial_locations)
            f.create_dataset("epoch_locations", data=epoch_filenames)
        h5_file = debug_file
    with h5py.File(h5_file, "r") as f:
        trial_locations = json.loads(f["trial_locations"][:][0])
        epoch_locations = json.loads(f["epoch_locations"][:][0])
    with open(output_dir / "trial_locations.json", "w") as j:
        json.dump(trial_locations, j)
    with open(output_dir / "epoch_locations.json", "w") as j:
        json.dump(epoch_locations, j)

    return str(h5_file), output_dir, str(reference_image_fp)


def get_frame_rate(session: dict):
    """Attempt to pull frame rate from session.json
    Returns none if frame rate not in session.json

    Parameters
    ----------
    session: dict
        session metadata

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


if __name__ == "__main__":  # pragma: nocover
    # Set the log level and name the logger
    logger = logging.getLogger("Suite2P motion correction")
    logger.setLevel(logging.INFO)
    start_time = dt.now()
    # Parse command-line arguments
    parser = MotionCorrectionSettings()
    # General settings
    data_dir = parser.input_dir
    output_dir = parser.output_dir
    session_fp = next(data_dir.rglob("session.json"))
    description_fp = next(data_dir.rglob("data_description.json"))
    subject_fp = next(data_dir.rglob("subject.json"))
    with open(session_fp, "r") as j:
        session = json.load(j)
    with open(description_fp, "r") as j:
        data_description = json.load(j)
    with open(subject_fp, "r") as j:
        subject = json.load(j)
    subject_id = subject.get("subject_id", "")
    name = data_description.get("name", "")
    setup_logging(
        "aind-ophys-motion-correction", mouse_id=subject_id, session_name=name
    )
    frame_rate_hz = get_frame_rate(session)

    reference_image_fp = ""

    if parser.data_type == "TIFF":
        unique_id = "plane_0"
        try:
            input_file = next(data_dir.rglob("*/pophys"))
        except StopIteration:
            input_file = next(data_dir.rglob("pophys"))
        output_dir = make_output_directory(output_dir, unique_id)
    elif parser.data_type.lower() == "h5":
        unique_id = "MOp2_3_0"  # TODO: remove when upgrade to data-schema v2
        if "Bergamo" in session.get("rig_id", ""):
            h5_file, output_dir, reference_image_fp = singleplane_motion_correction(
                data_dir, output_dir, session, unique_id, debug=parser.debug
            )
        else:
            unique_id = "_".join(str(data_description["name"]).split("_")[-3:])
            h5_file, output_dir, frame_rate_hz = multiplane_motion_correction(
                data_dir, output_dir, debug=parser.debug
            )
        input_file = str(h5_file)
    else:
        raise ValueError(
            f"Data type {parser.data_type} not supported. "
            "Please use 'TIFF' or 'h5'."
        )

    # We convert to dictionary
    args = vars(parser)
    args["input_dir"] = str(args["input_dir"])
    args["output_dir"] = str(args["output_dir"])
    if not frame_rate_hz:
        frame_rate_hz = parser.frame_rate
        logging.warning("User input frame rate used. %s", frame_rate_hz)
    reference_image = None
    args["refImg"] = []
    if reference_image_fp:
        args["refImg"] = [reference_image_fp]
    # We construct the paths to the outputs
    if parser.data_type == "TIFF":
        basename = unique_id
    else:
        basename = os.path.basename(input_file)
    args["movie_frame_rate_hz"] = frame_rate_hz
    for key, default in (
        ("motion_corrected_output", "_registered.h5"),
        ("motion_diagnostics_output", "_motion_transform.csv"),
        ("max_projection_output", "_maximum_projection.png"),
        ("avg_projection_output", "_average_projection.png"),
        ("registration_summary_output", "_registration_summary.png"),
        ("motion_correction_preview_output", "_motion_preview.webm"),
        ("output_json", "_motion_correction_output.json"),
    ):
        args[key] = os.path.join(
            str(output_dir), os.path.splitext(basename)[0] + default
        )

    # These are hardcoded parameters of the wrapper. Those are tracked but
    # not exposed.

    # Lower quantile threshold for avg projection histogram adjustment of movie (default: 0.1)
    args["movie_lower_quantile"] = 0.1
    # Upper quantile threshold for avg projection histogram adjustment of movie (default: 0.999)
    args["movie_upper_quantile"] = 0.999
    # Before creating the webm, the movies will be averaged into bins of this many seconds.
    args["preview_frame_bin_seconds"] = 2.0
    # The preview movie will playback at this factor times real-time.
    args["preview_playback_factor"] = 10.0

    # Number of batches to load from the movie for smoothing parameter testing.
    # Batches are evenly spaced throughout the movie.
    args["n_batches"] = 20
    # Minimum value of the parameter search for smooth_sigma.
    args["smooth_sigma_min"] = 0.65
    # Maximum value of the parameter search for smooth_sigma.
    args["smooth_sigma_max"] = 2.15
    # Number of steps to grid between smooth_sigma and smooth_sigma_max.
    args["smooth_sigma_steps"] = 4
    # Minimum value of the parameter search for smooth_sigma_time.
    args["smooth_sigma_time_min"] = 0
    # Maximum value of the parameter search for smooth_sigma_time.
    args["smooth_sigma_time_max"] = 6
    # Number of steps to grid between smooth_sigma and smooth_sigma_time_max.
    # Large values will add significant time motion correction
    args["smooth_sigma_time_steps"] = 7

    # This is part of a complex scheme to pass an image that is a bit too
    # complicated. Will remove when tested.

    # Set suite2p parser.
    suite2p_args = suite2p.default_ops()

    # Here we overwrite the parameters for suite2p that will not change in our
    # processing pipeline. These are parameters that are not exposed to
    # minimize code length. Those are not set to default.
    if parser.data_type == "h5":
        suite2p_args["h5py"] = str(input_file)
    else:
        suite2p_args["data_path"] = str(input_file)
        suite2p_args["look_one_level_down"] = True
        suite2p_args["tiff_list"] = [str(i) for i in input_file.glob("*.tif*")]
    suite2p_args["roidetect"] = False
    suite2p_args["do_registration"] = parser.do_registration
    suite2p_args["align_by_chan"] = parser.align_by_chan
    suite2p_args["reg_tif"] = False  # We save our own outputs here
    suite2p_args["nimg_init"] = (
        500  # Nb of images to compute reference. This value is a bit high. Suite2p has it at 300 normally
    )
    suite2p_args["maxregshift"] = (
        parser.maxregshift  # Max allowed registration shift as a fraction of frame max(width and height)
    )
    # These parameters are at the same value as suite2p default. This is just here
    # to make it clear we need those parameters to be at the same value as
    # suite2p default but those lines could be deleted.
    suite2p_args["maxregshiftNR"] = (
        parser.maxregshiftNR  # Maximum shift allowed in pixels for a block in rigid registration.
    )
    suite2p_args["batch_size"] = (
        parser.batch_size
    )  # Number of frames to process at once
    if suite2p_args.get("h5py", ""):
        suite2p_args["h5py_key"] = "data"  # h5 path in the file.
    suite2p_args["smooth_sigma"] = (
        parser.smooth_sigma  # Standard deviation in pixels of the gaussian used to smooth the phase correlation.
    )
    suite2p_args["smooth_sigma_time"] = (
        parser.smooth_sigma_time  # "Standard deviation in time frames of the gaussian used to smooth the data before phase correlation is computed
    )
    suite2p_args["nonrigid"] = (
        parser.nonrigid
    )  # If True, non-rigid registration is performed.
    suite2p_args["block_size"] = (
        parser.block_size
    )  # Block dimensions in y, x in pixels.
    suite2p_args["snr_thresh"] = (
        parser.snr_thresh  # If a block is below the above snr threshold. Apply smoothing to the block.
    )

    # This is to overwrite image reference creation.
    suite2p_args["refImg"] = args["refImg"]
    suite2p_args["force_refImg"] = args["force_refImg"]

    # if data is in a S3 bucket, copy it to /scratch for faster access
    if suite2p_args.get("h5py", ""):
        if is_S3(suite2p_args["h5py"]):
            dst = "/scratch/" + Path(suite2p_args["h5py"]).name
            logger.info(f"copying {suite2p_args['h5py']} from S3 bucket to {dst}")
            shutil.copy(suite2p_args["h5py"], dst)
            suite2p_args["h5py"] = dst

    if suite2p_args.get("tiff_list", ""):
        check_and_warn_on_datatype(
            filepath=suite2p_args["tiff_list"][0],
            logger=logger.warning,
            filetype="tiff",
        )
    else:
        check_and_warn_on_datatype(
            filepath=suite2p_args["h5py"],
            logger=logger.warning,
            filetype="h5",
            h5py_key=suite2p_args["h5py_key"],
        )

    if args["auto_remove_empty_frames"]:
        logger.info(
            "Attempting to find empty frames at the start and end of the movie."
        )
        if suite2p_args.get("tiff_list", ""):
            lowside, highside = find_movie_start_end_empty_frames(
                filepath=suite2p_args["tiff_list"],
                logger=logger.warning,
            )
        else:
            lowside, highside = find_movie_start_end_empty_frames(
                filepath=suite2p_args["h5py"],
                h5py_key=suite2p_args["h5py_key"],
                logger=logger.warning,
            )
        args["trim_frames_start"] = lowside
        args["trim_frames_end"] = highside
        logger.info(f"Found ({lowside}, {highside}) at the start/end of the movie.")

    if suite2p_args["force_refImg"] and len(suite2p_args["refImg"]) == 0:
        suite2p_args, args = update_suite2p_args_reference_image(
            suite2p_args,
            args,
        )
    if reference_image_fp:
        suite2p_args, args = update_suite2p_args_reference_image(
            suite2p_args, args, reference_image_fp=reference_image_fp
        )

        # register with Suite2P
        logger.info(f"attempting to motion correct {suite2p_args['h5py']}")
    # make a tempdir for Suite2P's output
    tmp_dir = tempfile.TemporaryDirectory(dir=args["tmp_dir"])
    tdir = tmp_dir.name
    suite2p_args["save_path0"] = tdir
    logger.info(f"Running Suite2P with output going to {tdir}")
    args["tmp_dir"] = str(args["tmp_dir"])
    # Make a copy of the args to remove the NumpyArray, refImg, as
    # numpy.ndarray can't be serialized with json. Converting to list
    # and writing to the logger causes the output to be unreadable.
    copy_of_args = copy.deepcopy(suite2p_args)
    copy_of_args.pop("refImg")

    msg = f"running Suite2P v{suite2p.version} with args\n"
    msg += f"{json.dumps(copy_of_args, indent=2, sort_keys=True)}\n"
    logger.info(msg)

    # If we are using a external reference image (including our own
    # produced by compute_referece) communicate this in the log.
    if suite2p_args["force_refImg"]:
        logger.info(f"\tUsing custom reference image: {suite2p_args['refImg']}")

    if suite2p_args.get("h5py", ""):
        suite2p_args["h5py"] = suite2p_args["h5py"]
    suite2p.run_s2p(suite2p_args)
    data_path = ""
    if suite2p_args.get("h5py", ""):
        data_path = suite2p_args["h5py"][0]
    else:
        data_path = suite2p_args["data_path"][0]

    if not data_path:
        raise ValueError("No data path found in suite2p_args")
    bin_path = list(Path(tdir).rglob("data.bin"))[0]
    ops_path = list(Path(tdir).rglob("ops.npy"))[0]
    # Suite2P ops file contains at least the following keys:
    # ["Lx", "Ly", "nframes", "xrange", "yrange", "xoff", "yoff",
    #  "corrXY", "meanImg"]
    ops = np.load(ops_path, allow_pickle=True).item()

    # identify and clip offset outliers
    detrend_size = int(frame_rate_hz * args["outlier_detrend_window"])
    xlimit = int(ops["Lx"] * args["outlier_maxregshift"])
    ylimit = int(ops["Ly"] * args["outlier_maxregshift"])
    logger.info(
        "checking whether to clip where median-filtered "
        "offsets exceed (x,y) limits of "
        f"({xlimit},{ylimit}) [pixels]"
    )
    delta_x, x_clipped = identify_and_clip_outliers(
        np.array(ops["xoff"]), detrend_size, xlimit
    )
    delta_y, y_clipped = identify_and_clip_outliers(
        np.array(ops["yoff"]), detrend_size, ylimit
    )
    clipped_indices = list(set(x_clipped).union(set(y_clipped)))
    logger.info(f"{len(x_clipped)} frames clipped in x")
    logger.info(f"{len(y_clipped)} frames clipped in y")
    logger.info(f"{len(clipped_indices)} frames will be adjusted for clipping")

    # accumulate data from Suite2P's binary file
    data = suite2p.io.BinaryFile(ops["Ly"], ops["Lx"], bin_path).data

    if args["clip_negative"]:
        data[data < 0] = 0
        data = np.uint16(data)

    # anywhere we've clipped the offset, translate the frame
    # using Suite2P's shift_frame by the difference resulting
    # from clipping, for example, if Suite2P moved a frame
    # by 100 pixels, and we have clipped that to 30, this will
    # move it -70 pixels
    if not suite2p_args["nonrigid"]:
        # If using non-rigid, we can't modify the output frames and have
        # the shifts make sense. Hence we don't calculate which shifts
        # to clip given that the shift will no longer make sense.
        for frame_index in clipped_indices:
            dx = delta_x[frame_index] - ops["xoff"][frame_index]
            dy = delta_y[frame_index] - ops["yoff"][frame_index]
            data[frame_index] = suite2p.registration.rigid.shift_frame(
                data[frame_index], dy, dx
            )

    # If we found frames that are empty at the end and beginning of the
    # movie, we reset their motion shift and set their shifts to 0.
    reset_frame_shift(
        data,
        delta_y,
        delta_x,
        args["trim_frames_start"],
        args["trim_frames_end"],
    )
    # Create a boolean lookup of frames we reset as they were found
    # to be empty.
    is_valid = np.ones(len(data), dtype="bool")
    is_valid[: args["trim_frames_start"]] = False
    is_valid[len(data) - args["trim_frames_end"] :] = False

    # write the hdf5
    with h5py.File(args["motion_corrected_output"], "w") as f:
        f.create_dataset("data", data=data, chunks=(1, *data.shape[1:]))
        # Sort the reference image used to register. If we do not used
        # our custom reference image creation code, this dataset will
        # be empty.
        f.create_dataset("ref_image", data=suite2p_args["refImg"])
        # Write a copy of the configuration output of this dataset into the
        # HDF5 file.
        args_copy = copy.deepcopy(args)
        suite_args_copy = copy.deepcopy(suite2p_args)
        # We have to pop the ref image out as numpy arrays can't be
        # serialized into json. The reference image is instead stored in
        # the 'ref_image' dataset.
        suite_args_copy.pop("refImg")
        args_copy.pop("refImg")
        args_copy["suite2p_args"] = suite_args_copy
        f.create_dataset(name="metadata", data=json.dumps(args_copy).encode("utf-8"))
        # save Suite2p registration metrics
        f.create_group("reg_metrics")
        f.create_dataset("reg_metrics/regDX", data=ops.get("regDX", []))
        f.create_dataset("reg_metrics/regPC", data=ops.get("regPC", []))
        f.create_dataset("reg_metrics/tPC", data=ops.get("tPC", []))
    logger.info(f"saved Suite2P output to {args['motion_corrected_output']}")
    # make projections
    mx_proj = projection_process(data, projection="max")
    av_proj = projection_process(data, projection="avg")
    if parser.data_type == "TIFF":
        input_file = input_file[0]
    write_data_process(
        args_copy,
        input_file,
        args["motion_corrected_output"],
        output_dir,
        basename.split(".")[0],
        start_time,
        end_time=dt.now(),
    )
    # TODO: normalize here, if desired
    # save projections
    for im, dst_path in zip(
        [mx_proj, av_proj],
        [
            args["max_projection_output"],
            args["avg_projection_output"],
        ],
    ):
        with Image.fromarray(im) as pilim:
            pilim.save(dst_path)
        logger.info(f"wrote {dst_path}")

    # Save motion offset data to a csv file
    # TODO: This *.csv file is being created to maintain compatibility
    # with current ophys processing pipeline. In the future this output
    # should be removed and a better data storage format used.
    # 01/25/2021 - NJM
    if suite2p_args["nonrigid"]:
        # Convert data to string for storage in the CSV output.
        nonrigid_x = [
            np.array2string(
                arr,
                separator=",",
                suppress_small=True,
                max_line_width=4096,
            )
            for arr in ops["xoff1"]
        ]
        nonrigid_y = [
            np.array2string(
                arr,
                separator=",",
                suppress_small=True,
                max_line_width=4096,
            )
            for arr in ops["yoff1"]
        ]
        nonrigid_corr = [
            np.array2string(
                arr,
                separator=",",
                suppress_small=True,
                max_line_width=4096,
            )
            for arr in ops["corrXY1"]
        ]
        motion_offset_df = pd.DataFrame(
            {
                "framenumber": list(range(ops["nframes"])),
                "x": ops["xoff"],
                "y": ops["yoff"],
                "x_pre_clip": ops["xoff"],
                "y_pre_clip": ops["yoff"],
                "correlation": ops["corrXY"],
                "is_valid": is_valid,
                "nonrigid_x": nonrigid_x,
                "nonrigid_y": nonrigid_y,
                "nonrigid_corr": nonrigid_corr,
            }
        )
    else:
        motion_offset_df = pd.DataFrame(
            {
                "framenumber": list(range(ops["nframes"])),
                "x": delta_x,
                "y": delta_y,
                "x_pre_clip": ops["xoff"],
                "y_pre_clip": ops["yoff"],
                "correlation": ops["corrXY"],
                "is_valid": is_valid,
            }
        )
    motion_offset_df.to_csv(path_or_buf=args["motion_diagnostics_output"], index=False)
    logger.info(
        f"Writing the LIMS expected 'OphysMotionXyOffsetData' "
        f"csv file to: {args['motion_diagnostics_output']}"
    )

    if len(clipped_indices) != 0 and not suite2p_args["nonrigid"]:
        logger.warning(
            "some offsets have been clipped and the values "
            "for 'correlation' in "
            "{args['motion_diagnostics_output']} "
            "where (x_clipped OR y_clipped) = True are not valid"
        )

    # create and write the summary png
    motion_offset_df = pd.read_csv(args["motion_diagnostics_output"])
    png_out_path = make_png(
        Path(args["max_projection_output"]),
        Path(args["avg_projection_output"]),
        motion_offset_df,
        Path(args["registration_summary_output"]),
    )
    logger.info(f"wrote {png_out_path}")

    # create and write the nonrigid summary png
    if "nonrigid_x" in motion_offset_df.keys():
        p = Path(args["registration_summary_output"])
        nonrigid_png_out_path = make_nonrigid_png(
            Path(args["motion_corrected_output"]),
            Path(args["avg_projection_output"]),
            motion_offset_df,
            p.parent.joinpath(p.stem + "_nonrigid" + p.suffix),
        )
        logger.info(f"wrote {nonrigid_png_out_path}")

    # downsample and normalize the input movies
    ds_partial = partial(
        downsample_normalize,
        frame_rate=args["movie_frame_rate_hz"],
        bin_size=args["preview_frame_bin_seconds"],
        lower_quantile=args["movie_lower_quantile"],
        upper_quantile=args["movie_upper_quantile"],
    )
    if suite2p_args.get("h5py", ""):
        h5_file = suite2p_args["h5py"]
        processed_vids = [
            ds_partial(i)
            for i in [
                Path(h5_file),
                Path(args["motion_corrected_output"]),
            ]
        ]
    else:
        tiff_array = tiff_to_numpy(suite2p_args["tiff_list"])
        processed_vids = [
            ds_partial(i)
            for i in [
                tiff_array,
                Path(args["motion_corrected_output"]),
            ]
        ]

    logger.info(
        "finished downsampling motion corrected and non-motion corrected movies"
    )

    # tile into 1 movie, raw on left, motion corrected on right
    try:
        tiled_vids = np.block(processed_vids)

        # make into a viewable artifact
        playback_fps = (
            args["preview_playback_factor"] / args["preview_frame_bin_seconds"]
        )
        encode_video(tiled_vids, args["motion_correction_preview_output"], playback_fps)
        logger.info("wrote " f"{args['motion_correction_preview_output']}")
    except:
        logger.info("Could not write motion correction preview")
    # compute crispness of mean image using raw and registered movie
    if suite2p_args.get("h5py", ""):
        with (
            h5py.File(h5_file) as f_raw,
            h5py.File(args["motion_corrected_output"], "r+") as f,
        ):
            mov_raw = f_raw["data"]
            mov = f["data"]
            regDX = f["reg_metrics/regDX"][:]
            crispness = compute_crispness(mov_raw, mov)
            logger.info(
                "computed crispness of mean image before and after registration"
            )

            # compute residual optical flow using Farneback method
            if f["reg_metrics/regPC"][:].any():
                flows, farnebackDX = compute_residual_optical_flow(
                    f["reg_metrics/regPC"]
                )
                f.create_dataset("reg_metrics/farnebackROF", data=flows)
                f.create_dataset("reg_metrics/farnebackDX", data=farnebackDX)
                logger.info(
                    "computed residual optical flow of top PCs using Farneback method"
                )
                # create image of PC_low, PC_high, and the residual optical flow between them
            f.create_dataset("reg_metrics/crispness", data=crispness)
            logger.info(
                "appended additional registration metrics to"
                f"{args['motion_corrected_output']}"
            )

    else:
        with h5py.File(args["motion_corrected_output"], "r+") as f:
            regDX = f["reg_metrics/regDX"][:]
            crispness = compute_crispness(tiff_array, f["data"])
            logger.info(
                "computed crispness of mean image before and after registration"
            )
            if f["reg_metrics/regPC"][:].any():
                regPC = f["reg_metrics/regPC"]

                flows, farnebackDX = compute_residual_optical_flow(regPC)

                f.create_dataset("reg_metrics/farnebackROF", data=flows)
                f.create_dataset("reg_metrics/farnebackDX", data=farnebackDX)
                logger.info(
                    "computed residual optical flow of top PCs using Farneback method"
                )

            f.create_dataset("reg_metrics/crispness", data=crispness)
            logger.info(
                "appended additional registration metrics to"
                f"{args['motion_corrected_output']}"
            )

    if regDX.any() and farnebackDX.any():
        for iPC in set(
            (
                np.argmax(regDX[:, -1]),
                np.argmax(farnebackDX[:, -1]),
            )
        ):
            p = Path(args["registration_summary_output"])
            flow_png(
                Path(args["motion_corrected_output"]),
                str(p.parent / p.stem),
                iPC,
            )
        logger.info(f"created images of PC_low, PC_high, and PC_rof for PC {iPC}")
    # Clean up temporary directory
    tmp_dir.cleanup()

    # Write QC metrics
    serialize_registration_summary_qcmetric()
    serialize_fov_quality_qcmetric()
