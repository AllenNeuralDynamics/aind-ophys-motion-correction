import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import logging
import argparse
import h5py
import numpy as np
import pandas as pd
import suite2p
from PIL import Image
from time import time

import registration_utils as utils
from registration_qc import RegistrationQC
import argparse
import json
import logging
import os
import subprocess
import warnings
from glob import glob
from itertools import product
from time import time
from typing import Callable, List, Tuple, Union
from pathlib import Path
import re
from datetime import datetime as dt
import pytz
import h5py
import numpy as np

from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess
from aind_ophys_utils.array_utils import normalize_array
from scipy.ndimage import median_filter
from scipy.stats import sigmaclip
from suite2p.registration.register import (pick_initial_reference,
                                           register_frames)
from suite2p.registration.rigid import (apply_masks, compute_masks, phasecorr,
                                        phasecorr_reference, shift_frame)
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

def load_initial_frames(
    file_path: str,
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
    with h5py.File(file_path, "r") as hdf5_file:
        # Load all frames as fancy indexing is slower than loading the full
        # data.
        max_frame = hdf5_file[h5py_key].shape[0] - trim_frames_end
        frame_window = hdf5_file[h5py_key][trim_frames_start:max_frame]
        # Total number of frames in the movie.
        tot_frames = frame_window.shape[0]
        requested_frames = np.linspace(
            0, tot_frames, 1 + min(n_frames, tot_frames), dtype=int
        )[:-1]
        frames = frame_window[requested_frames]
    return frames

def compute_reference(
    input_frames: np.ndarray,
    niter: int,
    maxregshift: float,
    smooth_sigma: float,
    smooth_sigma_time: float,
    mask_slope_factor: float = 3,
    logger: callable = None,
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
                    maskSlope=3 * smooth_sigma,
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
    logger: callable = None,
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
    logger : callable, optional
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

def check_and_warn_on_datatype(h5py_name: str, h5py_key: str, logger: Callable):
    """Suite2p assumes int16 types throughout code. Check that the input
    data is type int16 else throw a warning.

    Parameters
    ----------
    h5py_name : str
        Path to the HDF5 containing the data.
    h5py_key : str
        Name of the dataset to check.
    logger : Callable
        Logger to output logger warning to.
    """
    with h5py.File(h5py_name, "r") as h5_file:
        dataset = h5_file[h5py_key]

        if dataset.dtype.byteorder == ">":
            logger(
                "Data byteorder is big-endian which may cause issues in "
                "suite2p. This may result in a crash or unexpected "
                "results."
            )
        if dataset.dtype.name != "int16":
            logger(
                f"Data type is {dataset.dtype.name} and not int16. Suite2p "
                "assumes int16 data as input and throughout codebase. "
                "Non-int16 data may result in unexpected results or "
                "crashes."
            )

def find_movie_start_end_empty_frames(
    h5py_name: str, h5py_key: str, n_sigma: float = 5, logger: callable = None
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
    logger : callable, optional
        Function to print warning messages to.

    Returns
    -------
    trim_frames : Tuple[int, int]
        Tuple of the number of frames to cut from the start and end of the
        movie as (n_trim_start, n_trim_end).
    """
    # Load the data.
    with h5py.File(h5py_name, "r") as h5_file:
        frames = h5_file[h5py_key][:]
    # Find the midpoint of the movie.
    midpoint = frames.shape[0] // 2

    # We discover empty or extrema frames by comparing the mean of each frames
    # to the mean of the full movie.
    means = frames.mean(axis=(1, 2))
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
        ).flatten() + midpoint)

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
        np.arange(frames.shape[0] - highside, frames.shape[0], dtype=end_idxs.dtype),
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
) -> Tuple[np.ndarray, List]:
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

def now() -> str:
    """Generates string with current date and time in PST
    
    Returns
    -------
    str 
        YYYY-MM-DD_HH-MM-SS
    """
    current_dt = dt.now(tz=pytz.timezone("America/Los_Angeles"))
    return f"{current_dt.strftime('%Y-%m-%d')}_{current_dt.strftime('%H-%M-%S')}"

def make_output_directory(output_dir: str, h5_file: str, plane: str=None) -> str:
    """Creates the output directory if it does not exist
    
    Parameters
    ----------
    output_dir: str
        output directory
    h5_file: str 
        h5 file path
    plane: str
        plane number
    
    Returns
    -------
    output_dir: str
        output directory
    """
    exp_to_match = r"Other_\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    try:
        parent_dir = re.findall(exp_to_match, h5_file)[0] + "_processed_" + now()
    except IndexError:
        return output_dir
    if plane:
        output_dir = os.path.join(output_dir, parent_dir, plane)
    else:
        output_dir = os.path.join(output_dir, parent_dir)
    os.makedirs(output_dir, exist_ok=True)
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
    
def write_output_metadata(metadata: dict, raw_movie: Union[str, Path], motion_corrected_movie: Union[str, Path]) -> None:
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
    processing = Processing(
        data_processes=[
            DataProcess(
                name="Other",
                version="0.0.1",
                start_date_time=dt.now(),  # TODO: Add actual dt
                end_date_time=dt.now(),  # TODO: Add actual dt
                input_location=raw_movie,
                output_location=motion_corrected_movie,
                code_url="https:/3+/github.com/AllenNeuralDynamics/aind-ophys-motion-correction/tree/main/code",
                parameters=metadata,
            )
        ],
    )
    processing.write_standard_file(
        output_directory=Path(os.path.dirname(motion_corrected_movie))
        )

def set_default_outputs(data):
    stem = Path(data["h5py"]).stem
    for key, default in (
        ("motion_corrected_output", "_registered.h5"),
        ("motion_diagnostics_output", "_motion_transform.csv"),
        ("max_projection_output", "_maximum_projection.png"),
        ("avg_projection_output", "_average_projection.png"),
        ("registration_summary_output", "_registration_summary.png"),
        ("motion_correction_preview_output", "_motion_preview.webm"),
        ("output_json", "_motion_correction_output.json"),
    ):
        if data[key] is None:
            data[key] = "../results/" + stem + default
    return data

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

if __name__ == "__main__":  # pragma: nocover
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Suite2P Registration")

    # s2p IO settings (file paths)
    parser.add_argument("--h5py", type=str, required=True,
        help="Path to input video.",
    )

    parser.add_argument("--h5py_key", type=str, default="data",
        help="Key in h5py where data array is stored (default: data)",
    )

    parser.add_argument("--tmp_dir", type=str, default="/scratch",
        help="Directory into which to write temporary files produced by Suite2P (default: /scratch)",
    )

    parser.add_argument("--smooth_sigma", type=float, default=1.15,
        help="Standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV). (default: 1.15)",
    )

    parser.add_argument("--smooth_sigma_time", type=float, default=0.0,
        help="Standard deviation in time frames of the gaussian used to smooth the data before phase correlation is computed. Might need this to be set to 1 or 2 for low SNR data. (default: 0.0)",
    )

    parser.add_argument("--nonrigid", action="store_true", default=True,
        help="Turns on Suite2P's non-rigid registration algorithm",
    )

    parser.add_argument("--block_size", type=int, nargs=2, default=[128, 128],
        help="Block dimensions in y, x in pixels. Must be a multiple of 2. block_size=[128, 128] will yield 16 blocks for a 512x512 movie (default: [128, 128])",
    )

    parser.add_argument("--snr_thresh", type=float, default=1.2,
        help="If a block is below the above snr threshold. Apply smoothing to the block. SNR is calculated on the value of the phase correlation of the blocks. (default: 1.2)",
    )

    parser.add_argument(
        "--force_refImg",
        action="store_true",
        default=True,
        help="Force the use of an external reference image (default: True)",
    )

    parser.add_argument("--avg_projection_output", type=str, default=None,
        help="Desired path for *.png of the avg projection of the motion corrected video.",
    )

    parser.add_argument("--output_json", type=str, default=None,
        help="Destination path for output json",
    )

    parser.add_argument("--registration_summary_output", type=str,
        default=None,
        help="Desired path for *.png for summary QC plot",
    )

    parser.add_argument("--motion_correction_preview_output", type=str,
        default=None, help="Desired path for *.webm motion preview"
    )

    parser.add_argument("--movie_lower_quantile", type=float, default=0.1,
        help="Lower quantile threshold for avg projection histogram adjustment of movie (default: 0.1)",
    )

    parser.add_argument("--movie_upper_quantile", type=float, default=0.999,
        help="Upper quantile threshold for avg projection histogram adjustment of movie (default: 0.999)",
    )

    parser.add_argument("--preview_frame_bin_seconds", type=float,
        default=2.0,
        help=(
            "before creating the webm, the movies will be "
            "averaged into bins of this many seconds."
        ),
    )

    parser.add_argument("--preview_playback_factor", type=float, default=10.0,
        help=(
            "the preview movie will playback at this factor " "times real-time."
        ),
    )

    parser.add_argument("--outlier_detrend_window", type=float, default=3.0,
        help=(
            "for outlier rejection in the xoff/yoff outputs "
            "of suite2p, the offsets are first de-trended "
            "with a median filter of this duration [seconds]. "
            "This value is ~30 or 90 samples in size for 11 and 31"
            "Hz sampling rates respectively."
        )
    )

    parser.add_argument("--outlier_maxregshift", type=float, default=0.05,
        help=(
            "units [fraction FOV dim]. After median-filter "
            "detrending, outliers more than this value are "
            "clipped to this value in x and y offset, independently."
            "This is similar to Suite2P's internal maxregshift, but"
            "allows for low-frequency drift. Default value of 0.05 "
            "is typically clipping outliers to 512 * 0.05 = 25 "
            "pixels above or below the median trend."
        )
    )

    parser.add_argument("--clip_negative", action="store_true", default=False,
        help=(
            "Whether or not to clip negative pixel "
            "values in output. Because the pixel values "
            "in the raw  movies are set by the current "
            "coming off a photomultiplier tube, there can "
            "be pixels with negative values (current has a "
            "sign), possibly due to noise in the rig. "
            "Some segmentation algorithms cannot handle "
            "negative values in the movie, so we have this "
            "option to artificially set those pixels to zero."
        )
    )

    parser.add_argument("--max_reference_iterations", type=int, default=8,
        help="Maximum number of iterations for creating a reference image (default: 8)",
    )

    parser.add_argument("--auto_remove_empty_frames", action="store_true",
        default=True,
        help=("Automatically detect empty noise frames at the start and "
            "end of the movie. Overrides values set in "
            "trim_frames_start and trim_frames_end. Some movies "
            "arrive with otherwise quality data but contain a set of "
            "frames that are empty and contain pure noise. When "
            "processed, these frames tend to receive "
            "large random shifts that throw off motion border "
            "calculation. Turning on this setting automatically "
            "detects these frames before processing and removes them "
            "from reference image creation,  automated smoothing "
            "parameter searches, and finally the motion border "
            "calculation. The frames are still written however any "
            "shift estimated is removed and their shift is set to 0 "
            "to avoid large motion borders."
        )
    )

    parser.add_argument("--trim_frames_start", type=int, default=0,
        help="Number of frames to remove from the start of the movie "
        "if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still "
        "written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and "
        "trim_frames_start > 0"
    )

    parser.add_argument(
        "--trim_frames_end",
        type=int,
        default=0,
        help="Number of frames to remove from the end of the movie "
        "if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still "
        "written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and "
        "trim_frames_start > 0"
    )

    parser.add_argument("--do_optimize_motion_params", action="store_true",
        default=False,
        help="Do a search for best parameters of smooth_sigma and "
        "smooth_sigma_time. Adds significant runtime cost to "
        "motion correction and should only be run once per "
        "experiment with the resulting parameters being stored "
        "for later use."
    )

    parser.add_argument(
        "--use_ave_image_as_reference",
        action="store_true",
        default=False,
        help="Only available if `do_optimize_motion_params` is set. "
        "After the a best set of smoothing parameters is found, "
        "use the resulting average image as the reference for the "
        "full registration. This can be used as two step "
        "registration by setting by setting "
        "smooth_sigma_min=smooth_sigma_max and "
        "smooth_sigma_time_min=smooth_sigma_time_max and "
        "steps=1."
    )

    parser.add_argument(
        "--n_batches",
        type=int,
        default=20,
        help="Number of batches of size suite2p_args['batch_size'] to "
        "load from the movie for smoothing parameter testing. "
        "Batches are evenly spaced throughout the movie."
    )

    # smooth_sigma_min
    parser.add_argument(
        "--smooth_sigma_min",
        type=float,
        default=0.65,
        help="Minimum value of the parameter search for smooth_sigma (default: 0.65)",
    )

    # smooth_sigma_max
    parser.add_argument(
        "--smooth_sigma_max",
        type=float,
        default=2.15,
        help="Maximum value of the parameter search for smooth_sigma (default: 2.15)",
    )

    # smooth_sigma_steps
    parser.add_argument(
        "--smooth_sigma_steps",
        type=int,
        default=4,
        help="Number of steps to grid between smooth_sigma and smooth_sigma_max (default: 4)",
    )

    # smooth_sigma_time_min
    parser.add_argument(
        "--smooth_sigma_time_min",
        type=float,
        default=0,
        help="Minimum value of the parameter search for smooth_sigma_time (default: 0)",
    )

    # smooth_sigma_time_max
    parser.add_argument(
        "--smooth_sigma_time_max",
        type=float,
        default=6,
        help="Maximum value of the parameter search for smooth_sigma_time (default: 6)",
    )

    # smooth_sigma_time_steps
    parser.add_argument(
        "--smooth_sigma_time_steps",
        type=int,
        default=7,
        help="Number of steps to grid between smooth_sigma and "
        "smooth_sigma_time_max. Large values will add significant "
        "time motion correction."
    )

    # Allen-specific options
    parser.add_argument("--movie_frame_rate_hz", type=float, required=True,
        help="Frame rate of movie, usually 31Hz or 11Hz",
    )
    parser.add_argument( "--motion_corrected_output", type=str, default=None,
        help="Destination path for hdf5 motion corrected video.",
    )
    parser.add_argument("--motion_diagnostics_output", type=str, default=None,
        help="Desired save path for *.csv file containing motion correction offset data",
    )
    parser.add_argument("--max_projection_output", type=str, default=None,
        help="Desired path for *.png of the max projection of the motion corrected video.",
    )

   # Generate input json
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-filename", type=str, help="Path to raw movie", default="/data/Other_667826_2023-04-10_16-08-00/Other/ophys/planes/70/70um.h5"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="/results/"
    )
    # Parse command-line arguments
    args = parser.parse_args()
    # General settings

    h5_file = args.input_filename

    # if not plane:
    try:
        plane = os.path.dirname(h5_file).split("/")[-1]
        assert plane == int
    except AssertionError:
        plane = None
    output_dir = make_output_directory(args.output_dir, h5_file, plane)

    try:
        frame_rate_hz = get_frame_rate_platform_json(h5_file)
    except Exception:
        frame_rate_hz = 30.

    data = {"h5py": h5_file, "movie_frame_rate_hz": frame_rate_hz}
    for key, default in (
        ("motion_corrected_output", "_registered.h5"),
        ("motion_diagnostics_output", "_motion_transform.csv"),
        ("max_projection_output", "_maximum_projection.png"),
        ("avg_projection_output", "_average_projection.png"),
        ("registration_summary_output", "_registration_summary.png"),
        ("motion_correction_preview_output", "_motion_preview.webm"),
        ("output_json", "_motion_correction_output.json"),
    ):
        data[key] = os.path.join(
            output_dir, os.path.splitext(os.path.basename(h5_file))[0] + default
        )

    # Set the log level and name the logger
    logger = logging.getLogger('suite2p_motion_correction')
    logger.setLevel(logging.INFO)

    # Set suite2p args.
    suite2p_args = suite2p.default_ops()
    for k in self.args:
        if k in suite2p_args or k == "refImg":
            suite2p_args[k] = self.args[k]

    # Here we overwrite the parameters for suite2p that will not change in our 
    # processing pipeline. These are parameters that are not exposed to 
    # minimize code length. 
    suite2p_args["roidetect"] = False
    suite2p_args["do_registration"] = 1
    suite2p_args["data_path"]=[] # TODO: remove this if not needed by suite2p
    suite2p_args["reg_tif"]= False # We save our own outputs here
    suite2p_args["nimg_init"]= 5000 # Nb of images to compute reference. This value is a bit high. Suite2p has it at 300 normally
    suite2p_args["maxregshift"]= 0.2 # Max allowed registration shift as a fraction of frame max(width and height)

    # These parameters are at the same value as suite2p default. This is just here
    # to make it clear we need those parameters to be at the same value as
    # suite2p default but those lines could be deleted.
    suite2p_args["maxregshiftNR"]= 0.5 # Maximum shift allowed in pixels for a block in rigid registration. 
    suite2p_args["batch_size"]= 500 # Number of frames to process at once


    # if data is in a S3 bucket, copy it to /scratch for faster access
    if utils.is_S3(suite2p_args["h5py"]):
        dst = "/scratch/" + Path(suite2p_args["h5py"]).name
        logger.info(f"copying {suite2p_args['h5py']} from S3 bucket to {dst}")
        shutil.copy(suite2p_args["h5py"], dst)
        suite2p_args["h5py"] = dst

    utils.check_and_warn_on_datatype(
        h5py_name=suite2p_args["h5py"],
        h5py_key=suite2p_args["h5py_key"],
        logger=logger.warning,
    )

    if self.args["auto_remove_empty_frames"]:
        logger.info(
            "Attempting to find empty frames at the start " "and end of the movie."
        )
        lowside, highside = utils.find_movie_start_end_empty_frames(
            h5py_name=suite2p_args["h5py"],
            h5py_key=suite2p_args["h5py_key"],
            logger=logger.warning,
        )
        self.args["trim_frames_start"] = lowside
        self.args["trim_frames_end"] = highside
        logger.info(
            f"Found ({lowside}, {highside}) at the " "start/end of the movie."
        )

    if suite2p_args["force_refImg"] and len(suite2p_args["refImg"]) == 0:
        # Use our own version of compute_reference to create the initial
        # reference image used by suite2p.
        logger.info(
            f'Loading {suite2p_args["nimg_init"]} frames '
            "for reference image creation."
        )
        intial_frames = utils.load_initial_frames(
            file_path=suite2p_args["h5py"],
            h5py_key=suite2p_args["h5py_key"],
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=self.args["trim_frames_start"],
            trim_frames_end=self.args["trim_frames_end"],
        )

        if self.args["do_optimize_motion_params"]:
            logger.info(
                "Attempting to optimize registration " "parameters Using:"
            )
            logger.info(
                "\tsmooth_sigma range: "
                f'{self.args["smooth_sigma_min"]} - '
                f'{self.args["smooth_sigma_max"]}, '
                f'steps: {self.args["smooth_sigma_steps"]}'
            )
            logger.info(
                "\tsmooth_sigma_time range: "
                f'{self.args["smooth_sigma_time_min"]} - '
                f'{self.args["smooth_sigma_time_max"]}, '
                f'steps: {self.args["smooth_sigma_time_steps"]}'
            )

            # Create linear spaced arrays for the range of smooth
            # parameters to try.
            smooth_sigmas = np.linspace(
                self.args["smooth_sigma_min"],
                self.args["smooth_sigma_max"],
                self.args["smooth_sigma_steps"],
            )
            smooth_sigma_times = np.linspace(
                self.args["smooth_sigma_time_min"],
                self.args["smooth_sigma_time_max"],
                self.args["smooth_sigma_time_steps"],
            )

            optimize_result = utils.optimize_motion_parameters(
                initial_frames=intial_frames,
                smooth_sigmas=smooth_sigmas,
                smooth_sigma_times=smooth_sigma_times,
                suite2p_args=suite2p_args,
                trim_frames_start=self.args["trim_frames_start"],
                trim_frames_end=self.args["trim_frames_end"],
                n_batches=self.args["n_batches"],
                logger=logger.info,
            )
            if self.args["use_ave_image_as_reference"]:
                suite2p_args["refImg"] = optimize_result["ave_image"]
            else:
                suite2p_args["refImg"] = optimize_result["ref_image"]
            suite2p_args["smooth_sigma"] = optimize_result["smooth_sigma"]
            suite2p_args["smooth_sigma_time"] = optimize_result["smooth_sigma_time"]
        else:
            # Create the initial reference image and store it in the
            # suite2p_args dictionary. 8 iterations is the current default
            # in suite2p.
            tic =-time()
            logger.info("Creating custom reference image...")
            suite2p_args["refImg"] = utils.compute_reference(
                input_frames=intial_frames,
                niter=self.args["max_reference_iterations"],
                maxregshift=suite2p_args["maxregshift"],
                smooth_sigma=suite2p_args["smooth_sigma"],
                smooth_sigma_time=suite2p_args["smooth_sigma_time"],
            )
            tic +=time()
            logger.info(f"took {tic}s")

    # register with Suite2P
    logger.info("attempting to motion correct " f"{suite2p_args['h5py']}")
    # make a tempdir for Suite2P's output
    tmp_dir = tempfile.TemporaryDirectory(dir=self.args["tmp_dir"])
    tdir = tmp_dir.name
    suite2p_args["save_path0"] = tdir
    logger.info(f"Running Suite2P with output going to {tdir}")

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
        logger.info(
            "\tUsing custom reference image: " f'{suite2p_args["refImg"]}'
        )

    suite2p_args["h5py"] = [suite2p_args["h5py"]]
    suite2p.run_s2p(suite2p_args)
    suite2p_args["h5py"] = suite2p_args["h5py"][0]

    bin_path = list(Path(tdir).rglob("data.bin"))[0]
    ops_path = list(Path(tdir).rglob("ops.npy"))[0]
    # Suite2P ops file contains at least the following keys:
    # ["Lx", "Ly", "nframes", "xrange", "yrange", "xoff", "yoff",
    #  "corrXY", "meanImg"]
    ops = np.load(ops_path, allow_pickle=True).item()

    # identify and clip offset outliers
    detrend_size = int(
        self.args["movie_frame_rate_hz"] * self.args["outlier_detrend_window"]
    )
    xlimit = int(ops["Lx"] * self.args["outlier_maxregshift"])
    ylimit = int(ops["Ly"] * self.args["outlier_maxregshift"])
    logger.info(
        "checking whether to clip where median-filtered "
        "offsets exceed (x,y) limits of "
        f"({xlimit},{ylimit}) [pixels]"
    )
    delta_x, x_clipped = utils.identify_and_clip_outliers(
        np.array(ops["xoff"]), detrend_size, xlimit
    )
    delta_y, y_clipped = utils.identify_and_clip_outliers(
        np.array(ops["yoff"]), detrend_size, ylimit
    )
    clipped_indices = list(set(x_clipped).union(set(y_clipped)))
    logger.info(f"{len(x_clipped)} frames clipped in x")
    logger.info(f"{len(y_clipped)} frames clipped in y")
    logger.info(
        f"{len(clipped_indices)} frames will be adjusted " "for clipping"
    )

    # accumulate data from Suite2P's binary file
    data = suite2p.io.BinaryFile(ops["Ly"], ops["Lx"], bin_path).data

    if self.args["clip_negative"]:
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
    utils.reset_frame_shift(
        data,
        delta_y,
        delta_x,
        self.args["trim_frames_start"],
        self.args["trim_frames_end"],
    )
    # Create a boolean lookup of frames we reset as they were found
    # to be empty.
    is_valid = np.ones(len(data), dtype="bool")
    is_valid[: self.args["trim_frames_start"]] = False
    is_valid[len(data) - self.args["trim_frames_end"] :] = False

    # write the hdf5
    with h5py.File(self.args["motion_corrected_output"], "w") as f:
        f.create_dataset("data", data=data, chunks=(1, *data.shape[1:]))
        # Sort the reference image used to register. If we do not used
        # our custom reference image creation code, this dataset will
        # be empty.
        f.create_dataset("ref_image", data=suite2p_args["refImg"])
        # Write a copy of the configuration output of this dataset into the
        # HDF5 file.
        args_copy = copy.deepcopy(self.args)
        suite_args_copy = copy.deepcopy(suite2p_args)
        # We have to pop the ref image out as numpy arrays can't be
        # serialized into json. The reference image is instead stored in
        # the 'ref_image' dataset.
        suite_args_copy.pop("refImg")
        args_copy.pop("refImg")
        args_copy["suite2p_args"] = suite_args_copy
        f.create_dataset(
            name="metadata", data=json.dumps(args_copy).encode("utf-8")
        )
        # save Suite2p registration metrics
        f.create_group("reg_metrics")
        f.create_dataset("reg_metrics/regDX", data=ops["regDX"])
        f.create_dataset("reg_metrics/regPC", data=ops["regPC"])
        f.create_dataset("reg_metrics/tPC", data=ops["tPC"])
    logger.info(
        "saved Suite2P output to " f"{self.args['motion_corrected_output']}"
    )
    # make projections
    mx_proj = utils.projection_process(data, projection="max")
    av_proj = utils.projection_process(data, projection="avg")
    utils.write_output_metadata(
        args_copy,
        suite2p_args["h5py"],
        self.args["motion_corrected_output"])
    # TODO: normalize here, if desired
    # save projections
    for im, dst_path in zip(
        [mx_proj, av_proj],
        [
            self.args["max_projection_output"],
            self.args["avg_projection_output"],
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
    motion_offset_df.to_csv(
        path_or_buf=self.args["motion_diagnostics_output"], index=False
    )
    logger.info(
        f"Writing the LIMS expected 'OphysMotionXyOffsetData' "
        f"csv file to: {self.args['motion_diagnostics_output']}"
    )
    if len(clipped_indices) != 0 and not suite2p_args["nonrigid"]:
        logger.warning(
            "some offsets have been clipped and the values "
            "for 'correlation' in "
            "{self.args['motion_diagnostics_output']} "
            "where (x_clipped OR y_clipped) = True are not valid"
        )

    qc_args = {
        k: self.args[k]
        for k in [
            "movie_frame_rate_hz",
            "max_projection_output",
            "avg_projection_output",
            "motion_diagnostics_output",
            "motion_corrected_output",
            "motion_correction_preview_output",
            "registration_summary_output",
            "log_level",
        ]
    }

    qc_args.update({"uncorrected_path": suite2p_args["h5py"]})
    rqc = RegistrationQC(input_data=qc_args, args=[])
    rqc.run()

    # Clean up temporary directory
    tmp_dir.cleanup()

    outj = {
        k: self.args[k]
        for k in [
            "motion_corrected_output",
            "motion_diagnostics_output",
            "max_projection_output",
            "avg_projection_output",
            "registration_summary_output",
            "motion_correction_preview_output",
        ]
    }
    self.output(outj, indent=2)
