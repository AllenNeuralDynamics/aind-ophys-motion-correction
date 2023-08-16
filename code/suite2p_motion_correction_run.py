import argparse
import glob
import os

from registration_utils import (
    make_output_directory,
    get_plane,
    write_json,
    get_frame_rate_platform_json,
)


def suite2p_motion_correction_input_json(
    input_dir: str, output_dir: str, plane=None
) -> dict:
    """Generates suite2p motion correction input json for a given plane

    Args:
        input_dir (str): path to data directory
        output_dir (str): path to results directory
        plane (int, optional): Plane to process if session directory is passed. Defaults to None.

    Returns:
        dict: suite2p motion correction input json
    """
    if plane is None:
        h5_file = glob.glob(f"{input_dir}/*.h5")[0]
    else:
        h5_file = glob.glob(
            f"{input_dir}/Other_[0-9]*/Other/ophys/planes/{plane}/*um.h5"
        )[0]
    plane = str(plane)
    output_dir = make_output_directory(output_dir, plane)
    frame_rate_hz = get_frame_rate_platform_json(input_dir)
    return {
        "suite2p_args": {"h5py": h5_file, "tmp_dir": "/scratch"},
        "movie_frame_rate_hz": frame_rate_hz,
        "motion_corrected_output": os.path.join(
            output_dir, f"{plane}um_suite2p_motion_output.h5"
        ),
        "motion_diagnostics_output": os.path.join(
            output_dir, f"{plane}um_suite2p_rigid_motion_transform.csv"
        ),
        "max_projection_output": os.path.join(
            output_dir, f"{plane}um_suite2p_maximum_projection.png"
        ),
        "avg_projection_output": os.path.join(
            output_dir, f"{plane}um_suite2p_average_projection.png"
        ),
        "registration_summary_output": os.path.join(
            output_dir, f"{plane}um_suite2p_registration_summary.png"
        ),
        "motion_correction_preview_output": os.path.join(
            output_dir, f"{plane}um_suite2p_motion_preview.webm"
        ),
        "output_json": os.path.join(
            output_dir, f"{plane}um_suite2p_motion_correction_output.json"
        ),
    }


def main():
    """suite2p motion correction generates the bulk of the output data for the rest of the processing pipeline.
    Requires the split, 2photon  h5 movie to motion correct"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, help="Input directory")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory")
    parser.add_argument("-p", "--plane", type=int, help="Plane depth", default=None)
    parser.add_argument(
        "-a", "--action", type=str, help="Action to perform", default="create-input-json"
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not args.plane:
        plane = get_plane(input_dir)
    else:
        plane = args.plane
    if args.action == "create-input-json":  # this is the only action for now
        data = suite2p_motion_correction_input_json(input_dir, output_dir, plane=plane)
        write_json("/data/input.json", data)


if __name__ == "__main__":
    main()
