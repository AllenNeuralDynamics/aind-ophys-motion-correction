import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import argschema
import h5py
import numpy as np
import pandas as pd
import suite2p
from PIL import Image
from time import time

import registration_utils as utils
from registration_qc import RegistrationQC
from schemas import (Suite2PRegistrationInputSchema,
                     Suite2PRegistrationOutputSchema)

# w/o this line the registration progress (registered/total frames) isn't shown in real time
sys.stdout = open(1, "w", buffering=1)


class Suite2PRegistration(argschema.ArgSchemaParser):
    default_schema = Suite2PRegistrationInputSchema
    default_output_schema = Suite2PRegistrationOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args["log_level"])
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA", "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        # Set suite2p args.
        suite2p_args = suite2p.default_ops()
        for k in self.args:
            if k in suite2p_args or k == "refImg":
                suite2p_args[k] = self.args[k]
        suite2p_args["roidetect"] = False

        # if data is in a S3 bucket, copy it to /scratch for faster access
        if utils.is_S3(suite2p_args["h5py"]):
            dst = "/scratch/" + Path(suite2p_args["h5py"]).name
            self.logger.info(f"copying {suite2p_args['h5py']} from S3 bucket to {dst}")
            shutil.copy(suite2p_args["h5py"], dst)
            suite2p_args["h5py"] = dst

        utils.check_and_warn_on_datatype(
            h5py_name=suite2p_args["h5py"],
            h5py_key=suite2p_args["h5py_key"],
            logger=self.logger.warning,
        )

        if self.args["auto_remove_empty_frames"]:
            self.logger.info(
                "Attempting to find empty frames at the start " "and end of the movie."
            )
            lowside, highside = utils.find_movie_start_end_empty_frames(
                h5py_name=suite2p_args["h5py"],
                h5py_key=suite2p_args["h5py_key"],
                logger=self.logger.warning,
            )
            self.args["trim_frames_start"] = lowside
            self.args["trim_frames_end"] = highside
            self.logger.info(
                f"Found ({lowside}, {highside}) at the " "start/end of the movie."
            )

        if suite2p_args["force_refImg"] and len(suite2p_args["refImg"]) == 0:
            # Use our own version of compute_reference to create the initial
            # reference image used by suite2p.
            self.logger.info(
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
                self.logger.info(
                    "Attempting to optimize registration " "parameters Using:"
                )
                self.logger.info(
                    "\tsmooth_sigma range: "
                    f'{self.args["smooth_sigma_min"]} - '
                    f'{self.args["smooth_sigma_max"]}, '
                    f'steps: {self.args["smooth_sigma_steps"]}'
                )
                self.logger.info(
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
                    logger=self.logger.info,
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
                self.logger.info("Creating custom reference image...")
                suite2p_args["refImg"] = utils.compute_reference(
                    input_frames=intial_frames,
                    niter=self.args["max_reference_iterations"],
                    maxregshift=suite2p_args["maxregshift"],
                    smooth_sigma=suite2p_args["smooth_sigma"],
                    smooth_sigma_time=suite2p_args["smooth_sigma_time"],
                )
                tic +=time()
                self.logger.info(f"took {tic}s")

        # register with Suite2P
        self.logger.info("attempting to motion correct " f"{suite2p_args['h5py']}")
        # make a tempdir for Suite2P's output
        tmp_dir = tempfile.TemporaryDirectory(dir=self.args["tmp_dir"])
        tdir = tmp_dir.name
        suite2p_args["save_path0"] = tdir
        self.logger.info(f"Running Suite2P with output going to {tdir}")

        # Make a copy of the args to remove the NumpyArray, refImg, as
        # numpy.ndarray can't be serialized with json. Converting to list
        # and writing to the logger causes the output to be unreadable.
        copy_of_args = copy.deepcopy(suite2p_args)
        copy_of_args.pop("refImg")

        msg = f"running Suite2P v{suite2p.version} with args\n"
        msg += f"{json.dumps(copy_of_args, indent=2, sort_keys=True)}\n"
        self.logger.info(msg)

        # If we are using a external reference image (including our own
        # produced by compute_referece) communicate this in the log.
        if suite2p_args["force_refImg"]:
            self.logger.info(
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
        self.logger.info(
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
        self.logger.info(f"{len(x_clipped)} frames clipped in x")
        self.logger.info(f"{len(y_clipped)} frames clipped in y")
        self.logger.info(
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
            utils.write_output_metadata(
                json.dumps(args_copy),
                suite2p_args["h5py"],
                self.args["motion_corrected_output"])
            f.create_dataset(
                name="metadata", data=json.dumps(args_copy).encode("utf-8")
            )
            # save Suite2p registration metrics
            f.create_group("reg_metrics")
            f.create_dataset("reg_metrics/regDX", data=ops["regDX"])
            f.create_dataset("reg_metrics/regPC", data=ops["regPC"])
            f.create_dataset("reg_metrics/tPC", data=ops["tPC"])
        self.logger.info(
            "saved Suite2P output to " f"{self.args['motion_corrected_output']}"
        )

        # make projections
        mx_proj = utils.projection_process(data, projection="max")
        av_proj = utils.projection_process(data, projection="avg")
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
            self.logger.info(f"wrote {dst_path}")

        # Save motion offset data to a csv file
        # TODO: This *.csv file is being created to maintain compatability
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
        self.logger.info(
            f"Writing the LIMS expected 'OphysMotionXyOffsetData' "
            f"csv file to: {self.args['motion_diagnostics_output']}"
        )
        if len(clipped_indices) != 0 and not suite2p_args["nonrigid"]:
            self.logger.warning(
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


if __name__ == "__main__":  # pragma: nocover
    s2preg = Suite2PRegistration()
    s2preg.run()
