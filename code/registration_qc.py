import json
from functools import partial
from pathlib import Path

import argschema
import cv2
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
from aind_ophys_utils.array_utils import normalize_array
from aind_ophys_utils.video_utils import downsample_h5_video, encode_video
from PIL import Image
from suite2p.registration.nonrigid import make_blocks

from schemas import H5InputFile

mpl.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class RegistrationQCException(Exception):
    pass


class RegistrationQCInputSchema(argschema.ArgSchema):
    movie_frame_rate_hz = argschema.fields.Float(
        required=True, description="frame rate of movie, usually 31Hz or 11Hz"
    )
    preview_frame_bin_seconds = argschema.fields.Float(
        required=False,
        default=2.0,
        description=(
            "before creating the webm, the movies will be "
            "averaged into bins of this many seconds."
        ),
    )
    preview_playback_factor = argschema.fields.Float(
        required=False,
        default=10.0,
        description=(
            "the preview movie will playback at this factor " "times real-time."
        ),
    )
    uncorrected_path = H5InputFile(
        required=True, description=("path to uncorrected original movie.")
    )
    motion_corrected_output = H5InputFile(
        required=True, description=("path to motion corrected movie.")
    )
    motion_diagnostics_output = argschema.fields.InputFile(
        required=True,
        description=(
            "Saved path for *.csv file containing motion " "correction offset data"
        ),
    )
    max_projection_output = argschema.fields.InputFile(
        required=True,
        description=(
            "Saved path for *.png of the max projection of the "
            "motion corrected video."
        ),
    )
    avg_projection_output = argschema.fields.InputFile(
        required=True,
        description=(
            "Saved path for *.png of the avg projection of the "
            "motion corrected video."
        ),
    )
    registration_summary_output = argschema.fields.OutputFile(
        required=True, description=("Desired path for *.png summary plot.")
    )
    motion_correction_preview_output = argschema.fields.OutputFile(
        required=True, description="Desired path for *.webm motion preview"
    )
    movie_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.1,
        description=(
            "lower quantile threshold for avg projection "
            "histogram adjustment of movie"
        ),
    )
    movie_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.999,
        description=(
            "upper quantile threshold for avg projection "
            "histogram adjustment of movie"
        ),
    )


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
    ds = downsample_h5_video(
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


class RegistrationQC(argschema.ArgSchemaParser):
    default_schema = RegistrationQCInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args["log_level"])

        # create and write the summary png
        motion_offset_df = pd.read_csv(self.args["motion_diagnostics_output"])
        png_out_path = make_png(
            Path(self.args["max_projection_output"]),
            Path(self.args["avg_projection_output"]),
            motion_offset_df,
            Path(self.args["registration_summary_output"]),
        )
        self.logger.info(f"wrote {png_out_path}")

        # create and write the nonrigid summary png
        if "nonrigid_x" in motion_offset_df.keys():
            p = Path(self.args["registration_summary_output"])
            nonrigid_png_out_path = make_nonrigid_png(
                Path(self.args["motion_corrected_output"]),
                Path(self.args["avg_projection_output"]),
                motion_offset_df,
                p.parent.joinpath(p.stem + "_nonrigid" + p.suffix),
            )
            self.logger.info(f"wrote {nonrigid_png_out_path}")

        # downsample and normalize the input movies
        ds_partial = partial(
            downsample_normalize,
            frame_rate=self.args["movie_frame_rate_hz"],
            bin_size=self.args["preview_frame_bin_seconds"],
            lower_quantile=self.args["movie_lower_quantile"],
            upper_quantile=self.args["movie_upper_quantile"],
        )
        processed_vids = [
            ds_partial(i)
            for i in [
                Path(self.args["uncorrected_path"]),
                Path(self.args["motion_corrected_output"]),
            ]
        ]
        self.logger.info(
            "finished downsampling motion corrected " "and non-motion corrected movies"
        )

        # tile into 1 movie, raw on left, motion corrected on right
        try:
            tiled_vids = np.block(processed_vids)

            # make into a viewable artifact
            playback_fps = self.args["preview_playback_factor"] \
                / self.args["preview_frame_bin_seconds"]
            encode_video(
                tiled_vids, self.args["motion_correction_preview_output"], playback_fps
            )
            self.logger.info("wrote " f"{self.args['motion_correction_preview_output']}")
        except ValueError:
            self.logger.info(f"Could not create motion correction preview output")
        # compute crispness of mean image using raw and registered movie
        with (
            h5py.File(self.args["uncorrected_path"]) as f_raw,
            h5py.File(self.args["motion_corrected_output"], "r+") as f,
        ):
            mov_raw = f_raw["data"]
            mov = f["data"]
            crispness = [
                np.sqrt(np.sum(np.array(np.gradient(np.mean(m, 0))) ** 2))
                for m in (mov_raw, mov)
            ]
            self.logger.info(
                "computed crispness of mean image before and after registration"
            )

            # compute residual optical flow using Farneback method
            regPC = f["reg_metrics/regPC"]
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
            farnebackDX = np.transpose(
                [flows_norm.mean((1, 2)), flows_norm.max((1, 2))]
            )
            f.create_dataset("reg_metrics/crispness", data=crispness)
            f.create_dataset("reg_metrics/farnebackROF", data=flows)
            f.create_dataset("reg_metrics/farnebackDX", data=farnebackDX)
            self.logger.info(
                "computed residual optical flow of top PCs using Farneback method"
            )
            self.logger.info(
                "appended additional registration metrics to"
                f"{self.args['motion_corrected_output']}"
            )

            # create image of PC_low, PC_high, and the residual optical flow between them
            for iPC in set(
                (
                    np.argmax(f["reg_metrics/regDX"][:, -1]),
                    np.argmax(farnebackDX[:, -1]),
                )
            ):
                p = Path(self.args["registration_summary_output"])
                flow_png(
                    Path(self.args["motion_corrected_output"]),
                    str(p.parent / p.stem),
                    iPC,
                )
                self.logger.info(
                    f"created images of PC_low, PC_high, and PC_rof for PC {iPC}"
                )
        return


if __name__ == "__main__":
    rqc = RegistrationQC()
    rqc.run()
