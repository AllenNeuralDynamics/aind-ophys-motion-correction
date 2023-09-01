from pathlib import Path
import argschema
import marshmallow as mm


class Suite2PRegistrationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")

    """
    s2p parameter names are copied from:
    https://github.com/MouseLand/suite2p/blob/master/suite2p/run_s2p.py
    descriptions here should indicate why certain default setting
    choices have been made for Allen production.
    """
    # s2p IO settings (file paths)
    h5py = argschema.fields.InputFile(
        required=True,
        description=(
            "Path to input video. In Allen production case, "
            "assumed to be motion-corrected."
        ),
    )
    h5py_key = argschema.fields.Str(
        required=False,
        missing="data",
        default="data",
        description="key in h5py where data array is stored",
    )
    data_path = argschema.fields.List(
        argschema.fields.Str,
        cli_as_single_argument=True,
        required=False,
        default=[],
        description=(
            "Allen production specifies h5py as the source of "
            "the data, but Suite2P still wants this key in the "
            "args."
        ),
    )
    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default="/scratch",
        description=(
            "Directory into which to write temporary files " "produced by Suite2P"
        ),
    )

    # s2p registration settings
    do_registration = argschema.fields.Int(
        default=1, description=("0 skips registration.")
    )
    reg_tif = argschema.fields.Bool(
        default=False, description="whether to save registered tiffs"
    )
    maxregshift = argschema.fields.Float(
        default=0.2,
        description=(
            "max allowed registration shift, as a fraction of "
            "frame max(width and height)"
        ),
    )
    nimg_init = argschema.fields.Int(
        default=5000,
        description=(
            "How many frames to use to compute reference " "image for registration"
        ),
    )
    batch_size = argschema.fields.Int(
        default=500, description=("Number of frames to process at once.")
    )
    smooth_sigma = argschema.fields.Float(
        default=1.15,
        description=(
            "Standard deviation in pixels of the gaussian used "
            "to smooth the phase correlation between the reference "
            "image and the frame which is being registered. A "
            "value of >4 is recommended for one-photon "
            "recordings (with a 512x512 pixel FOV)."
        ),
    )
    smooth_sigma_time = argschema.fields.Float(
        default=0.0,
        description=(
            "Standard deviation in time frames of the gaussian "
            "used to smooth the data before phase correlation is "
            "computed. Might need this to be set to 1 or 2 for "
            "low SNR data."
        ),
    )
    nonrigid = argschema.fields.Boolean(
        default=True,
        required=False,
        description=("Turns on Suite2P's non-rigid registration algorithm"),
    )
    block_size = argschema.fields.List(
        argschema.fields.Int,
        cli_as_single_argument=True,
        default=[128, 128],
        required=False,
        description=(
            "Block dimensions in y, x in pixels. Must be a multiple "
            "of 2. block_size=[128, 128] will yield 16 blocks for a "
            "512x512 movie."
        ),
    )
    snr_thresh = argschema.fields.Float(
        default=1.2,
        required=False,
        description=(
            "If a block is below the above snr threshold. Apply "
            "smoothing to the block. SNR is calculated on the "
            "value of the phase correlation of the blocks."
        ),
    )
    maxregshiftNR = argschema.fields.Int(
        default=5,
        required=False,
        description=(
            "Maximum shift allowed in pixels for a block in "
            "rigid registration. This value is relative to the "
            "rigid shift."
        ),
    )

    refImg = argschema.fields.NumpyArray(
        default=[],
        required=False,
        description="Reference image to use instead of suite2p's internal "
        "calculation. By default we compute our own reference "
        "image to feed into suite2p. This is done by leaving the "
        "default value here and setting force_refImg to True.",
    )
    force_refImg = argschema.fields.Bool(
        default=True,
        required=False,
        description="Force suite2p to use a external reference image instead "
        "of computing one internally. To use automated reference "
        "image generation in ophys_etl, set this value to True and"
        "refImg to an empty list or array (Default).",
    )

    # Allen-specific options
    movie_frame_rate_hz = argschema.fields.Float(
        required=True, description="frame rate of movie, usually 31Hz or 11Hz"
    )
    motion_corrected_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="destination path for hdf5 motion corrected video.",
    )
    motion_diagnostics_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Desired save path for *.csv file containing motion "
            "correction offset data"
        ),
    )
    max_projection_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Desired path for *.png of the max projection of the "
            "motion corrected video."
        ),
    )
    avg_projection_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Desired path for *.png of the avg projection of the "
            "motion corrected video."
        ),
    )
    output_json = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="destination path for output json",
    )
    registration_summary_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Desired path for *.png for summary QC plot",
    )
    motion_correction_preview_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Desired path for *.webm motion preview",
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
    preview_frame_bin_seconds = argschema.fields.Float(
        required=False,
        default=2.0,
        description=(
            "before creating the webm, the movies will be "
            "aveaged into bins of this many seconds."
        ),
    )
    preview_playback_factor = argschema.fields.Float(
        required=False,
        default=10.0,
        description=(
            "the preview movie will playback at this factor " "times real-time."
        ),
    )
    outlier_detrend_window = argschema.fields.Float(
        required=False,
        default=3.0,
        description=(
            "for outlier rejection in the xoff/yoff outputs "
            "of suite2p, the offsets are first de-trended "
            "with a median filter of this duration [seconds]. "
            "This value is ~30 or 90 samples in size for 11 and 31"
            "Hz sampling rates respectively."
        ),
    )
    outlier_maxregshift = argschema.fields.Float(
        required=False,
        default=0.05,
        description=(
            "units [fraction FOV dim]. After median-filter "
            "detrending, outliers more than this value are "
            "clipped to this value in x and y offset, independently."
            "This is similar to Suite2P's internal maxregshift, but"
            "allows for low-frequency drift. Default value of 0.05 "
            "is typically clipping outliers to 512 * 0.05 = 25 "
            "pixels above or below the median trend."
        ),
    )
    clip_negative = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "Whether or not to clip negative pixel "
            "values in output. Because the pixel values "
            "in the raw  movies are set by the current "
            "coming off a photomultiplier tube, there can "
            "be pixels with negative values (current has a "
            "sign), possibly due to noise in the rig. "
            "Some segmentation algorithms cannot handle "
            "negative values in the movie, so we have this "
            "option to artificially set those pixels to zero."
        ),
    )
    max_reference_iterations = argschema.fields.Int(
        required=False,
        default=8,
        description="Maximum number of iterations to preform when creating a "
        "reference image.",
    )
    auto_remove_empty_frames = argschema.fields.Boolean(
        required=False,
        default=True,
        allow_none=False,
        description="Automatically detect empty noise frames at the start and "
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
        "to avoid large motion borders.",
    )
    trim_frames_start = argschema.fields.Int(
        required=False,
        default=0,
        allow_none=False,
        description="Number of frames to remove from the start of the movie "
        "if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still "
        "written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and "
        "trim_frames_start > 0",
    )
    trim_frames_end = argschema.fields.Int(
        required=False,
        default=0,
        allow_none=False,
        description="Number of frames to remove from the end of the movie "
        "if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still "
        "written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and "
        "trim_frames_start > 0",
    )
    do_optimize_motion_params = argschema.fields.Bool(
        default=False,
        required=False,
        description="Do a search for best parameters of smooth_sigma and "
        "smooth_sigma_time. Adds significant runtime cost to "
        "motion correction and should only be run once per "
        "experiment with the resulting parameters being stored "
        "for later use.",
    )
    use_ave_image_as_reference = argschema.fields.Bool(
        default=False,
        required=False,
        description="Only available if `do_optimize_motion_params` is set. "
        "After the a best set of smoothing parameters is found, "
        "use the resulting average image as the reference for the "
        "full registration. This can be used as two step "
        "registration by setting by setting "
        "smooth_sigma_min=smooth_sigma_max and "
        "smooth_sigma_time_min=smooth_sigma_time_max and "
        "steps=1.",
    )
    n_batches = argschema.fields.Int(
        default=20,
        required=False,
        description="Number of batches of size suite2p_args['batch_size'] to "
        "load from the movie for smoothing parameter testing. "
        "Batches are evenly spaced throughout the movie.",
    )
    smooth_sigma_min = argschema.fields.Float(
        default=0.65,
        required=False,
        description="Minimum value of the parameter search for smooth_sigma.",
    )
    smooth_sigma_max = argschema.fields.Float(
        default=2.15,
        required=False,
        description="Maximum value of the parameter search for smooth_sigma.",
    )
    smooth_sigma_steps = argschema.fields.Int(
        default=4,
        required=False,
        description="Number of steps to grid between smooth_sigma and "
        "smooth_sigma_max. Large values will add significant time "
        "motion correction.",
    )
    smooth_sigma_time_min = argschema.fields.Float(
        default=0,
        required=False,
        description="Minimum value of the parameter search for " "smooth_sigma_time.",
    )
    smooth_sigma_time_max = argschema.fields.Float(
        default=6,
        required=False,
        description="Maximum value of the parameter search for " "smooth_sigma_time.",
    )
    smooth_sigma_time_steps = argschema.fields.Int(
        default=7,
        required=False,
        description="Number of steps to grid between smooth_sigma and "
        "smooth_sigma_time_max. Large values will add significant "
        "time motion correction.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpdir = None

    @mm.pre_load
    def set_default_outputs(self, data, **kwargs):
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

    @mm.post_load
    def check_trim_frames(self, data, **kwargs):
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


class Suite2PRegistrationOutputSchema(argschema.schemas.DefaultSchema):
    motion_corrected_output = argschema.fields.InputFile(
        required=True,
        description="destination path for hdf5 motion corrected video.",
    )
    motion_diagnostics_output = argschema.fields.InputFile(
        required=True,
        description=("Path of *.csv file containing motion correction offsets"),
    )
    max_projection_output = argschema.fields.InputFile(
        required=True,
        description=(
            "Desired path for *.png of the max projection of the "
            "motion corrected video."
        ),
    )
    avg_projection_output = argschema.fields.InputFile(
        required=True,
        description=(
            "Desired path for *.png of the avg projection of the "
            "motion corrected video."
        ),
    )
    registration_summary_output = argschema.fields.InputFile(
        required=True, description="Desired path for *.png for summary QC plot"
    )
    motion_correction_preview_output = argschema.fields.InputFile(
        required=True, description="Desired path for *.webm motion preview"
    )
