"""Centralised aind-data-schema interaction helpers.

All imports of ``aind-data-schema`` and ``aind-data-schema-models`` symbols
live in this module. Callers interact with schema objects only through the
functions here, so schema-version changes require edits in one place.
"""

import json
import logging
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

import pytz
from aind_data_schema.components.configs import ImagingConfig
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Stage, Status
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName
from aind_qcportal_schema.metric_value import DropdownMetric

logger = logging.getLogger(__name__)

CODE_URL = (
    "https://github.com/AllenNeuralDynamics/"
    "aind-ophys-motion-correction/tree/main/code"
)

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_acquisition(path: Path) -> Acquisition:
    """Parse an acquisition.json file into an Acquisition object."""
    return Acquisition.model_validate_json(path.read_text())


def load_data_description(path: Path) -> DataDescription:
    """Parse a data_description.json file into a DataDescription object."""
    return DataDescription.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Acquisition navigation
# ---------------------------------------------------------------------------


def get_frame_rate(acquisition: Acquisition) -> Optional[float]:
    """Extract frame rate from acquisition data_streams.

    Traverses the v2 ImagingConfig hierarchy:
    data_streams → configurations[ImagingConfig] → sampling_strategy.frame_rate.

    Returns None if frame rate cannot be found.
    """
    for data_stream in acquisition.data_streams:
        for config in data_stream.configurations:
            if isinstance(config, ImagingConfig):
                sampling = getattr(config, "sampling_strategy", None)
                if sampling and hasattr(sampling, "frame_rate"):
                    return float(sampling.frame_rate)
    return None


def get_bci_conditioning_epochs(acquisition: Acquisition) -> list:
    """Return stimulus epochs for 'single neuron BCI conditioning'.

    Used by the Bergamo single-plane path to locate BCI epoch
    boundaries for reference image generation.
    """
    return [
        ep
        for ep in (acquisition.stimulus_epochs or [])
        if ep.stimulus_name == "single neuron BCI conditioning"
    ]


# ---------------------------------------------------------------------------
# QC schema helpers
# ---------------------------------------------------------------------------


def pending_qc_status() -> QCStatus:
    """Return a PENDING QCStatus timestamped to US/Pacific."""
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return QCStatus(
        evaluator="Pending review",
        status=Status.PENDING,
        timestamp=dt.now(seattle_tz).isoformat(),
    )


def build_registration_summary_metric(
    unique_id: str,
    reference_filepath: str,
) -> QCMetric:
    """Construct the Registration Summary QCMetric."""
    return QCMetric(
        name=f"{unique_id} Registration Summary",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "Registration Summary", "type": "Operational QC"},
        description=(
            "Review the registration summary plot to ensure that the "
            "motion correction is accurate and sufficient."
        ),
        status_history=[pending_qc_status()],
        reference=reference_filepath,
        value=DropdownMetric(
            value="",
            options=[
                "Motion correction successful",
                "No motion correction applied",
                "Motion correction failed",
                "Motion correction partially successful",
            ],
            status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL],
        ),
    )


def build_fov_quality_metric(
    unique_id: str,
    reference_filepath: str,
) -> QCMetric:
    """Construct the FOV Quality QCMetric."""
    return QCMetric(
        name=f"{unique_id} FOV Quality",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "FOV Quality", "type": "Operational QC"},
        description=(
            "Review the avg. and max. projections to ensure that the "
            "FOV quality is sufficient."
        ),
        status_history=[pending_qc_status()],
        reference=reference_filepath,
        value=DropdownMetric(
            value="Quality is sufficient",
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


# ---------------------------------------------------------------------------
# DataProcess helpers
# ---------------------------------------------------------------------------


def build_data_process(
    parameters: dict,
    start_time: dt,
    end_time: dt,
    output_path: Optional[str] = None,
    output_parameters: Optional[dict] = None,
) -> DataProcess:
    """Construct a v2 DataProcess for motion correction.

    Parameters
    ----------
    parameters : dict
        Capsule configuration parameters (MotionCorrectionSettings).
    start_time : dt
        Processing start time.
    end_time : dt
        Processing end time.
    output_path : str, optional
        Relative path to the parent output directory.
    output_parameters : dict, optional
        Actual parameters used during processing and output metrics.
    """
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return DataProcess(
        process_type=ProcessName.VIDEO_MOTION_CORRECTION,
        name="Suite2P motion correction",
        stage=ProcessStage.PROCESSING,
        code=Code(
            url=CODE_URL,
            name="aind-ophys-motion-correction",
            version=os.getenv("VERSION", ""),
            parameters=parameters,
        ),
        experimenters=[],
        start_date_time=start_time.astimezone(seattle_tz),
        end_date_time=end_time.astimezone(seattle_tz),
        output_path=output_path,
        output_parameters=output_parameters,
    )


def write_processing(
    data_process: DataProcess,
    output_dir: Path,
) -> None:
    """Write a Processing object containing a single DataProcess to processing.json.

    Parameters
    ----------
    data_process : DataProcess
        The data process to write.
    output_dir : Path
        Directory to write processing.json into.
    """
    processing = Processing(data_processes=[data_process])
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    out_path = output_dir / "processing.json"
    with open(out_path, "w") as f:
        json.dump(json.loads(processing.model_dump_json()), f, indent=4)
    logger.info("Saved processing metadata -> %s", out_path)
