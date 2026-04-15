"""Centralised aind-data-schema interaction helpers.

All imports of ``aind-data-schema`` and ``aind-data-schema-models`` symbols
live in this module. Callers interact with schema objects only through the
functions here, so schema-version changes require edits in one place.
"""

from datetime import datetime as dt
from pathlib import Path

import pytz
from aind_data_schema.components.configs import (
    CoupledPlane,
    ImagingConfig,
    PlanarImage,
)
from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import DropdownMetric

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_acquisition(path: Path) -> Acquisition:
    """Parse an acquisition.json file into an Acquisition object.

    Parameters
    ----------
    path : Path
        Path to the acquisition.json file.

    Returns
    -------
    Acquisition
        Validated Acquisition object.
    """
    return Acquisition.model_validate_json(path.read_text())


def load_data_description(path: Path) -> DataDescription:
    """Parse a data_description.json file into a DataDescription object.

    Parameters
    ----------
    path : Path
        Path to the data_description.json file.

    Returns
    -------
    DataDescription
        Validated DataDescription object.
    """
    return DataDescription.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Acquisition navigation
# ---------------------------------------------------------------------------


def get_imaging_planes(acquisition: Acquisition) -> list:
    """Return all Plane objects from acquisition, sorted by plane_index.

    Traverses data_streams → configurations (ImagingConfig) →
    images (PlanarImage) → planes.

    Works for both mesoscope (CoupledPlane, has plane_index) and Bergamo
    (plain Plane, no plane_index). Planes without plane_index sort to the
    front (key defaults to 0).

    Parameters
    ----------
    acquisition : Acquisition
        Validated Acquisition object.

    Returns
    -------
    list
        Plane objects sorted by plane_index (or 0 if absent).
    """
    planes = []
    for data_stream in acquisition.data_streams:
        for config in data_stream.configurations:
            if isinstance(config, ImagingConfig):
                for image in config.images:
                    if isinstance(image, PlanarImage):
                        for plane in image.planes:
                            planes.append(plane)
    return sorted(planes, key=lambda p: getattr(p, "plane_index", 0))


def get_fov_id(plane: CoupledPlane) -> str:
    """Return the FOV identifier string for an imaging plane.

    Format: ``"{targeted_structure_acronym}_{plane_index}"``.
    For single-plane instruments (e.g. Bergamo) that use plain ``Plane``
    objects without a ``plane_index`` attribute, the index defaults to 0.

    Parameters
    ----------
    plane : CoupledPlane
        Imaging plane with targeted_structure; plane_index is optional.

    Returns
    -------
    str
        FOV identifier string, e.g. ``"MO_0"`` or ``"VISp_3"``.
    """
    plane_index = getattr(plane, "plane_index", 0)
    return f"{plane.targeted_structure.acronym}_{plane_index}"


# ---------------------------------------------------------------------------
# QC schema helpers
# ---------------------------------------------------------------------------


def pending_qc_status() -> QCStatus:
    """Return a PENDING QCStatus timestamped to US/Pacific.

    Returns
    -------
    QCStatus
        A QCStatus with status=PENDING and evaluator='Automated'.
    """
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return QCStatus(
        evaluator="Automated",
        status=Status.PENDING,
        timestamp=dt.now(seattle_tz).isoformat(),
    )


def build_fov_qc_metric(
    unique_id: str,
    raw_tif_path: Path,
    intended_depth: int,
    targeted_structure_id: str,
    closest_z: float,
    merged_path: Path,
) -> QCMetric:
    """Construct the FOV parent-child matching QCMetric object.

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

    Returns
    -------
    QCMetric
        Constructed QCMetric object, not yet serialized.
    """
    return QCMetric(
        name=f"{unique_id} Parent-Child FOV",
        modality=Modality.POPHYS,
        stage=Stage.RAW,
        tags={
            "evaluation": "Op. QC: Field-of-view Matching",
            "type": "Operational QC",
        },
        description=(
            f"Side-by-side comparison of the parent session depth snapshot "
            f"({raw_tif_path.stem}, intended_depth={intended_depth}µm, "
            f"structure={targeted_structure_id}) and the current session's "
            f"averaged-depth image at the closest matching z-plane "
            f"(child z: {closest_z}µm). "
            f"Pass if the fields of view align spatially across sessions."
        ),
        status_history=[pending_qc_status()],
        reference=str(merged_path),
        value=DropdownMetric(
            value="",
            options=[
                "FOV Matches parent FOV",
                "FOV does not match parent FOV.",
            ],
            status=[Status.PASS, Status.FAIL],
        ),
    )


def build_vasculature_qc_metric(vasculature_output_fp: Path) -> QCMetric:
    """Construct the vasculature window-clarity QCMetric object.

    Parameters
    ----------
    vasculature_output_fp : Path
        Path to the saved vasculature PNG.

    Returns
    -------
    QCMetric
        Constructed QCMetric object, not yet serialized.
    """
    return QCMetric(
        name="Vasculature Image",
        modality=Modality.POPHYS,
        stage=Stage.RAW,
        tags={
            "evaluation": "Op. QC: Window Clarity",
            "type": "Operational QC",
        },
        description=(
            "Vasculature image captured at session start to assess brain "
            "surface health and cranial window clarity. Select the option "
            "that best describes the observed surface condition. Bruising, "
            "bubbles, or discoloration may indicate compromised tissue or "
            "optical path quality."
        ),
        status_history=[pending_qc_status()],
        reference=str(vasculature_output_fp),
        value=DropdownMetric(
            value="",
            options=[
                "Quality is sufficient",
                "Poor vasculature image quality",
                "Light bruising on surface of brain",
                "Severe bruising on surface of brain",
                "Vascularization of brain surface",
                "Discoloration of brain surface (white)",
                (
                    "Bubbles in objective immersion present, but do NOT"
                    " impact imaging quality"
                ),
                "Bubbles in objective immersion impact imaging quality",
            ],
            status=[
                Status.PASS,
                Status.PASS,
                Status.PASS,
                Status.FAIL,
                Status.PASS,
                Status.PASS,
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )
