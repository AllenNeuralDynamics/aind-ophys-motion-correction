# aind-data-schema v1 → v2 Upgrade Guide

**Audience:** Any AIND Code Ocean capsule (or its companion library) that depends on `aind-data-schema` v1.x and needs to move to v2.x.

**Proven on:** `aind-pophys-converter` + `aind-pophys-converter-capsule` (Feb–Mar 2026).

---

## Table of Contents

1. [Version Targets](#1-version-targets)
2. [Breaking Changes](#2-breaking-changes)
3. [QC v2 Reference API](#3-qc-v2-reference-api)
4. [Target Capsule Architecture](#4-target-capsule-architecture)
5. [Structured Logging with aind-log-utils](#5-structured-logging-with-aind-log-utils)
6. [Execution Playbook](#6-execution-playbook)
7. [Gotchas & Lessons Learned](#7-gotchas--lessons-learned)

---

## 1. Version Targets

| Package | v1 (typical) | v2 Target | Notes |
|---------|-------------|-----------|-------|
| `aind-data-schema` | 1.4.0 | `>=2.6.0,<3` | Core schema models |
| `aind-data-schema-models` | 0.7.5 | `>=5.4.1,<6` | Auto-generated enums/registries |
| `aind-qcportal-schema` | 0.4.0 | `>=0.6.4` | DropdownMetric, etc. Verify compatibility. |
| `aind-log-utils` | (none) | `>=0.2.6` | Optional — structured logging |
| `pydantic-settings` | (varies) | `>=2.8.1` | For `BaseSettings(cli_parse_args=True)` |

Update these in **two places** per capsule:
- `environment/Dockerfile`
- `.codeocean/environment.json` (mirrors Dockerfile)

---

## 2. Breaking Changes

### 2.1 QCEvaluation DELETED — Flat Metric Model

The biggest change. The hierarchical `QualityControl → QCEvaluation → QCMetric` model is now flat: `QualityControl → QCMetric`.

```python
# v1 imports:
from aind_data_schema.core.quality_control import (
    Modality, QCEvaluation, QCMetric, QCStatus, Stage, Status,
)

# v2 imports:
from aind_data_schema_models.modalities import Modality          # MOVED
from aind_data_schema.core.quality_control import (
    QCMetric, QCStatus, QualityControl, Stage, Status,            # QCEvaluation GONE
)
```

Key differences:
- `QCEvaluation` — **deleted**, remove all references
- `Modality` — moved from `aind_data_schema.core.quality_control` → `aind_data_schema_models.modalities`
- `QualityControl` — new top-level container (replaces `QCEvaluation` as the output object)

### 2.2 QCMetric Now Requires `modality` + `stage`

In v1, `modality` and `stage` lived on `QCEvaluation`. In v2, each `QCMetric` carries its own.

```python
# v1 — modality/stage on the wrapper:
metric = QCMetric(name="...", value=..., status_history=[...])
evaluation = QCEvaluation(
    name="My Evaluation",
    metrics=[metric],
    modality=Modality.POPHYS,
    stage=Stage.RAW,
    tags=["Operational QC"],
)

# v2 — modality/stage on each metric:
metric = QCMetric(
    name="...",
    modality=Modality.POPHYS,       # REQUIRED — was on QCEvaluation
    stage=Stage.RAW,                 # REQUIRED — was on QCEvaluation
    value=...,
    status_history=[...],
    tags={"evaluation": "My Evaluation", "type": "Operational QC"},  # dict
)
qc = QualityControl(
    metrics=[metric],
    default_grouping=["modality", "stage", ("evaluation",)],
    allow_tag_failures=[],
)
```

### 2.3 Tags: `List[str]` → `dict[str, str]`

```python
# v1 (on QCEvaluation):
tags=["Operational QC"]

# v2 (on each QCMetric):
tags={"category": "Operational QC", "evaluation": "FOV Matching"}
```

Tags enable hierarchical grouping in the QC portal. Choose meaningful keys — `default_grouping` on `QualityControl` references these keys.

### 2.4 QualityControl Requires `default_grouping`

```python
QualityControl(
    metrics=[...],
    default_grouping=["modality", "stage", ("evaluation",)],  # REQUIRED
    allow_tag_failures=[],   # tag values allowed to fail without failing overall QC
)
```

### 2.5 QCStatus Timezone-Aware Datetime

`QCStatus.timestamp` now enforces timezone-aware datetimes. Use `pytz` or `datetime.timezone`:

```python
import pytz
from datetime import datetime as dt

seattle_tz = pytz.timezone("America/Los_Angeles")
status = QCStatus(
    evaluator="Automated",
    status=Status.PENDING,
    timestamp=dt.now(seattle_tz).isoformat(),
)
```

### 2.6 Output Format: Single `quality_control.json`

**v1:** Each `QCEvaluation` saved as a separate JSON file (e.g., `merged_planes_evaluation.json`).

**v2:** All metrics go into one `QualityControl` object, written as a single `quality_control.json`. The QC portal expects this format.

If your capsule writes multiple QC files today, you have two options:
- **Option A:** Merge all metrics into one `quality_control.json` at the end (preferred).
- **Option B:** Write individual metric JSONs during processing, then assemble them into a single `QualityControl` at the end.

### 2.7 Metadata File Renames

| v1 Filename | v2 Filename | Model Class |
|-------------|-------------|-------------|
| `session.json` | `acquisition.json` | `Acquisition` (was `Session`) |
| `rig.json` | `instrument.json` | `Instrument` (was `Rig`) |

Other files unchanged: `subject.json`, `data_description.json`, `procedures.json`, `processing.json`.

**Search your codebase for:**
- `session.json` (file globs, `rglob`, `open()` calls)
- `rig.json` (same)
- `session_fp`, `session_path` (variable names)
- `rig_fp`, `rig_path` (variable names)

### 2.8 Field Renames Inside Acquisition (was Session)

If your code reads `session.json` / `acquisition.json` as raw JSON or via the schema model:

| v1 field (session.json) | v2 field (acquisition.json) | Notes |
|---|---|---|
| `rig_id` | `instrument_id` | Used for rig-type detection (e.g. "Bergamo") |
| `data_streams[].ophys_fovs` | Gone — restructured | See §2.9 |
| `data_streams[].ophys_fovs[].scanfield_z` | Nested in Plane objects | See §2.9 |
| `data_streams[].ophys_fovs[].scanimage_roi_index` | `CoupledPlane.plane_index` or similar | Check your modality |
| `stimulus_epochs` | `stimulus_epochs` (still exists) | Structure changed — check `StimulusEpoch` model |

### 2.9 DataStream Restructuring (ophys_fovs → ImagingConfig)

The `ophys_fovs` list inside `DataStream` is **gone**. Imaging data now lives in a typed configuration hierarchy:

```
v1: data_streams[].ophys_fovs[].scanfield_z
v2: data_streams[].configurations[ImagingConfig].images[PlanarImage].planes[Plane]
```

**v2 traversal pattern (from the pophys-converter upgrade):**
```python
from aind_data_schema.components.configs import (
    CoupledPlane, ImagingConfig, PlanarImage,
)
from aind_data_schema.core.acquisition import Acquisition

def get_imaging_planes(acquisition: Acquisition) -> list:
    """Return all Plane objects from acquisition data_streams."""
    planes = []
    for data_stream in acquisition.data_streams:
        for config in data_stream.configurations:
            if isinstance(config, ImagingConfig):
                for image in config.images:
                    if isinstance(image, PlanarImage):
                        for plane in image.planes:
                            planes.append(plane)
    return sorted(planes, key=lambda p: getattr(p, "plane_index", 0))
```

**Frame rate** is **not** on `PlanarImage` — it lives on `ImagingConfig.sampling_strategy.frame_rate`:

```python
def get_frame_rate(acquisition: Acquisition) -> Optional[float]:
    for data_stream in acquisition.data_streams:
        for config in data_stream.configurations:
            if isinstance(config, ImagingConfig):
                sampling = getattr(config, "sampling_strategy", None)
                if sampling and hasattr(sampling, "frame_rate"):
                    return float(sampling.frame_rate)
    return None
```

**Not every capsule needs this.** If your capsule only reads `instrument_id` or `stimulus_epochs` and doesn't traverse FOV data, you can skip this section. Search your code for `ophys_fovs`, `scanfield_z`, `scanimage_roi_index` to check.

### 2.10 Modality Import Path Change

```python
# v1:
from aind_data_schema.core.quality_control import Modality

# v2:
from aind_data_schema_models.modalities import Modality
```

Common modalities: `Modality.POPHYS`, `Modality.BEHAVIOR`, `Modality.BEHAVIOR_VIDEOS`, `Modality.FIB`.

### 2.11 Processing & DataProcess Overhaul

#### 2.11.1 Output Structure: `processing.json`

In v1, capsules typically wrote individual `*_data_process.json` files per processing step. In v2, all steps belong in a single `processing.json` using the `Processing` container:

```python
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage

processing = Processing(
    data_processes=[data_proc_1, data_proc_2, ...],
)

with open(output_dir / "processing.json", "w") as f:
    json.dump(json.loads(processing.model_dump_json()), f, indent=4)
```

The docs state: "the processing file should be appended to with each subsequent stage of processing or analysis." For capsules that run a single step, this is just one `DataProcess` in the list.

#### 2.11.2 DataProcess Constructor

`DataProcess` has a **completely different constructor** in v2. This is easy to miss — it was not documented in early v2 release notes.

```python
# v1:
from aind_data_schema.core.processing import DataProcess

data_proc = DataProcess(
    name=ProcessName.VIDEO_MOTION_CORRECTION,
    software_version="1.0.0",
    start_date_time=start_time.isoformat(),
    end_date_time=end_time.isoformat(),
    input_location="/data/input.h5",
    output_location="/results/output.h5",
    code_url="https://github.com/...",
    parameters={"batch_size": 500},
)

# v2:
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage

seattle_tz = pytz.timezone("America/Los_Angeles")

data_proc = DataProcess(
    process_type=ProcessName.VIDEO_MOTION_CORRECTION,
    name="Suite2P motion correction",         # Optional — defaults to process_type
    stage=ProcessStage.PROCESSING,
    code=Code(
        url="https://github.com/...",
        name="my-capsule",
        version=os.getenv("VERSION", ""),
        parameters={                           # Capsule's own configuration
            "input_dir": "/data/asset_name",
            "output_dir": "/results",
            "batch_size": 500,
            "nonrigid": True,
            "maxregshift": 0.1,
        },
    ),
    experimenters=[],
    start_date_time=start_time.astimezone(seattle_tz),
    end_date_time=end_time.astimezone(seattle_tz),
    output_path="/results/VISl_7/motion_correction/",  # Parent dir of all outputs
    output_parameters={                        # Two top-level keys:
        "args": {                              #   all params actually used (incl. dependency defaults)
            "batch_size": 500,
            "nonrigid": True,
            "suite2p_args": {"smooth_sigma": 1.15, "...": "..."},
        },
        "metrics": {                           #   output metrics from processing
            "crispness": [1.2, 1.8],
            "frames_clipped": 5,
        },
    },
)

processing = Processing(data_processes=[data_proc])
```

#### 2.11.3 Field Migration Table

| v1 field | v2 equivalent | Notes |
|---|---|---|
| `name` (ProcessName) | `process_type` | Renamed |
| — | `name` (str, optional) | New — free-text name, defaults to `process_type` |
| `software_version` | `Code.version` | Moved into `Code` object |
| `code_url` | `Code.url` | Moved into `Code` object |
| `parameters` | `Code.parameters` | Moved into `Code` — these are **input** parameters |
| `input_location` | Removed | No direct replacement in v2 |
| `output_location` | `output_path` | Now `Optional[AssetPath]` — **relative** path from metadata root |
| — | `output_parameters` | New — `Optional[dict]` for output metrics/summary stats |
| — | `stage` (required) | New — `ProcessStage.PROCESSING` or `.ANALYSIS` |
| — | `code` (required) | New — `Code` object |
| — | `code.name` | New — optional, name of the capsule/tool |
| — | `experimenters` (required) | New — `List[str]`, can be empty |
| `start_date_time` | `start_date_time` | Now requires **timezone-aware** datetime |
| — | `resources` | New — optional `ResourceUsage` for CPU/GPU/RAM tracking |

#### 2.11.4 Key Design Decisions

- **`Code.parameters`** = the capsule's own configuration — the parameters passed into *this code*, not the internal defaults of a dependency. For example, if your capsule wraps Suite2P, `Code.parameters` should contain the capsule's settings (`batch_size`, `nonrigid`, `maxregshift`, etc.), not Suite2P's full internal ops dict.
- **`output_parameters`** = a structured dict with two top-level keys: `"args"` (all parameters actually used during processing, including dependency internals like `suite2p_args`) and `"metrics"` (output metrics like `crispness`, `frames_clipped`). This separation makes it easy for downstream consumers to distinguish configuration from results.
- **`output_path`** is a relative path to the **parent directory** containing all outputs for this processing step (e.g. `"VISl_7/motion_correction/"`), not a path to a single file. It's relative to the metadata root folder.
- **Write `processing.json` at the end** of your capsule, after all outputs are generated, so you can populate `output_path` and `output_parameters` with actual results.
- **Timestamps must be timezone-aware.** Use `pytz.timezone("America/Los_Angeles")` and `.astimezone()`.

#### 2.11.5 Search Your Codebase

Look for these patterns to find code that needs updating:
- `DataProcess(` — constructor args changed
- `*_data_process.json` — should become a single `processing.json`
- `input_location` / `output_location` / `code_url` / `software_version` — removed fields
- `start_date_time=...isoformat()` — must be timezone-aware datetime, not string

### 2.12 setup_logging Kwarg Renames

`aind-log-utils` deprecated kwargs in recent versions:
- `mouse_id` → `subject_id`
- `session_name` → `asset_name`

---

## 3. QC v2 Reference API

### QCMetric

```python
class QCMetric(DataModel):
    name: str                                    # Metric name
    modality: Modality.ONE_OF                    # Required (was on QCEvaluation)
    stage: Stage                                 # Required (was on QCEvaluation)
    value: Any                                   # Scalar, dict, DropdownMetric, etc.
    status_history: List[QCStatus]               # Min length 1
    description: Optional[str] = None
    reference: Optional[str] = None              # Image URL or plot reference
    tags: dict[str, str] = {}                    # Key-value pairs (was List[str])
    evaluated_assets: Optional[List[str]] = None # For MULTI_ASSET stage only
```

### QualityControl

```python
class QualityControl(DataCoreModel):
    metrics: List[QCMetric | CurationMetric]     # Flat list
    key_experimenters: Optional[List[str]] = None
    notes: Optional[str] = None
    default_grouping: List[str | tuple[str, ...]] # Required — tag keys for viz
    allow_tag_failures: List[str] = []            # Tag values that can fail
    status: Optional[dict] = None                 # Auto-computed by validator
```

### Stage Enum

```python
Stage.RAW          # "Raw data"
Stage.PROCESSING   # "Processing"
Stage.ANALYSIS     # "Analysis"
Stage.MULTI_ASSET  # "Multi-asset" (requires evaluated_assets)
```

### Status Enum (unchanged from v1)

```python
Status.FAIL    # "Fail"
Status.PASS    # "Pass"
Status.PENDING # "Pending"
```

### Pending Status Helper

```python
import pytz
from datetime import datetime as dt
from aind_data_schema.core.quality_control import QCStatus, Status

def pending_qc_status() -> QCStatus:
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return QCStatus(
        evaluator="Automated",
        status=Status.PENDING,
        timestamp=dt.now(seattle_tz).isoformat(),
    )
```

### Complete QC Construction Example

```python
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import (
    QCMetric, QualityControl, Stage, Status,
)
from aind_qcportal_schema.metric_value import DropdownMetric

metric = QCMetric(
    name="My Check",
    modality=Modality.POPHYS,
    stage=Stage.RAW,
    tags={"evaluation": "Op. QC: Some Check", "type": "Operational QC"},
    description="What this metric checks.",
    status_history=[pending_qc_status()],
    reference=str(some_image_path),
    value=DropdownMetric(
        value="",
        options=["Pass option", "Fail option"],
        status=[Status.PASS, Status.FAIL],
    ),
)

qc = QualityControl(
    metrics=[metric],
    default_grouping=["modality", "stage", ("evaluation",)],
    allow_tag_failures=[],
)

# Serialize:
import json
with open(output_dir / "quality_control.json", "w") as f:
    json.dump(json.loads(qc.model_dump_json()), f, indent=4)
```

---

## 4. Target Capsule Architecture

### Module Layout

Capsules should organise schema-dependent code into a `code/utils/` package. This isolates all `aind-data-schema` interactions so schema version changes require edits in one place.

```
code/
├── run                        # Bash entry point
├── my_capsule.py              # Main processing logic
└── utils/
    ├── __init__.py
    ├── metadata_utils.py      # Schema imports, loaders, builders
    ├── qc.py                  # QC visual generation + metric serialization
    ├── logging_utils.py       # Structured JSON logging
    └── logging.yml            # dictConfig YAML
```

### `utils/metadata_utils.py` — Schema Interaction Layer

All `aind-data-schema` and `aind-data-schema-models` imports live here. Provides:

- **Document loaders** — `load_acquisition()`, `load_data_description()` that return validated Pydantic models
- **Complex traversals** — e.g. `get_frame_rate()` for navigating `ImagingConfig.sampling_strategy`, `get_bci_conditioning_epochs()` for filtering stimulus epochs
- **QC metric builders** — `build_registration_summary_metric()`, `build_fov_quality_metric()`, `pending_qc_status()` that construct v2 `QCMetric` objects with correct `modality`, `stage`, `tags`, and timezone-aware timestamps
- **DataProcess builder** — `build_data_process()` that constructs v2 `DataProcess` with `Code`, `ProcessStage`, etc.
- **Processing writer** — `write_processing()` that wraps `DataProcess` in a `Processing` container and writes `processing.json`

**Do not create simple getter wrappers** for trivial attribute access (e.g. `get_instrument_id(acq)` for `acq.instrument_id`). Only add functions for complex parsing or logic that encodes v2-specific structure.

```python
# utils/metadata_utils.py — minimal example
from aind_data_schema.components.configs import ImagingConfig
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Stage, Status
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName
from aind_qcportal_schema.metric_value import DropdownMetric

def load_acquisition(path):
    return Acquisition.model_validate_json(path.read_text())

def load_data_description(path):
    return DataDescription.model_validate_json(path.read_text())

def get_frame_rate(acquisition):
    for ds in acquisition.data_streams:
        for config in ds.configurations:
            if isinstance(config, ImagingConfig):
                sampling = getattr(config, "sampling_strategy", None)
                if sampling and hasattr(sampling, "frame_rate"):
                    return float(sampling.frame_rate)
    return None

def pending_qc_status():
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return QCStatus(
        evaluator="Pending review",
        status=Status.PENDING,
        timestamp=dt.now(seattle_tz).isoformat(),
    )

def build_my_metric(unique_id, reference_filepath):
    return QCMetric(
        name=f"{unique_id} My Check",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "My Check", "type": "Operational QC"},
        status_history=[pending_qc_status()],
        reference=reference_filepath,
        value=DropdownMetric(
            value="",
            options=["Pass", "Fail"],
            status=[Status.PASS, Status.FAIL],
        ),
    )

def build_data_process(parameters, start_time, end_time, output_path=None, output_parameters=None):
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return DataProcess(
        process_type=ProcessName.VIDEO_MOTION_CORRECTION,
        name="My processing step",
        stage=ProcessStage.PROCESSING,
        code=Code(url=CODE_URL, name="my-capsule", version=os.getenv("VERSION", ""), parameters=parameters),
        experimenters=[],
        start_date_time=start_time.astimezone(seattle_tz),
        end_date_time=end_time.astimezone(seattle_tz),
        output_path=output_path,
        output_parameters=output_parameters,
    )

def write_processing(data_process, output_dir):
    processing = Processing(data_processes=[data_process])
    with open(output_dir / "processing.json", "w") as f:
        json.dump(json.loads(processing.model_dump_json()), f, indent=4)
```

### `utils/qc.py` — QC Visual Generation + Serialization

Handles image manipulation (combining projections, adding labels) and writes metric JSON files. Imports metric builders from `metadata_utils`.

```python
# utils/qc.py — pattern
from utils.metadata_utils import build_my_metric

def write_my_metric(output_dir):
    # 1. Generate/find the reference image
    file_path = next(output_dir.rglob("*_my_image.png"))
    reference_filepath = Path(*file_path.parts[2:])  # relative to /results
    unique_id = reference_filepath.parts[0]

    # 2. Build the metric via metadata_utils
    metric = build_my_metric(unique_id, str(reference_filepath))

    # 3. Serialize
    with open(file_path.parent / f"{unique_id}_my_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
```

### Main Script Pattern

```python
# my_capsule.py
from utils.logging_utils import setup_logging
from utils.metadata_utils import load_acquisition, load_data_description, build_data_process, write_processing
from utils.qc import write_my_metric

def run(parser, acquisition):
    # ... do processing ...

    # Write QC metrics
    write_my_metric(output_dir)

    # Write processing.json (at the very end)
    data_proc = build_data_process(
        parameters=capsule_parameters,
        start_time=start_time,
        end_time=dt.now(),
        output_path=str(output_dir),
        output_parameters={"args": args_copy, "metrics": metrics},
    )
    write_processing(data_proc, output_dir)

if __name__ == "__main__":
    parser = MySettings()
    acquisition = load_acquisition(next(parser.input_dir.rglob("acquisition.json")))
    data_description = load_data_description(next(parser.input_dir.rglob("data_description.json")))

    setup_logging("my-capsule", subject_id=data_description.subject_id or "", acquisition_name=data_description.name or "")

    logger.info("Starting", extra={"event_type": "stage_start"})
    try:
        run(parser, acquisition)
        logger.info("Complete", extra={"event_type": "stage_complete"})
    except Exception:
        logger.exception("Failed", extra={"event_type": "stage_error"})
        raise
```

---

## 5. Structured Logging

Optional but recommended. Provides structured JSON logs + CloudWatch integration on Code Ocean.

### Recommended: Custom YAML-Based Logging

Use a custom `logging.yml` + `logging_utils.py` module instead of `aind-log-utils`. This gives full control over formatters, handlers, and context fields without an external dependency. Requires `pyyaml` and optionally `watchtower` (for CloudWatch on Code Ocean).

```
code/utils/
├── logging_utils.py    # setup_logging(), AindJsonFormatter, AindContextFilter, LoggingStream
└── logging.yml         # dictConfig YAML (formatters, filters, handlers)
```

### `utils/logging.yml`

Copy this file into your `code/utils/` directory. Update the `()` class paths if your module lives elsewhere.

```yaml
# AIND logging configuration (logging.config.dictConfig format)
#
# Fields emitted per log record are controlled by the `fields` list below.
# Add or remove entries to change what appears in each JSON log line.
# Standard LogRecord attributes (lineno, process, processName, thread,
# threadName) can be added by name. Custom fields injected by the filter
# (acquisition_name, process_name) are also referenced by name here.
#
# Special fields handled by the formatter:
#   timestamp  - ISO8601 local time derived from record.created
#   level      - maps to record.levelname
#   message    - maps to record.getMessage()
#
# The formatter and filter classes live in logging_utils.py (same folder).
# Load this file with setup_logging() from logging_utils.py.

version: 1
disable_existing_loggers: false

formatters:
  aind_json:
    # () tells dictConfig to instantiate this class directly
    (): utils.logging_utils.AindJsonFormatter
    fields:
      - timestamp
      - level
      - message
      - acquisition_name
      - process_name
      - event_type

filters:
  aind_fields:
    (): utils.logging_utils.AindContextFilter
    # Override these values at runtime by passing **fields to setup_logging()
    acquisition_name: "undefined"
    process_name: "undefined"

handlers:
  console:
    class: logging.StreamHandler
    formatter: aind_json
    filters:
      - aind_fields
    stream: ext://sys.stdout
    level: DEBUG

root:
  level: INFO
  handlers:
    - console
```

### `utils/logging_utils.py`

Copy this file into your `code/utils/` directory.

```python
"""AIND log formatters, filters, and handler setup for use with logging.yml dictConfig."""

import datetime
import json
import logging
import logging.config
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.yml")


def setup_logging(log_stream: str, **fields) -> None:
    """Load logging.yml and configure logging for the application.

    Also adds a CloudWatch handler so logs are sent to the default log group.
    Requires valid AWS credentials at runtime.

    Parameters
    ----------
    log_stream:
        CloudWatch log stream name, e.g. the process or capsule name.
    **fields:
        Arbitrary context fields to inject into every log record, e.g.
        ``acquisition_name="123456_2026-03-23_10-00-00"``,
        ``process_name="my-capsule"``.
    """
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        config["filters"]["aind_fields"].update(fields)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.DEBUG)
        logging.warning("logging.yml not found — falling back to basicConfig")

    add_cloudwatch_handler(log_stream=log_stream)


def add_cloudwatch_handler(log_stream: str) -> None:
    """Programmatically add a CloudWatch handler to the root logger."""
    try:
        import watchtower
    except ImportError:
        logging.warning("watchtower not installed — skipping CloudWatch handler")
        return

    handler = watchtower.CloudWatchLogHandler(
        log_stream_name=log_stream,
        log_group="aind/internal-logs"
    )

    root_logger = logging.getLogger()
    formatter = AindJsonFormatter()
    for existing_handler in root_logger.handlers:
        if isinstance(existing_handler.formatter, AindJsonFormatter):
            formatter = existing_handler.formatter
        for f in existing_handler.filters:
            if isinstance(f, AindContextFilter):
                handler.addFilter(f)
                break

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class AindJsonFormatter(logging.Formatter):
    """Formats log records as JSON with AIND standard fields."""

    _DEFAULT_FIELDS = ["timestamp", "level", "message", "acquisition_name"]

    def __init__(self, fields=None):
        super().__init__()
        self.fields = fields or self._DEFAULT_FIELDS

    def format(self, record):
        log_entry = {}
        for field in self.fields:
            if field == "timestamp":
                log_entry["timestamp"] = (
                    datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
                    .astimezone()
                    .isoformat(timespec="milliseconds")
                )
            elif field == "level":
                log_entry["level"] = record.levelname
            elif field == "message":
                log_entry["message"] = record.getMessage()
            else:
                value = getattr(record, field, None)
                if value is not None and value != "undefined":
                    log_entry[field] = value
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class AindContextFilter(logging.Filter):
    """Injects arbitrary context fields into every log record."""

    def __init__(self, **fields):
        super().__init__()
        self.fields = fields

    def filter(self, record):
        for key, value in self.fields.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class LoggingStream:
    """File-like stream that routes write() calls through a logger.

    Use as a stdout/stderr replacement to capture print() output from
    third-party libraries into structured JSON logs.

    Usage::

        old_stdout = sys.stdout
        sys.stdout = LoggingStream(logger, logging.INFO)
        try:
            third_party_function()
        finally:
            sys.stdout = old_stdout
    """

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, msg):
        if msg.strip():
            self.logger.log(self.level, msg.strip())

    def flush(self):
        pass
```

### Capturing Third-Party stdout

Libraries like Suite2P use `print()` instead of `logging`. Wrap their calls to get structured JSON output:

```python
from utils.logging_utils import LoggingStream

old_stdout, old_stderr = sys.stdout, sys.stderr
sys.stdout = LoggingStream(logger, logging.INFO)
sys.stderr = LoggingStream(logger, logging.WARNING)
try:
    suite2p.run_s2p(ops)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr
```

### Stage Lifecycle Events

Use `event_type` in `extra={}` for pipeline observability. Place these in the `__main__` block, **after** `setup_logging()` so they use the structured formatter:

```python
if __name__ == "__main__":
    parser = MySettings()
    acquisition = load_acquisition(...)
    data_description = load_data_description(...)

    setup_logging("my-capsule", subject_id=data_description.subject_id or "", acquisition_name=data_description.name or "")

    logger.info("Starting processing", extra={"event_type": "stage_start"})
    try:
        run(parser, acquisition)
        logger.info("Processing complete", extra={"event_type": "stage_complete"})
    except Exception:
        logger.exception("Processing failed", extra={"event_type": "stage_error"})
        raise
```

### Alternative: aind-log-utils

`aind-log-utils` provides similar functionality as an installable package. If using it, note the v2 kwarg renames: `mouse_id` → `subject_id`, `session_name` → `asset_name`.

---

## 6. Execution Playbook

Do these phases **sequentially**. Each phase is independently testable.

### Phase A: Schema Upgrade (do first)

Get the capsule working on v2 **in place**, keeping the current code structure. Shortest path to a working v2 capsule.

1. Bump deps in `Dockerfile` + `environment.json`
2. Fix imports — `QCEvaluation` → `QualityControl`, `Modality` import path
3. Fix QC construction — flat metrics, tags as dict, `default_grouping`
4. Fix file discovery — `session.json` → `acquisition.json`, `rig_id` → `instrument_id`
5. Fix any raw JSON traversal of session/acquisition structure
6. Update tests (session → acquisition fixtures)
7. **Test against real data**

### Phase B: Capsule/Library Refactor (do second)

Restructure to the metadata-manager pattern. Purely structural — no logic changes.

1. Create `settings.py`, `job.py`, `qc.py` (or equivalent) in the library
2. Create `utils/metadata_utils.py` — centralise all schema imports
3. Move orchestration from capsule → library `job.py`
4. Move QC construction from capsule → library
5. Reduce capsule to ~5 lines
6. Update `pyproject.toml` with new deps
7. Write library tests for new modules
8. **Test against real data**

### Phase C: Structured Logging (do last)

1. Add `aind-log-utils` (or custom logging module) to library + Dockerfile
2. Replace `logging.basicConfig()` with structured logging setup
3. Pass metadata (process_name, subject_id, asset_name) from acquisition/data_description
4. **Test against real data** — verify structured fields in output

### Why This Order?

Schema-first means you can validate against real data immediately. When something breaks you can isolate whether it's a schema issue (A), structural issue (B), or logging issue (C).

---

## 7. Gotchas & Lessons Learned

### From the pophys-converter upgrade:

1. **`rglob` doesn't follow symlinks in Python ≤3.12.** If your dev setup uses symlinks (e.g., test data assembly scripts), use `dir.glob()` instead of `input_dir.rglob()` for files inside symlinked directories.

2. **`QCStatus.timestamp` — use `.isoformat()` on a tz-aware datetime.** Don't pass a raw `datetime` object; v2's validator expects the string form from `.isoformat()`.

3. **`model_dump_json()` then `json.loads()` for serialization.** When writing schema objects to JSON files, use `json.loads(obj.model_dump_json())` to get a plain dict, then `json.dump()`. This avoids issues with non-serializable types.

4. **`aind-qcportal-schema` version matters.** The `DropdownMetric` signature changed between versions. Pin `>=0.6.4` for v2 schema compatibility.

5. **Check if the library reads session.json too.** Even if the library has no `aind-data-schema` import, it may read `session.json` as raw JSON and depend on v1 field names. `grep -r "session.json" src/` to find these.

6. **v2 data assets may arrive with split metadata.** During the transition, TIFF data and v2 metadata may be in separate Code Ocean data assets. Use a test data assembly script with symlinks to combine them for dev testing.

7. **`default_grouping` should reference tag keys you actually use.** A safe default: `["modality", "stage", ("evaluation",)]` — covers the standard AIND portal grouping. The `("evaluation",)` tuple indicates a secondary grouping level.

8. **Centralise schema imports early.** Having all `aind-data-schema` imports in one module (`metadata_utils.py`) makes future schema bumps much cheaper.

### From the motion-correction upgrade:

9. **`DropdownMetric.value` changed from `List[str]` to `str`.** In v1, `value=["My option"]` (list) and `value=[]` (empty list). In v2, `value="My option"` (string) and `value=""` (empty string). The validator rejects lists.

10. **`DataProcess` is a major breaking change, not minor.** The constructor has entirely different fields — `name` → `process_type`, `software_version`/`code_url`/`parameters`/`input_location`/`output_location` are all **removed** (extra forbidden). New required fields: `stage`, `code` (Code object), `experimenters`. Do not assume `DataProcess` is compatible — inspect the v2 source.

11. **Frame rate is on `ImagingConfig.sampling_strategy.frame_rate`, not on `PlanarImage`.** The upgrade guide's traversal pattern for planes is correct, but frame rate lives one level up on the config's `sampling_strategy` object, not on individual images.

12. **`data_description.subject_id` exists** — you may not need to load `subject.json` separately just for the subject ID.

13. **Suite2P mmap blocks temp dir cleanup.** `suite2p.io.BinaryFile` memory-maps `data.bin`. If you don't explicitly `.close()` the BinaryFile before calling `tmp_dir.cleanup()`, the mmap holds the file open and `shutil.rmtree` fails with "Directory not empty". Copy the data and close: `data = bin_file.data.copy(); bin_file.close()`.

14. **Dead code fallbacks may hide bugs.** During the upgrade, audit fallback code paths (e.g. "pull frame rate from platform.json if not in session.json"). If the fallback was never actually reached in production, it may contain bugs (e.g. literal string `"platform_data['sync_file']"` instead of evaluating the dict lookup). Loading metadata via Pydantic models catches these issues at validation time.
