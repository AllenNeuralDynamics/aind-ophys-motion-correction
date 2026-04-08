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

**Not every capsule needs this.** If your capsule only reads `instrument_id` or `stimulus_epochs` and doesn't traverse FOV data, you can skip this section. Search your code for `ophys_fovs`, `scanfield_z`, `scanimage_roi_index` to check.

### 2.10 Modality Import Path Change

```python
# v1:
from aind_data_schema.core.quality_control import Modality

# v2:
from aind_data_schema_models.modalities import Modality
```

Common modalities: `Modality.POPHYS`, `Modality.BEHAVIOR`, `Modality.BEHAVIOR_VIDEOS`, `Modality.FIB`.

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

After upgrading, capsules should follow the **metadata-manager pattern**: the capsule is a thin passthrough (~5 lines), all logic lives in the library.

### Capsule (`code/run_capsule.py`)

```python
"""Top-level run script."""
from my_library.job import run

if __name__ == "__main__":
    run()
```

### Library Settings (`settings.py`)

```python
from pathlib import Path
from typing import Optional, Union

from pydantic import Field
from pydantic_settings import BaseSettings


class JobSettings(BaseSettings, cli_parse_args=True):
    input_dir: Union[Path, str] = Field(
        description="directory where input files are found"
    )
    output_dir: Union[Path, str] = Field(
        description="directory where outputs are saved"
    )
    temp_dir: Optional[Path] = Field(
        default=None,
        description="temporary directory for intermediate files",
    )
    debug: bool = Field(
        default=False,
        description="debug mode",
    )
```

`cli_parse_args=True` means the library parses CLI args directly — the capsule doesn't pass anything.

### Library Job (`job.py`)

```python
import json
import logging
from pathlib import Path

from my_library.settings import JobSettings


def run() -> None:
    """Entry point — capsule calls this directly."""
    settings = JobSettings()
    input_dir = Path(settings.input_dir)
    output_dir = Path(settings.output_dir)

    # Discover metadata
    acquisition_fp = next(input_dir.rglob("acquisition.json"))
    data_description_fp = next(input_dir.rglob("data_description.json"))

    # ... set up logging, do work, write QC ...
```

### Centralise Schema Imports

Keep all `aind-data-schema` and `aind-data-schema-models` imports in **one utility module** (e.g., `utils/metadata_utils.py`). This way schema version changes only require edits in one place.

```python
# utils/metadata_utils.py
from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.quality_control import (
    QCMetric, QCStatus, QualityControl, Stage, Status,
)
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import DropdownMetric

def load_acquisition(path):
    return Acquisition.model_validate_json(path.read_text())

def load_data_description(path):
    return DataDescription.model_validate_json(path.read_text())
```

---

## 5. Structured Logging with aind-log-utils

Optional but recommended. Provides structured JSON logs + CloudWatch integration on Code Ocean.

### Quick Integration

```python
# In job.py, at the top of run():
from aind_log_utils.log import setup_logging

setup_logging(
    process_name="my-capsule-name",
    subject_id=subject_id,           # from acquisition or data_description
    asset_name=data_description.name,
    send_start_log=True,
)
```

### What It Does

1. Configures root logger with structured JSON formatter
2. Adds CloudWatch handler when AWS creds are available (Code Ocean)
3. Falls back to console-only when running locally
4. Injects metadata (hostname, capsule_id, subject_id, etc.) into every log record
5. Registers atexit handler for "Stopping" lifecycle logs

### Alternative: Custom YAML-Based Logging

The pophys-converter upgrade used a custom `logging.yml` + `setup_logging()` wrapper instead of `aind-log-utils` directly. Either approach works — the key is structured JSON output with consistent fields. See the pophys-converter `utils/logging.py` for the custom approach.

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
