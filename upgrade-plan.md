# aind-ophys-motion-correction: Schema v1 to v2 Upgrade Plan

**Date:** 2026-04-08
**Branch:** `upgrade-schema-v2`
**Reference implementations:** `scratch/aind-pophys-converter` and `scratch/aind-pophys-converter-capsule` (both on `upgrade-schema-v2`)

---

## Current State

The Dockerfile and `environment.json` have **already been partially updated**:
- `aind-data-schema==2.6.0` (done)
- `aind-data-schema-models==5.4.1` (done)
- `aind-log-utils==0.2.6` (done)

But the Python source code (`registration.py`) still uses **v1 API patterns** everywhere. The capsule will fail at runtime.

A v2 data asset is already attached: `multiplane-ophys_839909_2026-02-26_15-11-01_v2_converted`.

---

## Scope

This is a **Phase A (schema upgrade in prvolnZk3juxH5R3xCCZCfo8nnI3W2f35tvEZg2N4cTZLpeC5#UxupZi7s4gM3XbfF-rkerDFIaWkf4CKBIwCqzgAykiAlace)** — keep the monolith structure of `registration.py`, just make it work with v2. Phase B (capsule/library refactor) and Phase C (structured logging) are out of scope.

---

## Changes

### 1. Dockerfile — Add Missing Dependencies

**File:** `environment/Dockerfile` (line 9-13)
**File:** `.codeocean/environment.json`

The Dockerfile is missing `aind-qcportal-schema` and `pytz`. Both are needed at runtime — `aind-qcportal-schema` for `DropdownMetric` (already imported in `registration.py` line 37), and `pytz` for timezone-aware `QCStatus` timestamps (v2 requirement per guide section 2.5).

**Add to pip install:**
```
aind-qcportal-schema==0.6.4
pytz
```

Also mirror these additions in `.codeocean/environment.json`.

> **Note:** `aind-qcportal-schema` is already imported in the current code (line 37: `from aind_qcportal_schema.metric_value import DropdownMetric`), so it must already be a transitive dependency. But v2 best practice is to pin it explicitly to ensure compatibility.

---

### 2. Imports — Add v2 Schema Types

**File:** `code/registration.py` (lines 26-29)

**Current:**
```python
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_data_schema_models.process_names import ProcessName
```

**Target:**
```python
import pytz
from aind_data_schema.components.configs import ImagingConfig
from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import (
    QCMetric, QCStatus, Stage, Status,
)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName
```

**What's changing:**
- Add `import pytz` near top-level imports
- Add `ImagingConfig` from `aind_data_schema.components.configs` (for v2 frame rate traversal via `sampling_strategy`)
- Add `Acquisition` from `aind_data_schema.core.acquisition` (for Pydantic model loading)
- Add `Stage` from `quality_control` (required on each `QCMetric` in v2)
- Add `Modality` from `aind_data_schema_models.modalities` (moved in v2 — was on `QCEvaluation` in v1)
- `DataProcess`, `ProcessName`, `QCMetric`, `QCStatus`, `Status` — import paths unchanged
- **No `QualityControl` import** — reference converter doesn't use it; individual metrics are aggregated downstream

---

### 3. QC Metric: `serialize_registration_summary_qcmetric()` (lines 528-568)

**What's changing:**
- Add `modality=Modality.POPHYS` to `QCMetric` constructor
- Add `stage=Stage.PROCESSING` to `QCMetric` constructor (this capsule performs motion correction processing, not raw data ingestion)
- Add `tags={"evaluation": "Registration Summary", "type": "Operational QC"}`
- Fix `QCStatus(timestamp=dt.now())` to use timezone-aware datetime: `dt.now(pytz.timezone("America/Los_Angeles")).isoformat()`

**Reference:** `scratch/aind-pophys-converter/src/aind_pophys_converter/utils/metadata_utils.py` lines 135-200 show the exact v2 pattern for building QC metrics with `pending_qc_status()`.

---

### 4. QC Metric: `serialize_fov_quality_qcmetric()` (lines 571-623)

**Same changes as #3:**
- Add `modality=Modality.POPHYS`
- Add `stage=Stage.PROCESSING`
- Add `tags={"evaluation": "FOV Quality", "type": "Operational QC"}`
- Fix `QCStatus` timestamp to be timezone-aware

---

### 5. QC Output — Keep as Individual Serialized Metrics (No Change Needed)

**Current behavior:** Each QC function writes an individual `*_metric.json` file to the plane's output directory. These are standalone `QCMetric` JSON files.

**Decision:** Keep this pattern. Just like the pophys-converter reference, individual metric JSONs will be aggregated by a downstream process. No `quality_control.json` consolidation needed here. The only changes to QC output are the field additions in #3 and #4 above.

---

### 6. `session.json` to `acquisition.json` — Main Block (lines 2070-2074)

**Current:**
```python
session_fp = next(data_dir.rglob("session.json"))
# ...
with open(session_fp, "r") as j:
    session = json.load(j)
```

**Target:**
```python
acquisition_fp = next(data_dir.rglob("acquisition.json"))
acquisition = Acquisition.model_validate_json(acquisition_fp.read_text())
```

Load as a Pydantic `Acquisition` model (not a raw dict), following the reference converter pattern. This means all downstream access changes from dict-style (`session["rig_id"]`) to attribute-style (`acquisition.instrument_id`). Downstream functions that currently accept a `session: dict` parameter will need their type hints and access patterns updated accordingly.

---

### 7. `rig_id` to `instrument_id` — Bergamo Detection (line 2097)

**Current:**
```python
if "Bergamo" in session.get("rig_id", ""):
```

**Target:**
```python
if "Bergamo" in acquisition.instrument_id:
```

Direct attribute access — `instrument_id` is a required `str` field on `Acquisition` (never None), matching the reference converter pattern.

---

### 8. `session.json` to `acquisition.json` — `multiplane_motion_correction()` (lines 1801-1805)

**Current:**
```python
session_fp = next(data_dir.rglob("session.json"), "")
if not session_fp:
    raise f"Could not locate session.json in {session_fp}"
with open(session_fp) as f:
    session_data = json.load(f)
```

**Target:**
```python
acquisition_fp = next(data_dir.rglob("acquisition.json"), "")
if not acquisition_fp:
    raise ValueError(f"Could not locate acquisition.json in {data_dir}")
acquisition_data = Acquisition.model_validate_json(acquisition_fp.read_text())
```

Load as Pydantic model. Also fix the bare `raise` string (current code raises a string literal, not an exception). Downstream access changes from dict-style to attribute-style.

---

### 9. `ophys_fovs` Frame Rate — `multiplane_motion_correction()` (lines 1808-1819)

**Current:**
```python
try:
    frame_rate_hz = float(
        session_data["data_streams"][0]["ophys_fovs"][0]["frame_rate"]
    )
except KeyError:
    # fallback to platform.json
```

**Target:** The `ophys_fovs` structure is **gone** in v2. Since we're now loading as a Pydantic `Acquisition` model, traverse the v2 ImagingConfig hierarchy using the same pattern as the reference converter:

**Confirmed from v2 schema inspection:** `frame_rate` lives on `ImagingConfig.sampling_strategy.frame_rate` (via `SamplingStrategy`), NOT on `PlanarImage`.

```python
try:
    frame_rate_hz = None
    for ds in acquisition_data.data_streams:
        for config in ds.configurations:
            if isinstance(config, ImagingConfig) and config.sampling_strategy:
                frame_rate_hz = float(config.sampling_strategy.frame_rate)
                break
    if not frame_rate_hz:
        raise KeyError("frame_rate not found in acquisition")
except (KeyError, AttributeError):
    # fallback to platform.json (existing logic)
```

Keep the platform.json fallback.

---

### 10. `ophys_fovs` Frame Rate — `get_frame_rate()` (lines 2036-2057)

**Current:**
```python
def get_frame_rate(session: dict):
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
```

**Target:** Rewrite to traverse v2 `Acquisition` model. **Confirmed:** `frame_rate` is on `ImagingConfig.sampling_strategy.frame_rate`.

```python
def get_frame_rate(acquisition: Acquisition):
    """Attempt to pull frame rate from acquisition.json (v2 schema)."""
    for ds in acquisition.data_streams:
        for config in ds.configurations:
            if isinstance(config, ImagingConfig) and config.sampling_strategy:
                return float(config.sampling_strategy.frame_rate)
    return None
```

Rename parameter `session` to `acquisition`, change type hint from `dict` to `Acquisition`, update docstring. Update call site at line 2084.

---

### 11. `stimulus_epochs` — `generate_single_plane_reference()` (lines 1945, 1962-1967)

**Current:**
```python
def generate_single_plane_reference(fp: Path, session) -> Path:
    # ...
    for i in session["stimulus_epochs"]
        if i["stimulus_name"] == "single neuron BCI conditioning"
```

**Target:** Rename parameter `session` to `acquisition` (now an `Acquisition` model, not a dict). Access via Pydantic attributes.

**Confirmed from v2 schema inspection:** `StimulusEpoch` still has `stimulus_name` (str), but `output_parameters` is **gone**. The v2 model has a `notes` field (Optional[str]) where custom data like `output_parameters` would be JSON-encoded (confirmed by reference converter pattern).

```python
def generate_single_plane_reference(fp: Path, acquisition: Acquisition) -> Path:
    # ...
    for epoch in acquisition.stimulus_epochs:
        if epoch.stimulus_name == "single neuron BCI conditioning":
            output_parameters = json.loads(epoch.notes or "{}")
            tiff_stem = output_parameters.get("tiff_stem")
```

**v2 `StimulusEpoch` fields:** `stimulus_start_time`, `stimulus_end_time`, `stimulus_name`, `code`, `stimulus_modalities`, `performance_metrics`, `notes`, `active_devices`, `configurations`, `training_protocol_name`, `curriculum_status`.

---

### 12. `singleplane_motion_correction()` — Parameter Rename (line 1983-1984)

**Current:**
```python
def singleplane_motion_correction(
    h5_file: Path, output_dir: Path, session, unique_id: str, debug: bool = False
):
```

**Target:** Rename `session` parameter to `acquisition`, update type hint to `Acquisition`. Update all internal references to use attribute access.

Also update the call site at line 2098-2099:
```python
h5_file, output_dir, reference_image_fp = singleplane_motion_correction(
    data_dir, output_dir, acquisition, unique_id, debug=parser.debug
)
```

---

### 13. `DataProcess` / `ProcessName` — Verify Compatibility (lines 1467-1509)

**Current:**
```python
data_proc = DataProcess(
    name=ProcessName.VIDEO_MOTION_CORRECTION,
    software_version=os.getenv("VERSION", ""),
    start_date_time=start_time.isoformat(),
    end_date_time=end_time.isoformat(),
    input_location=str(raw_movie),
    output_location=str(motion_corrected_movie),
    code_url="...",
    parameters=metadata,
)
```

**Action:** Verify that `DataProcess` constructor signature and `ProcessName.VIDEO_MOTION_CORRECTION` still exist in v2. Based on the upgrade guide, `DataProcess` is likely unchanged. This should be a **no-change** item, but needs verification.

---

## Change Summary Table

| # | File | Location | Change | Risk |
|---|------|----------|--------|------|
| 1 | `Dockerfile` + `environment.json` | pip deps | Add `aind-qcportal-schema`, `pytz` | Low |
| 2 | `registration.py` | Lines 26-29 | Add `pytz`, `Acquisition`, `ImagingConfig`, `Stage`, `Modality` imports | Low |
| 3 | `registration.py` | Lines 528-568 | Add modality/stage/tags to registration summary QCMetric, fix timestamp | Low |
| 4 | `registration.py` | Lines 571-623 | Add modality/stage/tags to FOV quality QCMetric, fix timestamp | Low |
| 5 | `registration.py` | N/A | No change — keep individual metric JSONs, aggregated downstream | None |
| 6 | `registration.py` | Lines 2070-2074 | `session.json` to `acquisition.json`, rename vars | Low |
| 7 | `registration.py` | Line 2097 | `rig_id` to `instrument_id` | Low |
| 8 | `registration.py` | Lines 1801-1805 | `session.json` to `acquisition.json` in multiplane func | Low |
| 9 | `registration.py` | Lines 1808-1819 | Rewrite `ophys_fovs` frame rate for v2 structure | High |
| 10 | `registration.py` | Lines 2036-2057 | Rewrite `get_frame_rate()` for v2 structure | High |
| 11 | `registration.py` | Lines 1962-1967 | Rename `session` param, verify `stimulus_epochs` | Medium |
| 12 | `registration.py` | Lines 1983-1984 | Rename `session` param to `acquisition` | Low |
| 13 | `registration.py` | Lines 1467-1509 | Verify `DataProcess` compatibility (likely no change) | Low |

---

## Execution Order

1. **Dependencies** (#1) — Dockerfile + environment.json
2. **Imports** (#2) — Add new imports
3. **QC functions** (#3, #4) — Update both QC metric builders
4. **File renames + field renames** (#6, #7, #8, #12) — All the straightforward session-to-acquisition changes
5. **Frame rate extraction** (#9, #10) — The trickiest part; needs v2 acquisition.json inspection
6. **Stimulus epochs** (#11) — Verify and update Bergamo path
7. **Verification item** (#13) — DataProcess compatibility
8. **Test on Code Ocean** — Run against `multiplane-ophys_839909_2026-02-26_15-11-01_v2_converted`

---

## Resolved Questions (from v2 schema inspection)

1. **Frame rate field path:** `ImagingConfig.sampling_strategy.frame_rate` (via `SamplingStrategy` model). Not on `PlanarImage`.

2. **`stimulus_epochs` structure:** `StimulusEpoch.stimulus_name` still exists. `output_parameters` is **gone** — replaced by `notes` (Optional[str], JSON-encoded). Matches reference converter pattern.

3. **`instrument_id`:** Required `str` field on `Acquisition` (never None). Direct attribute access, no null guard needed.

## Verified Against Real v2 Data

Both data assets validated successfully with `Acquisition.model_validate_json()`:

**Multiplane** (`multiplane-ophys_839909_2026-02-26_15-11-01_v2`):
- `instrument_id`: `"422_MESO2_20260122"` (not Bergamo → multiplane path)
- `sampling_strategy.frame_rate`: `10.71` Hz
- `stimulus_epochs`: 8 epochs (no BCI conditioning — expected for mesoscope)

**Single-plane Bergamo** (`single-plane-ophys_767715_2025-07-25_17-40-22_v2`):
- `instrument_id`: `"442_Bergamo_2p_photostim"` (Bergamo detected)
- `sampling_strategy.frame_rate`: `52.1304` Hz (note: first ImagingConfig has `sampling_strategy=None`, others populated — our guard handles this)
- `stimulus_epochs`: 3 epochs, no "single neuron BCI conditioning" (different Bergamo experiment type), `notes` are simple strings not JSON-encoded `output_parameters`
- This is raw TIFF data (not converted h5) — tests the `data_type == "TIFF"` path

## Remaining Risk

- **No BCI conditioning test data** — Cannot test the `generate_single_plane_reference()` path that looks for `stimulus_name == "single neuron BCI conditioning"` and reads `output_parameters.tiff_stem`. Need a BCI Bergamo v2 data asset to verify whether `output_parameters` moved to `notes` as JSON.

