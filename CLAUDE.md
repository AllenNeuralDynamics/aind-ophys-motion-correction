# CLAUDE.md — aind-ophys-motion-correction Schema Upgrade

## Environment

You are working on a **Code Ocean capsule** for Suite2p-based motion correction of optical physiology data. The capsule has no companion library — all logic lives in `code/registration.py` (~2,640 lines).

On Code Ocean, real data is mounted at `/data/` (read-only) and outputs go to `/results/`. Locally, there is no data — the user will run data-dependent tests on Code Ocean.

## Task

Upgrade `aind-ophys-motion-correction` from `aind-data-schema==1.1.0` to `aind-data-schema>=2.6.0`.

**Reference guide:** `schema-v1-to-v2-upgrade-guide.md` (repo root) — read this first for the full breaking changes list, v2 QC API reference, and execution playbook.

## Capsule Layout

```
aind-ophys-motion-correction/
├── code/
│   ├── run                       # Bash entry point: python registration.py "$@"
│   ├── registration.py           # ALL capsule logic (~2,640 lines) — PRIMARY EDIT TARGET
│   └── sync_dataset.py           # NI-DAQ sync file reader (no schema deps, no changes needed)
├── environment/
│   ├── Dockerfile                # Pins aind-data-schema==1.1.0 — MUST UPDATE
│   └── environment.yml           # AWS creds config (no changes needed)
├── .codeocean/
│   ├── datasets.json             # Attached data assets
│   └── resources.json            # Compute resources
├── schema-v1-to-v2-upgrade-guide.md  # Generic v1→v2 upgrade reference
├── metadata/
│   └── metadata.yml              # Code Ocean metadata
└── resources/                    # Reference images for README
```

## Coding Style

- Type hints where present (not comprehensive in this codebase)
- Black formatting (no explicit config — use defaults)
- Conventional Commits: `feat(scope):`, `fix(scope):`, `chore(scope):`

## Guidelines

- Read ALL context before making changes — `registration.py` is a large monolith
- Never push without explicit user approval
- User runs data-dependent tests on Code Ocean — focus on correctness of code changes locally

---

## What Needs to Change

### Overview

This capsule has **three categories** of schema-dependent code:

1. **QC metrics** — Two functions write `QCMetric` objects (v2 requires new fields)
2. **Session.json reads** — Multiple locations read `session.json` as raw JSON using v1 field names
3. **DataProcess** — One function writes a `DataProcess` object (likely compatible, verify)

The capsule does **not** use `QCEvaluation` — it writes raw `QCMetric` objects to individual JSON files. In v2, `QCMetric` gains required `modality` and `stage` fields, and `QCStatus.timestamp` requires timezone-aware datetimes.

### Dependency Bump (Dockerfile)

**File:** `environment/Dockerfile`

Current (line 21):
```
aind-data-schema==1.1.0
```

Target:
```
aind-data-schema==2.6.0
aind-data-schema-models>=5.4.1,<6
aind-qcportal-schema>=0.6.4
pytz
```

Note: `aind-data-schema-models` is not currently in the Dockerfile — it needs to be **added** (it was an implicit transitive dep in v1, now needs explicit pinning). `pytz` is needed for timezone-aware QCStatus timestamps.

### Change Map

#### 1. Imports (lines 26-28)

```python
# CURRENT:
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_data_schema_models.process_names import ProcessName

# NEEDED — add Stage, Modality, QualityControl:
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import (
    QCMetric, QCStatus, QualityControl, Stage, Status,
)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName
```

Also add `import pytz` near the top for timezone-aware timestamps.

#### 2. QC Metric: Registration Summary (lines 528-568)

`serialize_registration_summary_qcmetric()` — needs:
- Add `modality=Modality.POPHYS` and `stage=Stage.PROCESSING` to `QCMetric`
- Add `tags={"evaluation": "Registration Summary", "type": "Operational QC"}`
- Fix `QCStatus(timestamp=dt.now())` → use timezone-aware datetime
- The function currently writes individual metric JSON files — decide whether to keep this pattern or collect metrics for a single `quality_control.json`

#### 3. QC Metric: FOV Quality (lines 571-623)

`serialize_fov_quality_qcmetric()` — same changes as above:
- Add `modality=Modality.POPHYS` and `stage=Stage.PROCESSING`
- Add `tags={"evaluation": "FOV Quality", "type": "Operational QC"}`
- Fix `QCStatus` timestamp

#### 4. DataProcess (lines 1467-1509) — MAJOR BREAKING CHANGE

`DataProcess` constructor changed significantly in v2. This was flagged as "likely compatible" but is NOT.

**v2 required new fields:** `process_type` (was `name`), `stage` (`ProcessStage.PROCESSING`), `code` (`Code` object), `experimenters` (list)
**v2 removed fields:** `software_version`, `input_location`, `output_location`, `code_url`, `parameters` — all now **extra forbidden**
**v2 moved:** `software_version` → `Code.version`, `code_url` → `Code.url`, `parameters` → `Code.parameters`
**v2 timestamps:** `start_date_time` now requires timezone-aware datetime

New imports needed:
```python
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessStage
```

#### 5. session.json → acquisition.json (MULTIPLE LOCATIONS)

**Main block (line 2070):**
```python
session_fp = next(data_dir.rglob("session.json"))
```
→ Change to `acquisition.json`, rename variable to `acquisition_fp`

**Main block (line 2073-2074):**
```python
with open(session_fp, "r") as j:
    session = json.load(j)
```
→ Rename to `acquisition`

**Bergamo detection (line 2097):**
```python
if "Bergamo" in session.get("rig_id", ""):
```
→ Change to `acquisition.get("instrument_id", "")`

**multiplane_motion_correction() (line 1801-1805):**
```python
session_fp = next(data_dir.rglob("session.json"), "")
if not session_fp:
    raise f"Could not locate session.json in {session_fp}"
with open(session_fp) as f:
    session_data = json.load(f)
```
→ Change to `acquisition.json` / `acquisition_data`

#### 6. ophys_fovs Traversal (CRITICAL — structure changed in v2)

**multiplane_motion_correction() (line 1808-1809):**
```python
frame_rate_hz = float(
    session_data["data_streams"][0]["ophys_fovs"][0]["frame_rate"]
)
```
In v2, `ophys_fovs` is gone. Frame rate lives in the ImagingConfig hierarchy. See `schema-v1-to-v2-upgrade-guide.md` §2.9.

**Decision needed:** Either traverse the v2 `ImagingConfig → PlanarImage → Plane` structure, or fall back to `platform.json` for frame rate (the fallback already exists at lines 1811-1819).

**get_frame_rate() (lines 2036-2057):**
```python
for i in session.get("data_streams", ""):
    if i.get("ophys_fovs", ""):
        frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
```
Same issue — `ophys_fovs` doesn't exist in v2.

#### 7. stimulus_epochs (Bergamo single-plane path)

**generate_single_plane_reference() (line 1964):**
```python
for i in session["stimulus_epochs"]
    if i["stimulus_name"] == "single neuron BCI conditioning"
```
`stimulus_epochs` still exists on `Acquisition` in v2, but the internal structure may have changed. Verify the `StimulusEpoch` model's fields (`stimulus_name`, `output_parameters.tiff_stem`).

#### 8. setup_logging() call (line 2081-2083)

```python
setup_logging(
    "aind-ophys-motion-correction", mouse_id=subject_id, session_name=name
)
```
This already uses `aind-log-utils`. Verify the kwargs still work with the current version. Consider if any kwargs need renaming.

---

## Execution Strategy

Follow the phased approach from the upgrade guide. For this capsule, Phase B (library refactor) is **out of scope** since this is a pure capsule with no companion library.

### Phase A: Schema Upgrade

1. Bump deps in `Dockerfile`
2. Add `import pytz` and update schema imports (add `Stage`, `Modality`, `QualityControl`)
3. Fix both QC functions — add `modality`, `stage`, `tags`; fix timezone on `QCStatus`
4. Fix all `session.json` → `acquisition.json` references (5 locations)
5. Fix `rig_id` → `instrument_id` (1 location)
6. Fix `ophys_fovs` frame rate extraction (2 locations) — either rewrite for v2 structure or rely on platform.json fallback
7. Verify `stimulus_epochs` access still works for Bergamo path
8. Verify `DataProcess` / `ProcessName` compatibility
9. **Test on Code Ocean** against a v2 data asset

### Phase B: QC Output Consolidation (optional)

Currently the capsule writes individual `*_metric.json` files. Consider consolidating into a single `quality_control.json` using `QualityControl(metrics=[...], default_grouping=[...])`. This matches the v2 portal's expected format.

### Phase C: Structured Logging Verification

Already uses `aind-log-utils`. Just verify compatibility with the latest version after the dep bump.

---

## Session.json Field Access Inventory

| Location | Line(s) | v1 Access Pattern | v2 Equivalent | Notes |
|---|---|---|---|---|
| Main block | 2070 | `rglob("session.json")` | `rglob("acquisition.json")` | File rename |
| Main block | 2097 | `session.get("rig_id", "")` | `acquisition.get("instrument_id", "")` | Bergamo detection |
| `multiplane_motion_correction` | 1801 | `rglob("session.json")` | `rglob("acquisition.json")` | File rename |
| `multiplane_motion_correction` | 1809 | `data_streams[0].ophys_fovs[0].frame_rate` | v2 ImagingConfig traversal or platform.json fallback | **Structure gone** |
| `get_frame_rate` | 2051-2053 | `data_streams[].ophys_fovs[0].frame_rate` | Same as above | **Structure gone** |
| `generate_single_plane_reference` | 1964 | `session["stimulus_epochs"]` | `acquisition["stimulus_epochs"]` | Verify field names |
| `singleplane_motion_correction` | 1983 | Receives `session` dict | Rename param to `acquisition` | Pass-through |

## QC Metric Inventory

| Function | Line(s) | Metric Name | Missing v2 Fields |
|---|---|---|---|
| `serialize_registration_summary_qcmetric` | 528-568 | `{id} Registration Summary` | `modality`, `stage`, `tags` |
| `serialize_fov_quality_qcmetric` | 571-623 | `{id} FOV Quality` | `modality`, `stage`, `tags` |

Both also need `QCStatus.timestamp` fixed for timezone awareness.
