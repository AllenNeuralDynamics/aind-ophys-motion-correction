# Bergamo Path — v2 Upgrade Plan

The Bergamo single-plane path in `registration.py` needs additional work beyond the straightforward schema rename. This doc tracks what has changed, what's done, and what remains.

## Completed

### Detection
- `session.get("rig_id", "")` → `acquisition.instrument_id` ✅

### Function Signatures
- `singleplane_motion_correction(h5_file, output_dir, session, unique_id, ...)` → `singleplane_motion_correction(data_dir, output_dir, acquisition, ...)` ✅
- `unique_id` now derived from h5 file stem instead of hardcoded `"MOp2_3_0"` ✅
- `generate_single_plane_reference(fp, session)` → `generate_single_plane_reference(fp, bci_epochs)` ✅

### Non-BCI Singleplane Support
- `singleplane_motion_correction` now checks for BCI epochs first ✅
- If no BCI conditioning epochs found, skips reference image generation and lets Suite2P create its own reference ✅
- BCI epoch filtering moved to `metadata_utils.get_bci_conditioning_epochs()` ✅

## Remaining — BCI-Specific Issues

### 1. `StimulusEpoch.output_parameters.tiff_stem` — v2 structure unknown

The v1 code accesses:
```python
bci_epoch_loc = bci_epochs[0].output_parameters.tiff_stem
```

In v2, `StimulusEpoch` no longer has `output_parameters` in its keys. The v2 epoch keys are:
```
object_type, stimulus_start_time, stimulus_end_time, stimulus_name,
code, stimulus_modalities, performance_metrics, notes, active_devices,
configurations, training_protocol_name, curriculum_status
```

**Action needed:** Find where `tiff_stem` lives in v2, or determine if this data is only in the h5 file's `epoch_locations` dataset (which already has the frame boundaries baked in).

### 2. No v2 BCI test data available

The singleplane test dataset (`single-plane-ophys_767715`) has `spontaneous activity` and `2p photostimulation` epochs, not BCI. Need a v2 Bergamo BCI dataset to validate the full path.

### 3. Frame rate for singleplane

The Bergamo path doesn't extract frame rate from acquisition — it relies on the user-input fallback. With v2, `get_frame_rate_from_acquisition()` should work if the acquisition has `ImagingConfig.sampling_strategy.frame_rate`. Verify this for Bergamo acquisitions.
