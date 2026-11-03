"""Thin Code Ocean entry point for Suite2P motion correction.

All logic lives in the ``aind-ophys-motion-correction-library`` package;
this wrapper only parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_motion_correction_library.job import run

if __name__ == "__main__":
    run()
