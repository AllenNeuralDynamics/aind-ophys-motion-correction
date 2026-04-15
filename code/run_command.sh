#!/usr/bin/env bash
rm -rf /results/*
rm -r /scratch/*
# bash ./run --input_dir /data/multiplane-ophys_839909_2026-02-26_15-11-01_v2_converted 2>&1 | tee /results/capsule_output.log
bash ./run --input_dir /data/single-plane-ophys_767715_2025-07-25_17-40-22_v2_converted 2>&1 | tee /results/capsule_output.log