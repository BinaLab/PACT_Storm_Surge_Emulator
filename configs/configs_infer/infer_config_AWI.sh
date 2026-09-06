#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/infer_config_common.sh"

NAME="CMIP6_AWI_Battery_P3_Best_TO_CMIP6_EC_EARTH"
ROOT_DIR="./Data/Grid4_New/CMIP6_AWI/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/CMIP6_EC_EARTH/graphs"
STATION="Battery"

CKPT_PATH="./Inference_Checkpoints/CMIP6_AWI_Battery_P3_Best.pth"
MODEL="perceiver3"
HISTORY_HOURS=12
