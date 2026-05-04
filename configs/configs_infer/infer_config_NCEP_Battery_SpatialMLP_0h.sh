#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/infer_config_common.sh"

NAME="NCEP_Battery_SpatialMLP_0h_TO_NCEP"
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
STATION="Battery"

CKPT_PATH="./checkpoints_Battery/best_spatial_mlp_0h_*.pth"
MODEL="spatial_mlp_0h"
HISTORY_HOURS=0
