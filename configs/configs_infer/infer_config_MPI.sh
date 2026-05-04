#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/infer_config_common.sh"

NAME="CMIP6_MPI_Battery_P3_Best_TO_CMIP6_MPI"
ROOT_DIR="./Data/Grid4_New/CMIP6_MPI/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/CMIP6_MPI/graphs"
STATION="Battery"

CKPT_PATH="./Inference_Checkpoints/CMIP6_MPI_Battery_P3_Best.pth"
MODEL="perceiver3"
HISTORY_HOURS=12
