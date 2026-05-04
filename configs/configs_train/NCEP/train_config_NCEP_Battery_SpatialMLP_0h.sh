#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
STATION="Battery"
MODEL="spatial_mlp_0h"
HISTORY_HOURS_LIST=(0)
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
