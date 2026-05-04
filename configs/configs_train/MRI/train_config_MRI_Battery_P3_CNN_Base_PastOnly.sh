#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="./Data/Grid4_New_PastOnly/CMIP6_MRI/graphs"
STATION="Battery"
MODEL="perceiver3_cnn"
HISTORY_HOURS_LIST=(12)
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
