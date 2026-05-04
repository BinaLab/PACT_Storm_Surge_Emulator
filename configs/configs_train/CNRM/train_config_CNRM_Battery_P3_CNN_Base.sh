#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="./Data/Grid4_New/CMIP6_CNRM/graphs"
STATION="Battery"
MODEL="perceiver3_cnn"
HISTORY_HOURS_LIST=(12)
TRAIN_RATIO="0.4545454545"
VAL_RATIO="0.0909090909"
