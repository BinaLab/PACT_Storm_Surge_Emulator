#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_MRI/graphs"
STATION="Battery"
MODEL="perceiver3"
HISTORY_HOURS_LIST=(12)
TRAIN_RATIO="0.4545454545"
VAL_RATIO="0.0909090909"
