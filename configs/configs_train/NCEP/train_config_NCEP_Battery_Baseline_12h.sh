#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="/home/exouser/media/share/PACT/Data/Grid4_New/NCEP/graphs"
STATION="Battery"
MODEL="baseline"
HISTORY_HOURS_LIST=(12)
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
LR_LIST=("3e-3" "5e-3")
