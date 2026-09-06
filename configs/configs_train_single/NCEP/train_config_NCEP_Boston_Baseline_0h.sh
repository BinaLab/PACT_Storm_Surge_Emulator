#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="/home/exouser/media/share/PACT/Data/Grid4_New/NCEP/graphs"
STATION="Boston"
MODEL="baseline"
HISTORY_HOURS_LIST=(0)
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
