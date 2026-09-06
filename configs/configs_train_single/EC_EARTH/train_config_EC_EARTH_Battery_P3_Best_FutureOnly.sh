#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_common.sh"
ROOT_DIR="/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_EC_EARTH/graphs"
STATION="Battery"
MODEL="perceiver3"
HISTORY_HOURS_LIST=(12)
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
SHUFFLE_YEARS="0"
FUTURE_ONLY="1"
FUTURE_YEAR_THRESHOLD="2030"
LOSS_MODE_LIST=("mse_tail_slope")
