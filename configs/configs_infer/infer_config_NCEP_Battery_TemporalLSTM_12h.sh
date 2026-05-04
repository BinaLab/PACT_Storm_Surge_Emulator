#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/infer_config_common.sh"

NAME="NCEP_Battery_TemporalLSTM_12h_TO_NCEP"
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
STATION="Battery"

CKPT_PATH="./checkpoints_Battery/best_temporal_lstm_12h_*.pth"
MODEL="temporal_lstm_12h"
HISTORY_HOURS=12
