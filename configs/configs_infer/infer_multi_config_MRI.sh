#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/infer_config_common.sh"

ROOT_DIR="./Data/Grid4_New/CMIP6_MRI/graphs"
STATION="Battery"
CKPT_PATH="./Inference_Checkpoints/CMIP6_MRI_Battery_P3_Best.pth"
MODEL="perceiver3"
HISTORY_HOURS=12

RUNS=(
  "CMIP6_MRI_Battery_P3_Best_TO_NCEP|./Data/Grid4_New/NCEP/graphs"
  "CMIP6_MRI_Battery_P3_Best_TO_CMIP6_AWI|./Data/Grid4_New/CMIP6_AWI/graphs"
  "CMIP6_MRI_Battery_P3_Best_TO_CMIP6_CNRM|./Data/Grid4_New/CMIP6_CNRM/graphs"
  "CMIP6_MRI_Battery_P3_Best_TO_CMIP6_EC_EARTH|./Data/Grid4_New/CMIP6_EC_EARTH/graphs"
  "CMIP6_MRI_Battery_P3_Best_TO_CMIP6_MPI|./Data/Grid4_New/CMIP6_MPI/graphs"
  "CMIP6_MRI_Battery_P3_Best_TO_CMIP6_MRI|./Data/Grid4_New/CMIP6_MRI/graphs"
)
