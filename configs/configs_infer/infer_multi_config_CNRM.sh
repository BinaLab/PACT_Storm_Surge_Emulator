#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/infer_config_common.sh"

ROOT_DIR="/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_CNRM/graphs"
STATION="Battery"
CKPT_PATH="./Inference_Checkpoints/CMIP6_CNRM_Battery_P3_Best.pth"
MODEL="perceiver3"
HISTORY_HOURS=12

RUNS=(
  "CMIP6_CNRM_Battery_P3_Best_TO_NCEP|/home/exouser/media/share/PACT/Data/Grid4_New/NCEP/graphs"
  "CMIP6_CNRM_Battery_P3_Best_TO_CMIP6_AWI|/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_AWI/graphs"
  "CMIP6_CNRM_Battery_P3_Best_TO_CMIP6_CNRM|/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_CNRM/graphs"
  "CMIP6_CNRM_Battery_P3_Best_TO_CMIP6_EC_EARTH|/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_EC_EARTH/graphs"
  "CMIP6_CNRM_Battery_P3_Best_TO_CMIP6_MPI|/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_MPI/graphs"
  "CMIP6_CNRM_Battery_P3_Best_TO_CMIP6_MRI|/home/exouser/media/share/PACT/Data/Grid4_New/CMIP6_MRI/graphs"
)
