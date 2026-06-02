python -m test_cmld \
  --cfg configs/config_cmld_humanml3d.yaml \
  --cfg_assets configs/assets.yaml \
  --is_test true \
  --guidance_mode v4 \
  --is_guidance true \
  "$@"
