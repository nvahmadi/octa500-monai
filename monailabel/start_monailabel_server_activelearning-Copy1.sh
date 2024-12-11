monailabel start_server \
    --app /data/Projects/octa500-monai/monailabel/apps/octa500_100train_800epochs \
    --studies /data/Projects/datasets/OCTA-500/OCTA500_MONAILABEL_3mm \
    --conf models segmentation \
    --conf use_pretrained_model false \
    --conf skip_scoring false \
    --conf skip_strategies false \
    --conf epistemic_enabled true \
    --conf epistemic_samples 5

