from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv

make_datasplit_csv(
    classes=["endo"],
    validation_prob=0.15,  # 15% for validation
    datasets=["jrc_cos7-1a", "jrc_jurkat-1", "jrc_mus-liver", "jrc_fly-mb-1a"],
    crops=["*"],
    csv_path="endo_datasplit_new.csv",
    dry_run=False
)
