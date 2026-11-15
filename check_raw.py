#!/usr/bin/env python
"""Quick pre-flight: which raw files are present, their size/md5/row count, and whether NSL-KDD matches published values.
Does not clean or modify anything.   python check_raw.py"""
from src.utils import load_config
from src.data_loader import load_dataset, available_datasets, DatasetNotFound, DOWNLOAD_HELP
cfg = load_config()
for name, ok in available_datasets(cfg).items():
    print(f"\n== {name}: {'files found' if ok else 'MISSING'}")
    if not ok:
        print("   expected:", DOWNLOAD_HELP[name]); continue
    try:
        df, infos = load_dataset(name, cfg)
        print(f"   {len(df):,} rows x {df.shape[1]} columns (incl. metadata)")
        for i in infos:
            print(f"   {i['file']:<55} {i['size_bytes']/1e6:8.1f} MB {i['rows']:>10,} rows  md5={i['md5']}  {i.get('validation','')}")
    except DatasetNotFound as e:
        print("  ", e)
