# Member 1 - Dataset Preparation & Feature Engineering Report

*Project: An Enhanced Intrusion Detection System for SDN Using Hybrid Neural Model and Ensemble Learning.*

> Every number in this report was read from files written by `run_pipeline.py`. Sections for datasets that have not been processed are marked **PENDING**.

Datasets processed end-to-end: **NSL-KDD, CICIDS2017, InSDN**. Not processed: **none**.

## 0. Specification conflicts (see also `Spec_Conflicts.md`)

The manuscript describes a CNN-RNN/LSTM hybrid with SMOTE and a train/test split, whereas the implementation brief specifies MLP + RF + XGBoost + Logistic-Regression stacking, SMOTE-NC and a 70:15:15 split. This pipeline implements the brief and exports 2-D tabular data that a CNN/LSTM can reshape later. The manuscript's Algorithm (Steps 2-5) normalises and applies SMOTE *before* splitting, which leaks information; this pipeline splits first.

## 1. Dataset descriptions

**NSL-KDD.** NSL-KDD (Tavallaee et al.): a de-duplicated revision of KDD'99 with 41 connection-level features and attack classes grouped as DoS, Probe, R2L and U2R. Dated (1998-era traffic) and not SDN-based.

**CICIDS2017.** Canadian Institute for Cybersecurity IDS 2017: five days of emulated enterprise traffic converted to bidirectional flows by CICFlowMeter, with labelled benign traffic and DoS/DDoS, brute-force, web, botnet, infiltration and port-scan attacks. Not captured in an SDN environment.

**InSDN.** InSDN (Elsayed et al.): traffic generated in an SDN testbed (OpenFlow/Open vSwitch with an SDN controller) with attacks launched against Metasploitable-2 and OVS targets. The only one of the three datasets collected in an SDN setting.

**NSL-KDD split protocol.** The multiclass split uses KDDTrain+ only (125,973 rows). KDDTest+ is a separate external binary test set (22,454 rows after duplicate removal, read from the generated external-test labels).

## 2. Dataset statistics (source row counts; NSL-KDD missing and duplicate counts are split by official source)

KDDTest+ lacks the difficulty column; missing-cell counts cover only features retained after cleaning.

| Dataset / source | Records | Features | Classes | Missing cells | Inf cells | Exact duplicates removed | Feature-duplicate rows | Conflicting duplicate groups |
|---|---|---|---|---|---|---|---|---|
| KDDTrain+ | 125,973 | 40 | 23 | 0 | 0 | 0 | 31 | 15 |
| KDDTest+ | 22,544 | 40 | 2 | 0 | 0 | 90 | 88 | — |
| CICIDS2017 | 3,119,345 | 84 | 15 | 24,243,926 | 4,376 | 288,804 | — | 0 |
| InSDN | 343,889 | 83 | 8 | 0 | 0 | 1 | — | 0 |

### Source files and validation

**NSL-KDD**

| file | format | size_bytes | rows | md5 |
|---|---|---|---|---|
| KDDTrain+.txt | txt | 14,738,831 | 125,973 | 120b13c9092ab72d4902f2079143cc2c |
| KDDTest+.txt | txt | 2,573,994 | 22,544 | 1e53a506c102628114ca1b1fa509b6a8 |

Validation against published reference values: all checks passed.

**CICIDS2017**

| file | format | size_bytes | rows | md5 |
|---|---|---|---|---|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | csv | 96,101,069 | 225,745 | b0916c20f14561d66e84afe33b4ece23 |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | csv | 101,874,777 | 286,467 | f7c179333baebe2bdeb772c334cb6dd7 |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | csv | 75,386,737 | 191,033 | c5bf0fdb609d9d8db3044f00e8d4d008 |
| Monday-WorkingHours.pcap_ISCX.csv | csv | 268,649,908 | 529,918 | 906f82030bfb3a9c45ef7a2f3b1cd365 |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | csv | 108,723,183 | 288,602 | 040030306fbd691336726a0d769654e4 |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | csv | 92,030,223 | 458,968 | 2c3e5dc6d3395266543d40ba0d15f29e |
| Tuesday-WorkingHours.pcap_ISCX.csv | csv | 174,696,560 | 445,909 | 149fca427503e8f04fe22fc75bb33234 |
| Wednesday-workingHours.pcap_ISCX.csv | csv | 285,642,925 | 692,703 | 0974e0e4eccebeb0a7cbcad9ae5d6550 |

Validation: file count matches expected; no published row-count/checksum reference available.

**InSDN**

| file | format | size_bytes | rows | md5 |
|---|---|---|---|---|
| metasploitable-2.csv | csv | 52,789,315 | 136,743 | 7a8ca7628b265d5d2495c7ad833dc459 |
| Normal_data.csv | csv | 33,608,939 | 68,424 | 90c839f64e9057bc46d51f449e45aeb2 |
| OVS.csv | csv | 55,312,637 | 138,722 | 7fd49cf4f18637a9a2d8c8e9b2858ecc |

Validation: file count matches expected; no published row-count/checksum reference available.

## 3. Original attack categories and record counts

### NSL-KDD

| class | count | percent | ratio_to_majority |
|---|---|---|---|
| normal | 77,054 | 51.88 | 1 |
| neptune | 41,214 | 27.75 | 1.87 |
| attack | 12,833 | 8.641 | 6.004 |
| satan | 3,633 | 2.446 | 21.21 |
| ipsweep | 3,599 | 2.423 | 21.41 |
| portsweep | 2,931 | 1.974 | 26.29 |
| smurf | 2,646 | 1.782 | 29.12 |
| nmap | 1,493 | 1.005 | 51.61 |
| back | 956 | 0.6437 | 80.6 |
| teardrop | 892 | 0.6006 | 86.38 |
| warezclient | 890 | 0.5993 | 86.58 |
| pod | 201 | 0.1353 | 383.4 |
| guess_passwd | 53 | 0.03569 | 1,454 |
| buffer_overflow | 30 | 0.0202 | 2,568 |
| warezmaster | 20 | 0.01347 | 3,853 |
| land | 18 | 0.01212 | 4,281 |
| imap | 11 | 0.007407 | 7,005 |
| rootkit | 10 | 0.006733 | 7,705 |
| loadmodule | 9 | 0.00606 | 8,562 |
| ftp_write | 8 | 0.005387 | 9,632 |
| multihop | 7 | 0.004713 | 1.101e+04 |
| phf | 4 | 0.002693 | 1.926e+04 |
| perl | 3 | 0.00202 | 2.568e+04 |
| spy | 2 | 0.001347 | 3.853e+04 |

### CICIDS2017

| class | count | percent | ratio_to_majority |
|---|---|---|---|
| BENIGN | 2,273,097 | 72.87 | 1 |
| DoS Hulk | 231,073 | 7.408 | 9.837 |
| PortScan | 158,930 | 5.095 | 14.3 |
| DDoS | 128,027 | 4.104 | 17.75 |
| DoS GoldenEye | 10,293 | 0.33 | 220.8 |
| FTP-Patator | 7,938 | 0.2545 | 286.4 |
| SSH-Patator | 5,897 | 0.189 | 385.5 |
| DoS slowloris | 5,796 | 0.1858 | 392.2 |
| DoS Slowhttptest | 5,499 | 0.1763 | 413.4 |
| Bot | 1,966 | 0.06303 | 1,156 |
| Web Attack - Brute Force | 1,507 | 0.04831 | 1,508 |
| Web Attack - XSS | 652 | 0.0209 | 3,486 |
| Infiltration | 36 | 0.001154 | 6.314e+04 |
| Web Attack - Sql Injection | 21 | 0.0006732 | 1.082e+05 |
| Heartbleed | 11 | 0.0003526 | 2.066e+05 |

### InSDN

| class | count | percent | ratio_to_majority |
|---|---|---|---|
| DDoS | 121,942 | 35.46 | 1 |
| Probe | 98,129 | 28.54 | 1.243 |
| Normal | 68,424 | 19.9 | 1.782 |
| DoS | 53,616 | 15.59 | 2.274 |
| BFA | 1,405 | 0.4086 | 86.79 |
| Web-Attack | 192 | 0.05583 | 635.1 |
| BOTNET | 164 | 0.04769 | 743.5 |
| U2R | 17 | 0.004943 | 7,173 |

## 4. Label harmonisation

Full table: `data/metadata/label_mapping.csv`. Labels that are NOT confidently harmonised (UNMAPPED, medium or unknown confidence): `outputs/statistics/labels_not_confidently_harmonized.csv`.

| dataset | original_label | harmonized_label | confidence |
|---|---|---|---|
| NSL-KDD | attack | UNMAPPED | unknown |
| NSL-KDD | back | DoS | high |
| NSL-KDD | buffer_overflow | UNMAPPED | none |
| NSL-KDD | ftp_write | UNMAPPED | none |
| NSL-KDD | guess_passwd | BruteForce | medium |
| NSL-KDD | imap | UNMAPPED | none |
| NSL-KDD | ipsweep | Probe | high |
| NSL-KDD | land | DoS | high |
| NSL-KDD | loadmodule | UNMAPPED | none |
| NSL-KDD | multihop | UNMAPPED | none |
| NSL-KDD | neptune | DoS | high |
| NSL-KDD | nmap | Probe | high |
| NSL-KDD | normal | Benign | high |
| NSL-KDD | perl | UNMAPPED | none |
| NSL-KDD | phf | UNMAPPED | none |
| NSL-KDD | pod | DoS | high |
| NSL-KDD | portsweep | Probe | high |
| NSL-KDD | rootkit | UNMAPPED | none |
| NSL-KDD | satan | Probe | high |
| NSL-KDD | smurf | DoS | high |
| NSL-KDD | spy | UNMAPPED | none |
| NSL-KDD | teardrop | DoS | high |
| NSL-KDD | warezclient | UNMAPPED | none |
| NSL-KDD | warezmaster | UNMAPPED | none |
| CICIDS2017 | BENIGN | Benign | high |
| CICIDS2017 | Bot | Botnet | high |
| CICIDS2017 | DDoS | DDoS | high |
| CICIDS2017 | DoS GoldenEye | DoS | high |
| CICIDS2017 | DoS Hulk | DoS | high |
| CICIDS2017 | DoS Slowhttptest | DoS | high |
| CICIDS2017 | DoS slowloris | DoS | high |
| CICIDS2017 | FTP-Patator | BruteForce | high |
| CICIDS2017 | Heartbleed | UNMAPPED | none |
| CICIDS2017 | Infiltration | Infiltration | high |
| CICIDS2017 | PortScan | Probe | high |
| CICIDS2017 | SSH-Patator | BruteForce | high |
| CICIDS2017 | Web Attack - Brute Force | WebAttack | medium |
| CICIDS2017 | Web Attack - Sql Injection | WebAttack | high |
| CICIDS2017 | Web Attack - XSS | WebAttack | high |
| InSDN | BFA | BruteForce | high |
| InSDN | BOTNET | Botnet | high |
| InSDN | DDoS | DDoS | high |
| InSDN | DoS | DoS | high |
| InSDN | Normal | Benign | high |
| InSDN | Probe | Probe | high |
| InSDN | U2R | UNMAPPED | none |
| InSDN | Web-Attack | WebAttack | high |

Unmapped labels are retained for per-dataset experiments and excluded only from cross-dataset (harmonised) experiments. No attack types were merged merely to align datasets.

For NSL-KDD, the label `attack` is a coarse binary label from KDDTest+; it is not a multiclass attack subtype and is therefore not harmonised.

## 5. Data cleaning

### NSL-KDD

Rows 148,517 -> 148,427 (0.06% removed). Removed rows: {"no_label": 0, "all_features_nan": 0, "exact_duplicates": 90}. Infinite cells set to NaN: 0. Non-numeric cells coerced to NaN: 0. Missing cells remaining for train-fitted imputation: 0. Conflicting feature-duplicate groups (same features, different labels) kept and reported: 632.

Post-cleaning class effect (counts before and after; rows removed; percent removed):

| class | before | after | removed | % removed |
|---|---|---|---|---|
| normal | 77,054 | 76,967 | 87 | 0.11 |
| neptune | 41,214 | 41,214 | 0 | 0.00 |
| attack | 12,833 | 12,830 | 3 | 0.02 |
| satan | 3,633 | 3,633 | 0 | 0.00 |
| ipsweep | 3,599 | 3,599 | 0 | 0.00 |
| portsweep | 2,931 | 2,931 | 0 | 0.00 |
| smurf | 2,646 | 2,646 | 0 | 0.00 |
| nmap | 1,493 | 1,493 | 0 | 0.00 |
| back | 956 | 956 | 0 | 0.00 |
| teardrop | 892 | 892 | 0 | 0.00 |
| warezclient | 890 | 890 | 0 | 0.00 |
| pod | 201 | 201 | 0 | 0.00 |
| guess_passwd | 53 | 53 | 0 | 0.00 |
| buffer_overflow | 30 | 30 | 0 | 0.00 |
| warezmaster | 20 | 20 | 0 | 0.00 |
| land | 18 | 18 | 0 | 0.00 |
| imap | 11 | 11 | 0 | 0.00 |
| rootkit | 10 | 10 | 0 | 0.00 |
| loadmodule | 9 | 9 | 0 | 0.00 |
| ftp_write | 8 | 8 | 0 | 0.00 |
| multihop | 7 | 7 | 0 | 0.00 |
| phf | 4 | 4 | 0 | 0.00 |
| perl | 3 | 3 | 0 | 0.00 |
| spy | 2 | 2 | 0 | 0.00 |

Removed columns:

| column | stage | reason |
|---|---|---|
| difficulty | identifier | NSL-KDD difficulty score produced by the dataset authors' classifier ensemble; not observable at deployment. Preserved as metadata (__difficulty) |
| num_outbound_cmds | constant | constant (<=1 distinct non-null value) over the whole dataset |

### CICIDS2017

Rows 3,119,345 -> 2,522,592 (19.13% removed). Removed rows: {"no_label": 288602, "all_features_nan": 0, "exact_duplicates": 308151}. Infinite cells set to NaN: 4,376. Non-numeric cells coerced to NaN: 0. Missing cells remaining for train-fitted imputation: 3,128. Conflicting feature-duplicate groups (same features, different labels) kept and reported: 697.

Post-cleaning class effect (counts before and after; rows removed; percent removed):

| class | before | after | removed | % removed |
|---|---|---|---|---|
| BENIGN | 2,273,097 | 2,096,714 | 176,383 | 7.76 |
| Bot | 1,966 | 1,953 | 13 | 0.66 |
| DDoS | 128,027 | 128,016 | 11 | 0.01 |
| DoS GoldenEye | 10,293 | 10,286 | 7 | 0.07 |
| DoS Hulk | 231,073 | 172,849 | 58,224 | 25.20 |
| DoS Slowhttptest | 5,499 | 5,228 | 271 | 4.93 |
| DoS slowloris | 5,796 | 5,385 | 411 | 7.09 |
| FTP-Patator | 7,938 | 5,933 | 2,005 | 25.26 |
| Heartbleed | 11 | 11 | 0 | 0.00 |
| Infiltration | 36 | 36 | 0 | 0.00 |
| PortScan | 158,930 | 90,819 | 68,111 | 42.86 |
| SSH-Patator | 5,897 | 3,219 | 2,678 | 45.41 |
| Web Attack - Brute Force | 1,507 | 1,470 | 37 | 2.46 |
| Web Attack - Sql Injection | 21 | 21 | 0 | 0.00 |
| Web Attack - XSS | 652 | 652 | 0 | 0.00 |

Removed columns:

| column | stage | reason |
|---|---|---|
| Flow ID | identifier | Flow identifier built from endpoints + ports; unique per flow, no behavioural content |
| Source IP | identifier | Host address; memorises the simulation topology (attacker IP) rather than behaviour |
| Destination IP | identifier | Host address; memorises the simulation topology (victim IP) rather than behaviour |
| Source Port | identifier | Ephemeral client port; near-random per flow, known shortcut feature for fixed attacker tooling |
| Timestamp | identifier | Capture time; not a traffic feature. Preserved as metadata (__timestamp) for leakage audits only |
| Bwd PSH Flags | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd URG Flags | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Avg Bytes/Bulk | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Avg Packets/Bulk | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Avg Bulk Rate | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd Avg Bytes/Bulk | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd Avg Packets/Bulk | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd Avg Bulk Rate | constant | constant (<=1 distinct non-null value) over the whole dataset |
| SYN Flag Count | duplicate_column | values identical to column 'Fwd PSH Flags' |
| CWE Flag Count | duplicate_column | values identical to column 'Fwd URG Flags' |
| Fwd Header Length.1 | duplicate_column | values identical to column 'Fwd Header Length' |
| Subflow Fwd Packets | duplicate_column | values identical to column 'Total Fwd Packets' |
| Subflow Bwd Packets | duplicate_column | values identical to column 'Total Backward Packets' |

### InSDN

Rows 343,889 -> 218,685 (36.41% removed). Removed rows: {"no_label": 0, "all_features_nan": 0, "exact_duplicates": 125204}. Infinite cells set to NaN: 0. Non-numeric cells coerced to NaN: 0. Missing cells remaining for train-fitted imputation: 0. Conflicting feature-duplicate groups (same features, different labels) kept and reported: 0.

Post-cleaning class effect (counts before and after; rows removed; percent removed):

| class | before | after | removed | % removed |
|---|---|---|---|---|
| BFA | 1,405 | 1,381 | 24 | 1.71 |
| BOTNET | 164 | 164 | 0 | 0.00 |
| DDoS | 121,942 | 1,520 | 120,422 | 98.75 |
| DoS | 53,616 | 50,440 | 3,176 | 5.92 |
| Normal | 68,424 | 66,965 | 1,459 | 2.13 |
| Probe | 98,129 | 98,015 | 114 | 0.12 |
| U2R | 17 | 17 | 0 | 0.00 |
| Web-Attack | 192 | 183 | 9 | 4.69 |

Removed columns:

| column | stage | reason |
|---|---|---|
| Flow ID | identifier | Flow identifier built from endpoints + ports; unique per flow, no behavioural content |
| Src IP | identifier | Host address; memorises the simulation topology (attacker IP) rather than behaviour |
| Dst IP | identifier | Host address; memorises the simulation topology (victim IP) rather than behaviour |
| Src Port | identifier | Ephemeral client port; near-random per flow, known shortcut feature for fixed attacker tooling |
| Timestamp | identifier | Capture time; not a traffic feature. Preserved as metadata (__timestamp) for leakage audits only |
| Fwd PSH Flags | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd URG Flags | constant | constant (<=1 distinct non-null value) over the whole dataset |
| CWE Flag Count | constant | constant (<=1 distinct non-null value) over the whole dataset |
| ECE Flag Cnt | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Byts/b Avg | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Pkts/b Avg | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Blk Rate Avg | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd Byts/b Avg | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd Pkts/b Avg | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Bwd Blk Rate Avg | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Init Fwd Win Byts | constant | constant (<=1 distinct non-null value) over the whole dataset |
| Fwd Seg Size Min | constant | constant (<=1 distinct non-null value) over the whole dataset |
| PSH Flag Cnt | duplicate_column | values identical to column 'Bwd PSH Flags' |
| URG Flag Cnt | duplicate_column | values identical to column 'Bwd URG Flags' |
| Fwd Seg Size Avg | duplicate_column | values identical to column 'Fwd Pkt Len Mean' |
| Subflow Fwd Pkts | duplicate_column | values identical to column 'Tot Fwd Pkts' |
| Subflow Bwd Pkts | duplicate_column | values identical to column 'Tot Bwd Pkts' |

Policy: exact duplicates = identical on all features and original label, first occurrence kept (file order). Duplicates are identified after removing identifier columns (Flow ID, IPs, ports, timestamp), so flows differing only in those fields count as duplicates. Missing values are never imputed during cleaning.

## 6. Splitting strategy

Per-class seeded random split on the original label (manual stratification) to 70/15/15; seed in `configs/config.yaml`. Row identifiers (`__row_id`) of every split are in `data/splits/<dataset>_split_indices.json`.

### NSL-KDD

Sizes: {'train': 88178, 'validation': 18897, 'test': 18898}; achieved fractions: {'train': 0.7, 'validation': 0.15, 'test': 0.15}.

Classes too small for exact 70/15/15 stratification:

| label | n | train | validation | test |
|---|---|---|---|---|
| perl | 3 | 1 | 1 | 1 |
| phf | 4 | 2 | 1 | 1 |
| spy | 2 | 1 | 0 | 1 |

Per-class counts per split:

| class | train | validation | test | total |
|---|---|---|---|---|
| normal | 47,141 | 10,101 | 10,101 | 67,343 |
| neptune | 28,850 | 6,182 | 6,182 | 41,214 |
| satan | 2,543 | 545 | 545 | 3,633 |
| ipsweep | 2,519 | 540 | 540 | 3,599 |
| portsweep | 2,051 | 440 | 440 | 2,931 |
| smurf | 1,852 | 397 | 397 | 2,646 |
| nmap | 1,045 | 224 | 224 | 1,493 |
| back | 670 | 143 | 143 | 956 |
| teardrop | 624 | 134 | 134 | 892 |
| warezclient | 622 | 134 | 134 | 890 |
| pod | 141 | 30 | 30 | 201 |
| guess_passwd | 37 | 8 | 8 | 53 |
| buffer_overflow | 22 | 4 | 4 | 30 |
| warezmaster | 14 | 3 | 3 | 20 |
| land | 12 | 3 | 3 | 18 |
| imap | 7 | 2 | 2 | 11 |
| rootkit | 6 | 2 | 2 | 10 |
| loadmodule | 7 | 1 | 1 | 9 |
| ftp_write | 6 | 1 | 1 | 8 |
| multihop | 5 | 1 | 1 | 7 |
| phf | 2 | 1 | 1 | 4 |
| perl | 1 | 1 | 1 | 3 |
| spy | 1 | 0 | 1 | 2 |

### CICIDS2017

Sizes: {'train': 1765816, 'validation': 378388, 'test': 378388}; achieved fractions: {'train': 0.7, 'validation': 0.15, 'test': 0.15}.

Classes too small for exact 70/15/15 stratification:

_(none)_

Per-class counts per split:

| class | train | validation | test | total |
|---|---|---|---|---|
| BENIGN | 1,467,700 | 314,507 | 314,507 | 2,096,714 |
| DoS Hulk | 120,995 | 25,927 | 25,927 | 172,849 |
| DDoS | 89,612 | 19,202 | 19,202 | 128,016 |
| PortScan | 63,573 | 13,623 | 13,623 | 90,819 |
| DoS GoldenEye | 7,200 | 1,543 | 1,543 | 10,286 |
| FTP-Patator | 4,153 | 890 | 890 | 5,933 |
| DoS slowloris | 3,769 | 808 | 808 | 5,385 |
| DoS Slowhttptest | 3,660 | 784 | 784 | 5,228 |
| SSH-Patator | 2,253 | 483 | 483 | 3,219 |
| Bot | 1,367 | 293 | 293 | 1,953 |
| Web Attack - Brute Force | 1,030 | 220 | 220 | 1,470 |
| Web Attack - XSS | 456 | 98 | 98 | 652 |
| Infiltration | 26 | 5 | 5 | 36 |
| Web Attack - Sql Injection | 15 | 3 | 3 | 21 |
| Heartbleed | 7 | 2 | 2 | 11 |

### InSDN

Sizes: {'train': 153079, 'validation': 32803, 'test': 32803}; achieved fractions: {'train': 0.7, 'validation': 0.15, 'test': 0.15}.

Classes too small for exact 70/15/15 stratification:

_(none)_

Per-class counts per split:

| class | train | validation | test | total |
|---|---|---|---|---|
| Probe | 68,611 | 14,702 | 14,702 | 98,015 |
| Normal | 46,875 | 10,045 | 10,045 | 66,965 |
| DoS | 35,308 | 7,566 | 7,566 | 50,440 |
| DDoS | 1,064 | 228 | 228 | 1,520 |
| BFA | 967 | 207 | 207 | 1,381 |
| Web-Attack | 129 | 27 | 27 | 183 |
| BOTNET | 114 | 25 | 25 | 164 |
| U2R | 11 | 3 | 3 | 17 |

## 7. Leakage prevention and audit

* Split precedes every fitted step; imputers, vocabularies, scalers and selectors are fitted on the training split only.
* SMOTE-NC is applied to the training split only (and inside every training fold for CV/stacking via `src/model_ready.py`).
* Validation labels are used only to choose the RFE feature count.
* Unseen categories map to code -1 (tree) / all-zero one-hot (MLP).

**NSL-KDD** - cross-split audit: near-duplicates (rounded to 3 decimals) of test rows found in train: 5 (0.03% of test); of validation rows found in train: 4 (0.02%). No timestamp available for this dataset; temporal leakage cannot be quantified.

**CICIDS2017** - cross-split audit: near-duplicates (rounded to 3 decimals) of test rows found in train: 4,789 (1.27% of test); of validation rows found in train: 4,658 (1.23%). Splits are interleaved in time (random split); ranges overlap by construction.

**InSDN** - cross-split audit: near-duplicates (rounded to 3 decimals) of test rows found in train: 14 (0.04% of test); of validation rows found in train: 8 (0.02%). Splits are interleaved in time (random split); ranges overlap by construction.

## 8-9. Feature selection (Information Gain, RFE)

### NSL-KDD

* Data: TRAIN split only (stratified subsample); validation used ONLY to choose RFE feature count.
* Information Gain: smallest top-k with cumulative IG share >= 0.95 (IG>0); estimator `mutual_info_classif (k-NN continuous / discrete for categorical)`, n_neighbors=3, subsample rows=88,178. Selected 23 of 40.
* RFE: RandomForest(n_estimators=60, max_depth=12, class_weight=balanced_subsample), step=1, subsample rows=40,799, k grid=[5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60], tolerance=0.002 on validation macro-F1 (RandomForest proxy). Chosen k=30.
* IG->RFE hybrid: chosen k=15 from 23 IG-retained features.

Feature counts: {"all_features": 40, "info_gain": 23, "rfe": 30, "info_gain_rfe": 15}

### CICIDS2017

* Data: TRAIN split only (stratified subsample); validation used ONLY to choose RFE feature count.
* Information Gain: smallest top-k with cumulative IG share >= 0.95 (IG>0); estimator `mutual_info_classif (k-NN continuous / discrete for categorical)`, n_neighbors=3, subsample rows=152,082. Selected 51 of 66.
* RFE: RandomForest(n_estimators=60, max_depth=12, class_weight=balanced_subsample), step=1, subsample rows=43,462, k grid=[5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60], tolerance=0.002 on validation macro-F1 (RandomForest proxy). Chosen k=30.
* IG->RFE hybrid: chosen k=25 from 51 IG-retained features.

Feature counts: {"all_features": 66, "info_gain": 51, "rfe": 30, "info_gain_rfe": 25}

### InSDN

* Data: TRAIN split only (stratified subsample); validation used ONLY to choose RFE feature count.
* Information Gain: smallest top-k with cumulative IG share >= 0.95 (IG>0); estimator `mutual_info_classif (k-NN continuous / discrete for categorical)`, n_neighbors=3, subsample rows=150,006. Selected 47 of 61.
* RFE: RandomForest(n_estimators=60, max_depth=12, class_weight=balanced_subsample), step=1, subsample rows=40,657, k grid=[5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60], tolerance=0.002 on validation macro-F1 (RandomForest proxy). Chosen k=8.
* IG->RFE hybrid: chosen k=8 from 47 IG-retained features.

Feature counts: {"all_features": 61, "info_gain": 47, "rfe": 8, "info_gain_rfe": 8}

## 10. SMOTE-NC

### NSL-KDD

Method used: SMOTENC; categorical columns: ['protocol_type', 'service', 'flag']; k_neighbors used: 5; seed 42; params {'k_neighbors_requested': 5, 'target_per_class': 50000, 'max_ratio': 20, 'min_samples': 6, 'scale_for_neighbours': True}.

Not oversampled (fewer than min_samples real rows): {'multihop': 5, 'phf': 2, 'spy': 1, 'perl': 1}.

| class | train_before | train_after |
|---|---|---|
| back | 670 | 13,400 |
| buffer_overflow | 22 | 440 |
| nmap | 1,045 | 20,900 |
| normal | 47,141 | 47,141 |
| perl | 1 | 1 |
| phf | 2 | 2 |
| pod | 141 | 2,820 |
| portsweep | 2,051 | 41,020 |
| rootkit | 6 | 120 |
| satan | 2,543 | 47,141 |
| smurf | 1,852 | 37,040 |
| spy | 1 | 1 |
| ftp_write | 6 | 120 |
| teardrop | 624 | 12,480 |
| warezclient | 622 | 12,440 |
| warezmaster | 14 | 280 |
| guess_passwd | 37 | 740 |
| imap | 7 | 140 |
| ipsweep | 2,519 | 47,141 |
| land | 12 | 240 |
| loadmodule | 7 | 140 |
| multihop | 5 | 5 |
| neptune | 28,850 | 47,141 |

### CICIDS2017

Method used: SMOTENC; categorical columns: ['Protocol']; k_neighbors used: 5; seed 42; params {'k_neighbors_requested': 5, 'target_per_class': 50000, 'max_ratio': 20, 'min_samples': 6, 'scale_for_neighbours': True}.

Not oversampled (fewer than min_samples real rows): none.

| class | train_before | train_after |
|---|---|---|
| BENIGN | 1,467,700 | 1,467,700 |
| Bot | 1,367 | 27,340 |
| PortScan | 63,573 | 63,573 |
| SSH-Patator | 2,253 | 45,060 |
| Web Attack - Brute Force | 1,030 | 20,600 |
| Web Attack - Sql Injection | 15 | 300 |
| Web Attack - XSS | 456 | 9,120 |
| DDoS | 89,612 | 89,612 |
| DoS GoldenEye | 7,200 | 50,000 |
| DoS Hulk | 120,995 | 120,995 |
| DoS Slowhttptest | 3,660 | 50,000 |
| DoS slowloris | 3,769 | 50,000 |
| FTP-Patator | 4,153 | 50,000 |
| Heartbleed | 7 | 140 |
| Infiltration | 26 | 520 |

### InSDN

Method used: SMOTENC; categorical columns: ['Protocol']; k_neighbors used: 5; seed 42; params {'k_neighbors_requested': 5, 'target_per_class': 50000, 'max_ratio': 20, 'min_samples': 6, 'scale_for_neighbours': True}.

Not oversampled (fewer than min_samples real rows): none.

| class | train_before | train_after |
|---|---|---|
| BFA | 967 | 19,340 |
| BOTNET | 114 | 2,280 |
| DDoS | 1,064 | 21,280 |
| DoS | 35,308 | 50,000 |
| Normal | 46,875 | 50,000 |
| Probe | 68,611 | 68,611 |
| U2R | 11 | 220 |
| Web-Attack | 129 | 2,580 |

## 11. Selected features

### NSL-KDD

**info_gain** (23): src_bytes, service, flag, diff_srv_rate, same_srv_rate, dst_host_diff_srv_rate, dst_host_same_srv_rate, count, dst_host_srv_count, dst_bytes, dst_host_serror_rate, serror_rate, dst_host_srv_serror_rate, srv_serror_rate, dst_host_same_src_port_rate, logged_in, dst_host_srv_diff_host_rate, dst_host_count, srv_count, protocol_type, dst_host_rerror_rate, srv_diff_srv_rate, dst_host_srv_rerror_rate

**rfe** (30): src_bytes, service, dst_bytes, dst_host_srv_count, dst_host_count, dst_host_serror_rate, dst_host_diff_srv_rate, wrong_fragment, count, dst_host_srv_diff_host_rate, dst_host_same_src_port_rate, dst_host_srv_serror_rate, protocol_type, dst_host_same_srv_rate, duration, srv_count, flag, num_failed_logins, land, num_access_files, logged_in, hot, dst_host_rerror_rate, num_shells, diff_srv_rate, srv_diff_srv_rate, num_file_creations, num_compromised, same_srv_rate, rerror_rate

**info_gain_rfe** (15): src_bytes, service, dst_bytes, dst_host_srv_count, dst_host_count, dst_host_serror_rate, srv_count, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_same_srv_rate, count, dst_host_diff_srv_rate, protocol_type, flag, dst_host_srv_serror_rate

### CICIDS2017

**info_gain** (51): Average Packet Size, Packet Length Mean, Packet Length Std, Packet Length Variance, Total Length of Bwd Packets, Subflow Bwd Bytes, Avg Bwd Segment Size, Bwd Packet Length Mean, Total Length of Fwd Packets, Subflow Fwd Bytes, Init_Win_bytes_backward, Max Packet Length, Bwd Packet Length Max, Fwd Packet Length Max, Init_Win_bytes_forward, Fwd IAT Max, Flow IAT Max, Fwd Header Length, Fwd IAT Total, Flow Duration, Fwd IAT Mean, Avg Fwd Segment Size, Fwd Packet Length Mean, Destination Port, Bwd Header Length, Flow Bytes/s, Bwd Packet Length Std, Fwd Packets/s, Bwd Packets/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Fwd IAT Std, Fwd Packet Length Std, Bwd IAT Max, Bwd IAT Total, Total Backward Packets, Bwd IAT Mean, Total Fwd Packets, Bwd IAT Std, Idle Max, Idle Mean, Idle Min, Active Mean, Active Max, Active Min, Bwd IAT Min, Fwd IAT Min, act_data_pkt_fwd, Bwd Packet Length Min, min_seg_size_forward

**rfe** (30): Flow Duration, Packet Length Mean, Destination Port, Fwd Packet Length Max, Init_Win_bytes_backward, Flow IAT Mean, Total Length of Bwd Packets, Subflow Bwd Bytes, Bwd Header Length, Bwd Packets/s, min_seg_size_forward, Flow Packets/s, Average Packet Size, Bwd Packet Length Max, Flow IAT Max, Subflow Fwd Bytes, Fwd Header Length, Init_Win_bytes_forward, Total Length of Fwd Packets, Max Packet Length, Fwd IAT Std, Fwd IAT Mean, Avg Bwd Segment Size, Bwd Packet Length Mean, Fwd IAT Total, Fwd Packet Length Mean, Flow IAT Std, Total Backward Packets, Packet Length Std, Total Fwd Packets

**info_gain_rfe** (25): Flow IAT Mean, Fwd Packet Length Max, Destination Port, Subflow Bwd Bytes, Init_Win_bytes_backward, Average Packet Size, Bwd Header Length, Flow IAT Std, Bwd Packet Length Max, Bwd Packets/s, Packet Length Mean, min_seg_size_forward, Flow Packets/s, Avg Bwd Segment Size, Init_Win_bytes_forward, Fwd IAT Mean, Total Length of Fwd Packets, Flow Duration, Fwd Header Length, Bwd Packet Length Mean, Max Packet Length, Packet Length Std, Fwd IAT Std, Subflow Fwd Bytes, Fwd IAT Max

### InSDN

**info_gain** (47): Bwd Header Len, Fwd Header Len, Init Bwd Win Byts, Dst Port, Pkt Len Max, Pkt Len Std, Pkt Len Var, Pkt Len Mean, Pkt Size Avg, Bwd Pkt Len Mean, Bwd Seg Size Avg, TotLen Bwd Pkts, Subflow Bwd Byts, Bwd Pkt Len Max, Tot Fwd Pkts, Bwd Pkt Len Std, Tot Bwd Pkts, Fwd Pkt Len Mean, Fwd Pkt Len Max, Subflow Fwd Byts, TotLen Fwd Pkts, Fwd Pkts/s, Flow IAT Std, Flow IAT Min, Fwd Pkt Len Std, Flow Byts/s, Flow IAT Mean, Bwd IAT Tot, Bwd IAT Mean, Fwd IAT Max, Bwd IAT Max, Fwd IAT Tot, Flow Pkts/s, Bwd Pkts/s, Fwd IAT Mean, Bwd IAT Min, Flow IAT Max, Bwd IAT Std, Flow Duration, Down/Up Ratio, Fwd IAT Min, Protocol, Bwd Pkt Len Min, Pkt Len Min, Fwd Pkt Len Min, Fwd IAT Std, FIN Flag Cnt

**rfe** (8): Init Bwd Win Byts, Dst Port, Fwd Pkts/s, Bwd Header Len, Pkt Len Mean, Flow Duration, Fwd Header Len, Flow IAT Min

**info_gain_rfe** (8): Init Bwd Win Byts, Fwd Pkts/s, Dst Port, Bwd Header Len, Pkt Size Avg, Flow Duration, Fwd Header Len, Flow IAT Min

## 12. Experimental parameters

```json
{
  "seed": 42,
  "paths": {
    "raw": "data/raw",
    "processed": "data/processed",
    "splits": "data/splits",
    "metadata": "data/metadata",
    "stats": "outputs/statistics",
    "graphs": "outputs/graphs",
    "selected": "outputs/selected_features",
    "reports": "outputs/reports"
  },
  "split": {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
    "near_duplicate_decimals": 3
  },
  "cleaning": {
    "drop_exact_duplicates": true,
    "drop_all_nan_rows": true,
    "drop_constant_columns": true,
    "drop_duplicate_columns": true
  },
  "datasets": {
    "NSL-KDD": {
      "raw_dir": "data/raw/NSL-KDD",
      "label_column": "label",
      "categorical_columns": [
        "protocol_type",
        "service",
        "flag"
      ],
      "multiclass_source": "official_train_only",
      "expected_files": {
        "KDDTrain+.txt": {
          "rows": 125973,
          "md5": "120b13c9092ab72d4902f2079143cc2c"
        },
        "KDDTest+.txt": {
          "rows": 22544,
          "md5": "1e53a506c102628114ca1b1fa509b6a8"
        }
      }
    },
    "CICIDS2017": {
      "raw_dir": "data/raw/CICIDS2017",
      "label_column": "Label",
      "categorical_columns": [
        "Protocol"
      ],
      "timestamp_dayfirst": false,
      "expected_file_count": 8
    },
    "InSDN": {
      "raw_dir": "data/raw/InSDN",
      "label_column": "Label",
      "categorical_columns": [
        "Protocol"
      ],
      "timestamp_dayfirst": true,
      "expected_file_count": 3
    }
  },
  "feature_selection": {
    "stratified_subsample_rows": {
      "info_gain": 150000,
      "rfe": 40000
    },
    "min_rows_per_class_in_subsample": 500,
    "info_gain": {
      "cumulative_mass": 0.95,
      "n_neighbors": 3
    },
    "rfe": {
      "estimator": "random_forest",
      "n_estimators": 60,
      "max_depth": 12,
      "step": 1,
      "k_grid": [
        5,
        8,
        10,
        12,
        15,
        20,
        25,
        30,
        40,
        50,
        60
      ],
      "tolerance": 0.002,
      "proxy_n_estimators": 60
    }
  },
  "smote_nc": {
    "k_neighbors": 5,
    "target_per_class": 50000,
    "max_ratio": 20,
    "min_samples": 6,
    "scale_for_neighbours": true
  },
  "export": {
    "feature_configs": [
      "all_features",
      "info_gain",
      "rfe",
      "info_gain_rfe"
    ],
    "representations": [
      "tree",
      "mlp"
    ],
    "smote_nc": true
  }
}
```

## 13. Figures and tables generated

* `outputs/graphs/fig01_class_distribution_CICIDS2017.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig02_class_distribution_InSDN.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig03_class_distribution_NSL-KDD.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig04_missing_value_comparison.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig05_smote_before_after_CICIDS2017.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig05_smote_before_after_InSDN.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig05_smote_before_after_NSL-KDD.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig06_information_gain_ranking_CICIDS2017.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig06_information_gain_ranking_InSDN.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig06_information_gain_ranking_NSL-KDD.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig07_rfe_ranking_CICIDS2017.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig07_rfe_ranking_InSDN.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig07_rfe_ranking_NSL-KDD.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig08_feature_count_comparison.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig09_cleaning_class_effect_CICIDS2017.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig09_cleaning_class_effect_InSDN.png` (+ `.pdf`, `.svg`)
* `outputs/graphs/fig09_cleaning_class_effect_NSL-KDD.png` (+ `.pdf`, `.svg`)

Tables in `outputs/statistics/`: dataset_summary.csv, class_distribution.csv, missing_value_report.csv, duplicate_report.csv, feature_schema_report.csv, identifier_leakage_candidates.csv, per-dataset cleaning/split/leakage/SMOTE reports. Rankings in `outputs/selected_features/`.

## 14. Limitations and unresolved issues

* Random stratified splitting of flow records can be optimistic (flows of one attack run are correlated). Temporal/group-aware evaluation is recommended as a robustness check; see the leakage audit above.
* Only InSDN was collected in an SDN setting; CICIDS2017 and NSL-KDD generalise to SDN only indirectly.
* Information gain uses a k-NN mutual-information estimator on a stratified subsample (see parameters); RFE uses a Random-Forest estimator on a subsample. Results may vary with other estimators.
* SMOTE-NC interpolates numeric features, so synthetic count-valued features can be non-integers. Classes with too few real samples are not oversampled; use class weights for them.
* `Destination Port` is kept (CICIDS2017/InSDN) but is a possible shortcut feature; an ablation without it is advised.
* NSL-KDD multiclass experiments use KDDTrain+ only; KDDTest+ is kept separate for binary external evaluation.

* Classes with fewer than 30 real training rows have unreliable per-class metrics: NSL-KDD buffer_overflow (22), NSL-KDD warezmaster (14), NSL-KDD land (12), NSL-KDD imap (7), NSL-KDD rootkit (6), NSL-KDD loadmodule (7), NSL-KDD ftp_write (6), NSL-KDD multihop (5), NSL-KDD phf (2), NSL-KDD perl (1), NSL-KDD spy (1), CICIDS2017 Infiltration (26), CICIDS2017 Web Attack - Sql Injection (15), CICIDS2017 Heartbleed (7), InSDN U2R (11).

* SMOTE-NC also oversamples benign classes when they are below the fixed target: InSDN Normal (46,875 to 50,000).

* InSDN DDoS loses most flows to deduplication: 121,942 before, 1,520 after, 120,422 removed.

* NSL-KDD files, if taken from a third-party mirror, should be cited as such.

---
# Research-paper-ready text

### Dataset

The proposed system is trained and evaluated on three public benchmark datasets: NSL-KDD (multiclass KDDTrain+ split: 125,973 rows; 40 features; external binary KDDTest+: 22,454 rows after duplicate removal); CICIDS2017 (2,522,592 records after cleaning, 66 features); InSDN (218,685 records after cleaning, 61 features). Each dataset is processed independently and keeps its own label set; a harmonised label view is provided only for explicitly defined cross-dataset experiments.

The NSL-KDD multiclass split uses KDDTrain+ only; KDDTest+ is held out as a separate external binary test set. The `attack` label is a coarse binary label from KDDTest+.

### Data Preprocessing

Duplicate records (identical features and label) were removed, infinite values were converted to missing values, and identifier, timestamp and constant columns were removed with each removal documented. The data were then divided into 70% training, 15% validation and 15% test subsets using a seeded per-class stratified split. Missing values were imputed with training-set medians (numeric) or modes (categorical); categorical attributes were ordinal-encoded for tree-based models and one-hot encoded for the MLP; and numeric features were Min-Max scaled (MLP only), with every statistic fitted on the training subset only and re-used unchanged on validation and test data.

### Feature Selection

Features were ranked by information gain (mutual information with the class label) and by Recursive Feature Elimination with a Random-Forest estimator, both computed on training data only. The number of features retained by RFE was selected by validation macro-F1. Resulting feature counts - NSL-KDD: IG 23, RFE 30, IG-RFE 15 of 40 features; CICIDS2017: IG 51, RFE 30, IG-RFE 25 of 66 features; InSDN: IG 47, RFE 8, IG-RFE 8 of 61 features.

### Class Balancing

Class imbalance was addressed with SMOTE-NC applied exclusively to the training data - and, for cross-validation and out-of-fold stacking, only to the training portion of each fold - so that no synthetic sample reaches validation or test data. Minority classes were oversampled toward a capped target (never above the majority-class size and at most a fixed multiple of their real size); classes with too few real samples for a reliable neighbourhood were left unchanged. Exact parameters are listed in Section 12.
