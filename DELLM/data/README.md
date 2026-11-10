# Data folder

Nothing in here is committed to GitHub (see `.gitignore`). Put the downloads here:

```
data/raw/
  Edge-IIoTset/<attack_name>/*.pcap          # PCAPs, one folder per label
  CICIoT2023/<attack_name>/*.pcap            # PCAPs, one folder per label
  CIC-DDoS2019/*.csv                         # CSVs from CSVs-01-12.zip and CSVs-03-11.zip
  CIC-IDS2017/Monday-WorkingHours.pcap_ISCX.csv  # benign day (MachineLearningCSV.zip)
  BCCC-IoT-MQTT-IDS-2025/<label>/*.pcap
  SDN-SlowRate-DDoS/<label>/*.pcap           # test only
  AdDDoSDN/*.csv                             # test only
  DDoS-AT-2022/*.csv                         # test only
  LowRate-MQTT/<label>/*.pcap                # test only
```

Script `01_extract_flows.py` turns every `data/raw/<dataset>/<label>/*.pcap` into
`data/flows/<dataset>/<label>/*.csv`. Datasets that already ship CICFlowMeter CSVs
(CIC-DDoS2019, CIC-IDS2017, AdDDoSDN, DDoS-AT-2022) are read straight from `data/raw`.

If a folder name does not map to the right class, add an `extra_rules` entry for that
dataset in `configs/default.yaml` and re-run step 02. Step 02 prints a mapping report:
read it every time.
