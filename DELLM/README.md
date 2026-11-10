# DELLM

**Edge-Assisted DDoS Detection using a Low-Rank-Adapted Language Model in IIoT**

DELLM splits DDoS detection between the edge and the cloud:

* **Edge.** Each data cluster (C1–C4) has three small MLPs on 12 shared flow features. They are calibrated with temperature scaling. A flow is kept at the edge when its normalised entropy is at most τ. It is uploaded when it is uncertain, when a group trigger fires (a flood towards one destination), or when it is picked for a 1% random audit.
* **Cloud.** Uploaded flows are queued for 1 s or 1024 flows, grouped by destination, and ordered into sequences of 64. Each flow becomes a 768-dimensional token (the hidden layers of all 12 edge MLPs). A projection maps it to 4096 dimensions, and Llama 3.1 8B (LoRA r=8 on W_q, W_v, causal mask removed) reads the whole sequence. A linear head then labels every flow.
* **Classes (8).** Benign, NetDDoS, AppDDoS, LDDoS, DrDoS, Botnet, MQTT_DDoS, Frag_DDoS.
* **Attribution.** The system reports victims per 2 s window. It sorts sources into abused reflectors, likely spoofed addresses, or compromised devices.

```
repo/
├── configs/default.yaml      every setting (datasets, thresholds, model, training)
├── src/dellm/                the library
│   ├── data/                 loading, label mapping, Boruta feature selection
│   ├── edge/                 MLPs, temperature scaling, τ selection, group trigger, decisions
│   ├── cloud/                tokens, order generation, LLM model
│   ├── attribution.py        victims and source groups
│   └── metrics.py
├── scripts/00 … 10           the pipeline, run in order
├── tests/                    unit tests on synthetic data (no GPU needed)
├── tools/                    download helpers
└── data/README.md            where to put the datasets
```

---

## 1. Put the code on GitHub (do this once)

1. **Create an account.** Make a free account at https://github.com if you don't have one.
2. **Create the repository.** Click **+ → New repository**:
   * Name it `DELLM`.
   * Choose Private (recommended until the paper is submitted).
   * Do **not** tick "Add a README" (this folder already has one).
   * Click **Create repository**.
3. **Install Git.**
   * Windows: https://git-scm.com/download/win, with the default options.
   * Linux/macOS: Git is usually already installed.
4. **Unzip the code.** Unzip `DELLM.zip`, open a terminal (Windows: *Git Bash*) inside the `DELLM` folder, and run:

   ```bash
   git init
   git add .
   git commit -m "DELLM: initial code"
   git branch -M main
   git remote add origin https://github.com/<your-username>/DELLM.git
   git push -u origin main
   ```

   When Git asks for a password, paste a **personal access token**, not your GitHub password. Create one at GitHub → Settings → Developer settings → Personal access tokens → *Tokens (classic)* → Generate, with the `repo` scope ticked.
5. **Push later changes:**

   ```bash
   git add .
   git commit -m "what you changed"
   git push
   ```

The `.gitignore` file keeps datasets, trained models and results **out** of GitHub. They are far larger than GitHub allows (100 MB per file), so never remove those lines.

---

## 2. Set up the environment

You need Python 3.10–3.12. For the cloud model you also need an NVIDIA GPU (see section 5).

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 1) PyTorch: pick the command for your CUDA version at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) everything else
pip install -r requirements.txt
pip install -e .
# 3) check that the code works (synthetic data, takes seconds)
python -m pytest -q
```

### Get access to Llama

1. **Accept the licence.** Open https://huggingface.co/meta-llama/Llama-3.1-8B, log in, and accept the licence. Do the same for https://huggingface.co/meta-llama/Llama-3.2-1B, which is used for quick development runs. Approval usually takes minutes to a few hours.
2. **Create a token.** Go to https://huggingface.co/settings/tokens and create a *Read* token.
3. **Log in** with `huggingface-cli login` and paste the token.

---

## 3. Download the datasets

The links are below. Put the files in the places listed in `data/README.md`.

| Use | Dataset | Link | Format to download |
|---|---|---|---|
| C1 | Edge-IIoTset | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot | PCAP folders (normal + DDoS attacks) |
| C2 | CICIoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html | PCAP files of the DDoS / Mirai / benign captures |
| C3 | CIC-DDoS2019 | https://www.unb.ca/cic/datasets/ddos-2019.html | `CSVs-01-12.zip`, `CSVs-03-11.zip` |
| C3 (benign) | CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | `MachineLearningCSV.zip` → Monday file only |
| C4 | BCCC-IoT-MQTT-IDS-2025 | York University BCCC datasets page (link in the paper dataset table) | PCAPs (benign + MQTT DDoS) |
| test only | SDN-SlowRate-DDoS, AdDDoSDN, DDoS-AT-2022, Low-rate MQTT | links in the paper's dataset table | as provided |

How to download them:

* **UNB/CIC (CICIoT2023, CIC-DDoS2019, CIC-IDS2017).** Fill in the form on the dataset page and you get a link to a file listing.
  * To download many files at once **without WSL**, copy the file links into `urls.txt` (one per line) and run:
    `powershell -ExecutionPolicy Bypass -File tools\download_ciciot2023.ps1 urls.txt data\raw\CICIoT2023`
  * On Linux/macOS, run `bash tools/download_list.sh urls.txt data/raw/CICIoT2023` instead.
  * Both scripts resume broken downloads.
* **Kaggle.** Either click *Download*, or run `pip install kaggle`, put `kaggle.json` in `~/.kaggle/`, and run `kaggle datasets download mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`.
* **Which attacks to keep.** Only DDoS families and benign traffic. Single-source DoS, reconnaissance, brute force, injection and similar classes are dropped automatically by the label rules in `src/dellm/data/labels.py`, so you can download whole folders.

> **Why PCAPs for some datasets?** All datasets must have the *same* CICFlowMeter features plus source/destination IP and a timestamp. The trigger, the ordering and the attribution all need these. The CICIoT2023 and Edge-IIoTset CSVs have neither the same features nor the IPs, so step 01 rebuilds the flows from their PCAPs.

---

## 4. Run the pipeline (in this order)

Every script reads `configs/default.yaml`. Outputs go to `data/processed/`, `artifacts/` and `results/`.

| Step | Command | What it does | Needs GPU? |
|---|---|---|---|
| 01 | `python scripts/01_extract_flows.py --dataset CICIoT2023` (repeat per PCAP dataset) | PCAP → CICFlowMeter CSV, label = folder name | no |
| 02 | `python scripts/02_prepare_data.py` | maps labels to the 8 classes, cleans, 70/10/20 time split per label. **Read the mapping report it prints.** | no |
| 03 | `python scripts/03_select_features.py` | Boruta per cluster, then consensus 12 features → `artifacts/features.json` | no |
| 04 | `python scripts/04_train_edge.py --device cuda` | 3 MLPs per cluster, temperature T, threshold τ, trigger thresholds | optional |
| 05 | `python scripts/05_edge_decisions.py` | edge / uncertain / trigger / audit decision for every flow | optional |
| 06 | `python scripts/06_build_tokens.py` | 768-d token per flow | optional |
| 07 | `python scripts/07_build_sequences.py` | queue batches → ordered 64-flow sequences | no |
| 00 | `python scripts/00_check_mask.py --backbone meta-llama/Llama-3.2-1B` | checks that the causal mask is really removed | yes (or slow CPU) |
| 08 | `python scripts/08_train_cloud.py` | trains projection + LoRA + head, early stopping on validation macro-F1 | **yes** |
| 09 | `python scripts/09_evaluate.py` | final labels, macro-F1, upload rate, audit disagreement, victims, time per flow | **yes** |
| 10 | `python scripts/10_baseline_xgboost.py` | XGBoost baseline on the same 12 features | no |

### Develop small first, then scale up

1. **Small run.** In `configs/default.yaml`, set `max_rows_per_file: 200000` and run steps 02–07. This checks that every dataset loads and maps correctly.
2. **1B model.** Run steps 00 and 08 with the 1B model: `python scripts/08_train_cloud.py --backbone meta-llama/Llama-3.2-1B --out cloud_1b`.
3. **Full run.** Once everything works, set `max_rows_per_file: null`, re-run from step 02, and train the 8B model (`python scripts/08_train_cloud.py`).

### Ablations for the paper (same scripts, different flags)

| Experiment | Command |
|---|---|
| No LLM (projection → head) | `08_train_cloud.py --no-llm --out cloud_nollm`, then `09_evaluate.py --model-dir cloud_nollm --no-llm` |
| Frozen LLM (no LoRA) | `08_train_cloud.py --no-lora --out cloud_frozen`, then `09_evaluate.py --model-dir cloud_frozen` |
| Model size | `--backbone meta-llama/Llama-3.2-1B` / `Llama-3.2-3B` with different `--out` |
| Edge only | already in every results file under `edge_only` |
| Upload budget / error target | change `edge.eps`, `edge.budget` and re-run 04–09 |

---

## 5. Hardware

| Model | Setting | GPU memory (approx.) | Where |
|---|---|---|---|
| Llama 3.2 1B | bf16 + LoRA | ~8 GB | Colab T4 / any 8 GB+ GPU |
| Llama 3.1 8B | `load_in_4bit: true` (QLoRA) | ~12–16 GB | RTX 3090/4090, A10, L4, Colab Pro |
| Llama 3.1 8B | bf16 + LoRA | ~24–40 GB | A100 / A6000 / university cluster |

* **4-bit loading.** Set `cloud.load_in_4bit: true` and install `bitsandbytes`. This needs Linux or WSL with an NVIDIA GPU.
* **Running out of memory.** Lower `cloud.batch_size` and raise `cloud.grad_accum` by the same factor.

---

## 6. What to report

Each `results/results_<model>_<split>.json` file has, per dataset:

* `edge_only` and `final` (per-class precision, recall and F1, macro-F1, accuracy);
* `upload_rate` and the count of each upload reason;
* `audit_disagreement`;
* `cloud_seconds_per_flow`;
* the victim/incident counts per family and source group.

Report the numbers you actually get. If something underperforms, say so and explain it.

---

## 7. Notes and known limits

* **Tested so far.** The label mapping, temperature scaling, τ selection, trigger, edge decisions, ordering and attribution are unit-tested (`pytest`), and steps 02–03 were run end to end on synthetic data. The PyTorch/Transformers parts (steps 04–09) have only been syntax-checked, so expect small fixes on the first real run. The mask test in step 00 is the first thing to check.
* **Label mapping.** Different datasets name attacks differently. The rules in `src/dellm/data/labels.py` handle the common names. Anything unmapped is dropped and listed in step 02's report, so add `extra_rules` in the config where needed.
* **Test-only datasets.** These are never used for training. Their τ and trigger thresholds are recalibrated on their first 10% of benign flows only.
