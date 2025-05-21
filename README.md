# TSFMCarbon

A unified benchmarking framework for evaluating **Time Series Foundation Models** (TSFMs) on carbon intensity forecasting.

---

## 🚀 Running Commands

```bash
cd open_tsfm_carbon
```

### Chronos

```bash
nohup bash ./scripts/run_chronos.sh > ./logs/chronos.log 2>&1 &
```

### MOMENT

```bash
nohup bash ./scripts/run_moment.sh > ./logs/moment.log 2>&1 &
```

### TimesFM

```bash
nohup bash ./scripts/run_timesfm.sh > ./logs/timesfm.log 2>&1 &
```

### Uni2TS

```bash
nohup bash ./scripts/run_uni2ts.sh > ./logs/uni2ts.log 2>&1 &
```

### AutoARIMA

```bash
nohup bash ./scripts/run_auroarima.sh > ./logs/auroarima.log 2>&1 &
```

### TimeXer

```bash
bash ./time_series_library/scripts/long_term_forecast/ETT_script/run_all_regions_TimeXer.sh
```

### EWMA

```bash
python ./EWMA/ewmaCIForecast.py
```

---

## ⚙️ TSFM Setup Instructions

### Chronos Installation


```bash
# Create environment
virtualenv chronos -p python3.10
source chronos/bin/activate

# Install packages
pip install -r requirements.txt
python3.10 -m pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

---

### MOMENT Installation

You can find it in this repo.

> ⚠️ Only Python ≥ 3.10 is supported.

```bash
# Create environment
virtualenv moment -p python3.10
source moment/bin/activate

# Install packages
pip install -r requirements.txt
python3.10 -m pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

---

### Uni2TS Installation

```bash
# Clone repo
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts

# Create environment
virtualenv uni2ts -p python3.10
source uni2ts/bin/activate

# Install
pip install -e '.[notebook]'
pip install -r requirements.txt
```

---

### TimesFM Installation

Refer to the official repo for full instructions.

> ✅ As of 2024-09-30, you can install via:

```bash
pip install timesfm
```

---

### TimeXer

Repo: https://github.com/thuml/Time-Series-Library  
Follow the instructions in the official repo.
