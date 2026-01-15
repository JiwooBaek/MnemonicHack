# MnemonicHack Artifacts

This repository contains the source code and datasets for the paper  
    **"MnemonicHack: Recovering the Master Seed from Bitcoin
Hardware Wallets via Side-Channel Attacks"**.

## Directory Structure
```
.  
├── Dockerfile         
├── requirements.txt      
├── datasets/  
│   ├── keys/                 
│   ├── traces/               
│   └── trace_acquisition/    
├── firmware/  
│   ├── config.c              
│   ├── trigger.h             
│   └── config.diff          
├── DL-SCA/  
│   ├── logs_manual_prep/  
│   ├── models/                
│   ├── bip39_english.c       
│   └── train.py              
└── evaluation/  
    └── eval.ipynb            
```
## Requirements

You can set up the environment manually via `pip` or use the provided Docker image for a reproducible environment.

### Option A: Manual Installation

**1. Python Dependencies**
Install the required packages:

pip install -r requirements.txt

**2. PhyWhisperer USB & Hardware Drivers**
The following libraries are required for hardware control (PhyWhisperer USB):

pip install phywhisperer pyvisa pyvisa-py  
pip install libusb libusb-package  
pip install backports.tarfile  

### Option B: Docker Environment (Recommended)

A `Dockerfile` is provided to facilitate a reproducible environment with all necessary GPU libraries and dependencies.

**1. Prerequisites**
Ensure the **NVIDIA Container Toolkit** is installed to enable GPU support within the container.

* **Installation Guide:** Official Documentation (Copy link manually)
  `docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html`

**2. Build the Image**

docker build -t mnemonic-hack .

**3. Run the Container**
Launch the container with GPU access and volume mounting (to persist results and models):

docker run --gpus all -it --rm -v $(pwd):/workspace mnemonic-hack

---

## Quick Start (Reproduction of Results)

Follow these steps to quickly verify the model performance using the pre-captured test dataset.

### 1. Download Test Dataset
Due to size constraints, datasets are hosted externally.
* **Source:** OSF Repository
  (Please copy and paste the URL below manually)
  `osf.io/9y3gd/overview?view_only=8de7507e358644c390f68c7f4fe6dcc2`
* **Action:** Download `traces/filtered_trace_D_d_20000.npy` and place it in the `datasets/traces/` directory.

### 2. Run Evaluation
Execute the Jupyter notebook at `evaluation/eval.ipynb` to perform inference using the pre-trained model.

**Expected Metrics:**
The notebook outputs the following performance metrics:
* Test accuracy (Model performance).
* Mean word inference accuracy.
* Accuracy per word position.
* **Key Recovery Rate:** Probability of correctly recovering all 24 mnemonic words.
* **Rank Analysis:** For failed recoveries (where the full 24-word set is not recovered), the script reports the rank of the correct word within the model's output probability distribution.

---

## Full Experiment Workflow

Instructions to reproduce the experiment from scratch, including firmware setup, data acquisition, training, and testing.

### 1. Hardware Setup
* **Device:** 4 x Trezor One devices (labeled A, B, C, D).
* **Equipment:** Oscilloscope, EM probe, manual voltage probe, and fixture.

### 2. Firmware Preparation
Download the official Trezor firmware source:
* **URL:** `github.com/trezor/trezor-firmware`

### 3. Firmware Modification
To enable GPIO triggering, modify the firmware source as follows:
1.  Replace `legacy/firmware/config.c` in the original source with `firmware/config.c` provided in this repository.
2.  Copy `firmware/trigger.h` to `legacy/firmware/`.
3.  Build the firmware using Docker. Refer to the official build guide below:
    `docs.trezor.io/trezor-firmware/legacy/index.html?highlight=build#building-with-docker`
### 4. Trace Acquisition
Run the capture script:

python datasets/trace_acquisition/attack_capture.py

### 5. Preprocessing
Preprocess the raw traces for deep learning input:
* Run `DL-SCA/preprocessing.ipynb`.

### 6. Model Training
Train the model using the preprocessed data:

python DL-SCA/train.py

* **Output:** Trained models will be saved in `DL-SCA/models/`.
* **Note:** Training may take over 24 hours depending on the hardware.

### 7. Evaluation
Run `evaluation/eval.ipynb` as described in the **Quick Start** section to analyze the results of your trained model.

---

## Dataset Availability

The large-scale datasets used in Section 5 are hosted on the Open Science Framework (OSF) due to file size limits.

* **URL:** (Please copy and paste the link below manually)
  `osf.io/9y3gd/overview?view_only=8de7507e358644c390f68c7f4fe6dcc2`