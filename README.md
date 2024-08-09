# A Multi-Room Transition Dataset for Blind Estimation of Energy Decay

Philipp Götz<sup>1</sup>, Georg Götz<sup>2</sup>, Nils Meyer-Kahlen<sup>2</sup>, Kyung Yun Lee<sup>2</sup>, Karolina Prawda<sup>2</sup>, Emanuël A. P. Habets<sup>1</sup>, and Sebastian J. Schlecht<sup>3</sup>

_<sup>1</sup>International Audio Laboratories Erlangen, Germany\
<sup>2</sup>Acoustics Lab, Dpt. of Information and Communications Engineering, Aalto University, Finland\
<sup>3</sup>Friedrich-Alexander-University Erlangen-Nuremberg (FAU), Germany_

---
The multimodal dataset described in the IWAENC 2024 publication, including room impulse responses (RIRs) and 360˚-photos of each measurement position, is hosted at Zenodo (https://zenodo.org/records/11388246).

This is the accompanying code repository for the blind energy decay function (EDF) estimation method proposed in the paper and includes the source code of the model, the training routines and some basic visualizing of the results. 

## Installation
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

The Python packages required to run the code can be installed using the requirements.txt file
```
pip install -f requirements.txt
```
The model training and evaluation data is hosted at a Google-Drive and downloaded by running ``download.sh``. Depending on wether a GPU is available or not, a softlink to ``cpu.yaml`` or ``gpu.yaml`` can be created in ``configs/local``.
## Training
The pre-generated dataset from the Google-Drive is constructed as described in the paper. In a preliminary step the blind T60 estimator is trained by running
```
python src/train.py -cn train model=baseline_t60 hydra=baseline_t60
```
Upon convergence, the trained model is used as a baseline which computes linear EDCs from blind T60 estimates. As an additional, non-blind baseline (not based on speech but on RIRs directly), a pre-trained _DecayFitNet_ is used to generate multi-slope EDCs. The blind EDC estimator is trained using
```
python src/train.py -cn train model=baseline_edc
```
More information on DecayFitNet can be found at https://github.com/georg-goetz/DecayFitNet.