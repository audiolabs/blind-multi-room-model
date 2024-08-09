# Demo for decay analysis using the DecayFitNet Toolbox
import pickle
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# from librosa import resample

from toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from toolbox.BayesianDecayAnalysis import BayesianDecayAnalysis


# ===============================================================================
# Parameters
fs = 48000
n_slopes = 1  # 0 = number of active slopes is determined by network or bayesian analysis (between 1 and 3)
filter_frequencies = [125, 250, 500, 1000, 2000, 4000, 8000]


with open("../2023-blind-decay-slope-estimation/data/ertd_rirs.pkl", "rb") as f:
    data = pickle.load(f)

# Bayesian paramters: a_range and n_range are both exponents, i.e., actual range = 10^a_range or 10^n_range
parameter_ranges = {"t_range": [0.1, 3.5], "a_range": [-3, 0], "n_range": [-10, -2]}
n_iterations = 100

# ===============================================================================
# Analyze with BayesianDecayAnalysis

# Init Bayesian decay analysis
bda = BayesianDecayAnalysis(
    sample_rate=fs,
    n_slopes=n_slopes,
    parameter_ranges=parameter_ranges,
    n_iterations=n_iterations,
    filter_frequencies=filter_frequencies,
)

t_preds, a_preds, n_preds, av_t_preds, av_a_preds, av_n_preds = [], [], [], [], [], []
for rir in tqdm(data["roomTransitionSimulation"]["rirs"].T):
    (t_pred, a_pred, n_pred), norm_vals = bda.estimate_parameters(
        rir,
        input_is_edc=False,
    )

    # compute wide-band rt60
    edc = np.flip(np.cumsum(np.flip(rir**2))).copy()
    edc /= np.max(edc)
    (av_t_pred, av_a_pred, av_n_pred), norm_vals = bda.estimate_parameters(
        edc,
        input_is_edc=True,
    )

    t_preds.append(t_pred)
    a_preds.append(a_pred)
    n_preds.append(n_pred)

    av_t_preds.append(av_t_pred)
    av_a_preds.append(av_a_pred)
    av_n_preds.append(av_n_pred)

    # t_preds.append(t_pred[:, 0][:, None])
    # a_preds.append(a_pred[:, 0][:, None])
    # n_preds.append(n_pred[:, 0][:, None])

    # av_t_preds.append(av_t_pred[:, 0][:, None])
    # av_a_preds.append(av_a_pred[:, 0][:, None])
    # av_n_preds.append(av_n_pred[:, 0][:, None])

data["roomTransitionSimulation"]["dtimes"] = np.stack(t_preds).squeeze()
data["roomTransitionSimulation"]["dlevels"] = np.stack(a_preds).squeeze()
data["roomTransitionSimulation"]["nlevels"] = np.stack(n_preds).squeeze()

data["roomTransitionSimulation"]["av_dtimes"] = np.stack(av_t_preds).squeeze()
data["roomTransitionSimulation"]["av_dlevels"] = np.stack(av_a_preds).squeeze()
data["roomTransitionSimulation"]["av_nlevels"] = np.stack(av_n_preds).squeeze()

with open(
    "../2023-blind-decay-slope-estimation/data/ertd_rirs_1slope_bayes.pkl", "wb"
) as f:
    pickle.dump(data, f)

foo = 1

# # store estimates in dataframe
# df["dfn_dtimes"] = t_preds
# df["dfn_dlevels"] = a_preds
# df["dfn_nlevels"] = n_preds

# df["dfn_av_dtimes"] = av_t_preds
# df["dfn_av_dlevels"] = av_a_preds
# df["dfn_av_nlevels"] = av_n_preds
# df.to_pickle("../source/rirs/rirs_mean_dfn.pkl")

# ==============================================================================
