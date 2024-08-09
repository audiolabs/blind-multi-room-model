# Demo for decay analysis using the DecayFitNet Toolbox
import pickle
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import resample

from toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from toolbox.utils import calc_mse
from toolbox.core import discard_last_n_percent, decay_model, PreprocessRIR

# ===============================================================================
# Parameters
fs = 16000

n_slopes = 1  # 0 = number of active slopes is determined by network or bayesian analysis (between 1 and 3)

filter_frequencies = [125, 250, 500, 1000, 2000, 4000, 8000]

# Bayesian paramters: a_range and n_range are both exponents, i.e., actual range = 10^a_range or 10^n_range
parameter_ranges = {"t_range": [0.1, 3.5], "a_range": [-3, 0], "n_range": [-10, -2]}
n_iterations = 100

# ===============================================================================

# df = pd.read_pickle("../2023-blind-decay-slope-estimation/data/ertd_rirs.pkl")
df = pd.read_pickle("../source/rirs/rirs_meas_arni.pkl")

# ===============================================================================
# Analyze with DecayFitNet

# Prepare the model
decayfitnet = DecayFitNetToolbox(
    n_slopes=n_slopes, sample_rate=fs, filter_frequencies=filter_frequencies
)

# df = pd.read_pickle("../source/rirs/rirs_meas.pkl")

t_preds, a_preds, n_preds, av_t_preds, av_a_preds, av_n_preds = [], [], [], [], [], []
for ind, row in tqdm(df.iterrows()):
    (t_pred, a_pred, n_pred), norm_vals = decayfitnet.estimate_parameters(
        row.rir, analyse_full_rir=True
    )
    # compute wide-band rt60
    edc = np.flip(np.cumsum(np.flip(row.rir**2))).copy()
    edc /= np.max(edc)
    (av_t_pred, av_a_pred, av_n_pred), norm_vals = decayfitnet.estimate_parameters(
        edc, input_is_edc=True, analyse_full_rir=False
    )

    t_preds.append(t_pred)
    a_preds.append(a_pred)
    n_preds.append(n_pred)

    av_t_preds.append(av_t_pred)
    av_a_preds.append(av_a_pred)
    av_n_preds.append(av_n_pred)

# store estimates in dataframe
df["dfn_dtimes"] = t_preds
df["dfn_dlevels"] = a_preds
df["dfn_nlevels"] = n_preds

df["dfn_av_dtimes"] = av_t_preds
df["dfn_av_dlevels"] = av_a_preds
df["dfn_av_nlevels"] = av_n_preds
df.to_pickle("../source/rirs/rirs_meas_arni_dfn.pkl")

# ==============================================================================

# df = pd.read_pickle("../source/rirs/rirs_sim.pkl")
# t_preds, a_preds, n_preds, av_t_preds, av_a_preds, av_n_preds = [], [], [], [], [], []
# for ind, row in tqdm(df.iterrows()):
#     t, a, n, av_t, av_a, av_n = [], [], [], [], [], []
#     for rir in row.rir.T:
#         (t_pred, a_pred, n_pred), norm_vals = decayfitnet.estimate_parameters(
#             rir, analyse_full_rir=True
#         )
#         # compute wide-band rt60
#         edc = np.flip(np.cumsum(np.flip(rir**2))).copy()
#         edc /= np.max(edc)
#         (av_t_pred, av_a_pred, av_n_pred), norm_vals = decayfitnet.estimate_parameters(
#             edc, input_is_edc=True, analyse_full_rir=False
#         )
#         t.append(t_pred)
#         a.append(a_pred)
#         n.append(n_pred)

#         av_t.append(av_t_pred)
#         av_a.append(av_a_pred)
#         av_n.append(av_n_pred)

#     t_preds.append(t)
#     a_preds.append(a)
#     n_preds.append(n)

#     av_t_preds.append(av_t)
#     av_a_preds.append(av_a)
#     av_n_preds.append(av_n)

#     # if ind > 100:
#     #     break

# # store estimates in dataframe
# df["dfn_dtimes"] = t_preds
# df["dfn_dlevels"] = a_preds
# df["dfn_nlevels"] = n_preds

# df["dfn_av_dtimes"] = av_t_preds
# df["dfn_av_dlevels"] = av_a_preds
# df["dfn_av_nlevels"] = av_n_preds

# df.to_pickle("../source/rirs/rirs_sim_dfn.pkl")
