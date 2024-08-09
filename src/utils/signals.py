from typing import Any
import pickle
import numpy as np
from torch import Tensor
from scipy.signal import butter, sosfilt, zpk2sos

import onnx
import onnxruntime

from src.utils.DecayFitNet.python.toolbox.core import _postprocess_parameters


def get_edc(h: np.ndarray, log_edc: bool = True) -> np.ndarray:
    assert np.ndim(h) == 1
    edc = np.flip(np.cumsum(np.flip(np.abs(h) ** 2)))
    if log_edc:
        return 10 * np.log10(edc + 1e-10)
    else:
        return edc


def get_onset(rir: np.ndarray, thresh: float = 0.1) -> int:
    # get first index of exceeding a fraction of the peak
    return np.where(np.abs(rir) > np.max(np.abs(rir)) * thresh)[0][0]


class OctaveFilterbank:
    def __init__(
        self,
        fs: int = 16000,
        center_freqs: list = [125, 250, 500, 1000, 2000, 4000, 8000],
    ) -> None:
        # create filterbank
        self.sos = []
        for freq in center_freqs:
            flims = freq * np.power(2, np.array([-0.5, 0.5]))
            if any(flims >= fs / 2):
                z, p, k = butter(3, 2 * flims[0] / fs, btype="highpass", output="zpk")
            else:
                z, p, k = butter(3, 2 * flims / fs, btype="band", output="zpk")
            self.sos.append(zpk2sos(z, p, k))

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return np.stack([sosfilt(sos, signal) for sos in self.sos])


class EDCS:
    def __init__(
        self,
        fs: int = 16000,
        center_freqs: list = [125, 250, 500, 1000, 2000, 4000, 8000],
        edc_len: float = 1.0,
        edc_size: int | None = None,
        log_edc: bool = True,
        nornmalize: bool = False,
    ) -> None:

        self.fs = fs
        self.center_freqs = center_freqs
        self.edc_len = edc_len
        self.edc_size = edc_size
        self.log_edc = log_edc
        self.nornmalize = nornmalize

        self.fbank = OctaveFilterbank(fs, center_freqs)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        # slice to direct sound, take 1 s of edc
        # onset = get_onset(signal)
        signal = signal[get_onset(signal) :]
        bands = self.fbank(signal)
        edcs = np.stack([get_edc(band, log_edc=self.log_edc) for band in bands])
        # take only edc_len
        edcs = edcs[:, : int(np.round(self.fs * self.edc_len))]
        if self.edc_size is not None:
            xp = np.arange(edcs.shape[1])
            x = np.linspace(0, xp.max(), self.edc_size)
            edcs_i = np.stack([np.interp(x, xp, edc) for edc in edcs])
            edcs = edcs_i

        if self.nornmalize:
            if self.log_edc:
                edcs -= edcs.max(axis=1, keepdims=True)
            else:
                edcs /= edcs.max(axis=1, keepdims=True)
        return edcs


class DFNInference:

    def __init__(
        self,
        model_name: str = "src/utils/DecayFitNet/model/DecayFitNet_3slopes_v10.onnx",
        transform: str = "src/utils/DecayFitNet/model/input_transform_3slopes.pkl",
    ) -> None:

        onnx_model = onnx.load(model_name)
        onnx.checker.check_model(onnx_model)
        self.model = onnxruntime.InferenceSession(model_name)

        with open(transform, "rb") as handle:
            self.input_transform = pickle.load(handle)

        self.scale_adjust = {"t_adjust": 5.113781641523907, "n_adjust": 938.64}

    def __call__(self, edcs: Tensor) -> Any:
        t_preds, a_preds, n_preds = [], [], []
        for edc in edcs:
            dfn_input = {"input": edc.detach().cpu().numpy()}
            t_pred, a_pred, n_pred, n_slopes = self.model.run(None, dfn_input)
            # section below copied from DecayFitNet toolbox
            n_slopes_prediction = np.argmax(n_slopes, 1)
            n_slopes_prediction += 1  # because python starts at 0
            temp = np.tile(
                np.linspace(1, 3, 3, dtype=np.uint8), (n_slopes_prediction.shape[0], 1)
            )
            mask = np.tile(np.expand_dims(n_slopes_prediction, 1), (1, 3)) < temp
            a_pred[mask] = 0

            # Clamp noise to reasonable values to avoid numerical problems
            n_pred = np.clip(n_pred, -32, 32)
            # Go from noise exponent to noise value
            n_pred = np.power(10, n_pred)

            t_pred, a_pred, n_pred = _postprocess_parameters(
                t_pred, a_pred, n_pred, self.scale_adjust, False
            )
            t_preds.append(t_pred)
            a_preds.append(a_pred)
            n_preds.append(n_pred)

        # stack and convert to tensor
        t_preds = Tensor(np.stack(t_preds))
        a_preds = Tensor(np.stack(a_preds))
        n_preds = Tensor(np.stack(n_preds))

        return (t_preds, a_preds, n_preds)
