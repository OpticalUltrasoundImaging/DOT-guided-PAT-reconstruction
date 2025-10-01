from __future__ import annotations
import os
from scipy import io as spio
from dataclasses import dataclass, asdict
import numpy as np
from typing import Any, Sequence


def _load_sequence_info():
    mat_file_path = os.path.join('..','artifacts','LSequence.mat')
    mat = spio.loadmat(mat_file_path)
    Roi = mat.get('Roi')
    System = mat.get('System')
    return Roi, System

@dataclass
class LinearSystemParam:
    c: float
    fs: float
    N_ele: float
    pitch: float
    fc: float
    ele_width: float
    ele_height: float
    pixel_d: float
    N_sc: int
    N_ch: int
    Nfocus: int
    fc_scaled: float
    RxFnum: float
    FOV: float
    x0: float
    dx: float
    ScanPosition: np.ndarray
    ElePosition: np.ndarray
    half_rx_ch: float
    d_sample: np.ndarray

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

def _load_lsystem_param_us(Roi: Sequence[Any], System: Any, *,
                           offset: int = 0, default_Nfocus: int = 3000) -> LinearSystemParam:
    c = 1540.0  # speed of sound in tissue (m/s)
    transducer = System["Transducer"][0]
    parameters = System["Parameters"][0]

    fs = float(parameters['sampleFreqMHz'][0][0]) * 1e6  # Hz
    pitch = float(transducer['elementPitchCm'][0][0]) / 100.0 # convert cm->m
    fc = float(transducer['frequencyMHz'][0][0]) * 1e6 # convert MHz->Hz
    N_ele = int(transducer['elementCnt'][0][0])
    ele_width = float(transducer['elementWidthCm'][0][0]) / 100.0 # convert cm->m
    ele_height = 6e-3  # m
    pixel_d = c / fs / 2.0  # physical distance per sample ROUND-TRIP (m)
    N_sc = N_ele

    N_ch = int(parameters['receiveNum'][0][0])
    Nfocus = int(default_Nfocus)
    fc_scaled = fc / fs * Nfocus / 2.0
    RxFnum = 1.0
    lateral_length_cm = Roi[0]['lateralLength'][0][0]
    FOV = float(lateral_length_cm)*1e-2 # convert cm -> m
    x0 = -FOV / 2.0
    dx = FOV / (N_sc-1) # separation between scan lines
    ScanPosition = np.linspace(x0, x0 + (N_sc - 1) * dx, num = N_sc)
    ElePosition = np.arange(x0, -x0 + pitch * 0.5, pitch)
    half_rx_ch = N_ch * pitch * 0.5
    n_sample = np.arange(Nfocus, dtype=float) + float(offset)
    d_sample = n_sample * pixel_d

    return LinearSystemParam(
        c=c,
        fs=fs,
        N_ele=N_ele,
        pitch=pitch,
        fc=fc,
        ele_width=ele_width,
        ele_height=ele_height,
        pixel_d=pixel_d,
        N_sc=N_sc,
        N_ch=N_ch,
        Nfocus=Nfocus,
        fc_scaled=fc_scaled,
        RxFnum=RxFnum,
        FOV=FOV,
        x0=x0,
        dx=dx,
        ScanPosition=ScanPosition,
        ElePosition=ElePosition,
        half_rx_ch=half_rx_ch,
        d_sample=d_sample,
    )

def _load_lsystem_param_pa(Roi: Sequence[Any], System: Any, *,
                           offset: int = 0, default_Nfocus: int = 2048) -> LinearSystemParam:
    c = 1540.0  # speed of sound in tissue (m/s)
    transducer = System["Transducer"][0]
    parameters = System["Parameters"][0]

    fs = float(parameters['sampleFreqMHz'][0][0]) * 1e6  # Hz
    pitch = float(transducer['elementPitchCm'][0][0]) / 100.0 # convert cm->m
    fc = float(transducer['frequencyMHz'][0][0]) * 1e6 # convert MHz->Hz
    N_ele = int(transducer['elementCnt'][0][0])
    ele_width = float(transducer['elementWidthCm'][0][0]) / 100.0 # convert cm->m
    ele_height = 6e-3  # m
    pixel_d = c / fs  # physical distance per sample ONE-WAY (m)
    N_sc = N_ele

    N_ch = int(parameters['receiveNum'][0][0])
    Nfocus = int(default_Nfocus)
    fc_scaled = fc / fs * Nfocus / 2.0
    RxFnum = 1.0
    lateral_length_cm = Roi[0]['lateralLength'][0][0]
    FOV = float(lateral_length_cm)*1e-2 # convert cm -> m
    x0 = -FOV / 2.0
    dx = FOV / (N_sc-1) # separation between scan lines
    ScanPosition = np.linspace(x0, x0 + (N_sc - 1) * dx, num = N_sc)
    ElePosition = np.arange(x0, -x0 + pitch * 0.5, pitch)
    half_rx_ch = N_ch * pitch * 0.5
    n_sample = np.arange(Nfocus, dtype=float) + float(offset)
    d_sample = n_sample * pixel_d

    return LinearSystemParam(
        c=c,
        fs=fs,
        N_ele=N_ele,
        pitch=pitch,
        fc=fc,
        ele_width=ele_width,
        ele_height=ele_height,
        pixel_d=pixel_d,
        N_sc=N_sc,
        N_ch=N_ch,
        Nfocus=Nfocus,
        fc_scaled=fc_scaled,
        RxFnum=RxFnum,
        FOV=FOV,
        x0=x0,
        dx=dx,
        ScanPosition=ScanPosition,
        ElePosition=ElePosition,
        half_rx_ch=half_rx_ch,
        d_sample=d_sample,
    )

def linear_us_param():
    roi_us , sys_us = _load_sequence_info()
    return _load_lsystem_param_us(roi_us , sys_us)

def linear_pa_param():
    roi_us , sys_us = _load_sequence_info()
    return _load_lsystem_param_pa(roi_us , sys_us)

def list_subfolders(folder_name: str):
    all_items = os.listdir(folder_name)
    all_folders = [os.path.join(folder_name, f) for f in all_items if os.path.isdir(os.path.join(folder_name, f))]
    return all_folders