import sys
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from operator import itemgetter
from scipy.signal import find_peaks, savgol_filter

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.core.io_loaders import DatabaseLoader as dbl
from src.core.io_loaders import SpectralDataLoader as sdl
from src.core.processing import Preprocessing as pp
from src.core.processing import TimeCalibration as tc

def profile_aquisition(data_config: dict, rel_factor: float):
    data = sdl.load_csv(data_config)
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['absorption']*rel_factor, label='Absorption')
    plt.plot(data['time'], data['reference'], label='Reference')
    if data['fabry_perot'] is not None:
        plt.plot(data['time'], data['fabry_perot'], label='Fabry-Perot')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Acquisition Profile')
    plt.legend()
    plt.grid()
    plt.show()

def profile_preprocessing(data_config: dict, limits: Tuple[float, float], degree: int, smooth: bool):
    data = sdl.load_csv(data_config)
    absorption = data['absorption']
    reference = data['reference']
    fp = data['fabry_perot']
    time = data['time']

    plt.figure(figsize=(10, 6))
    absorption_normalized = pp.preprocess_data(time, absorption, limits, degree=degree, smooth=smooth)
    plt.plot(time, absorption_normalized, label='Absorption (normalized)')

    reference_normalized = pp.preprocess_data(time, reference, limits, degree=degree, smooth=smooth)
    plt.plot(time, reference_normalized, label='Reference (normalized)')

    if fp is not None:
        fp_normalized = pp.preprocess_data(time, fp, limits, degree=degree, smooth=smooth)
        plt.plot(time, fp_normalized, label='Fabry-Perot (normalized)')

    plt.axvline(x=limits[0], color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=limits[1], color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Preprocessed Data')
    plt.legend()
    plt.grid()
    plt.show()

def profile_calibration(line: str, data_config: dict, limits: Tuple[float, float], 
                        degree: int, smooth: bool, **kwargs):
    database = f'../src/database/{line}.json'
    vals = dbl.load_json(database)
    group_known_peaks = vals.get('peaks', [])

    method=kwargs.get('method', 'ref')

    data = sdl.load_csv(data_config)
    absorption = data['absorption']
    reference = data['reference']
    fp = data['fabry_perot']
    time = data['time']

    absorption_normalized = pp.preprocess_data(time, absorption, limits, degree=degree, smooth=smooth)
    reference_normalized = pp.preprocess_data(time, reference, limits, degree=degree, smooth=smooth)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    if method == 'fabry_perot' and fp is not None:
        fp_normalized = pp.preprocess_data(time, fp, limits, degree=degree, smooth=smooth)
        results = tc.time_to_frequency(time, method='fp', fabry_perot=fp_normalized, 
                                           known_spacing=kwargs.get('known_spacing'))
        ax2.plot(results['frequency'], fp_normalized, color='green', label='Fabry-Perot (calibrated)')
        
    else:
        results = tc.time_to_frequency(time, method=method, reference=reference_normalized, 
                                           group_known_peaks=group_known_peaks, 
                                           prominence=kwargs.get('prominence', 0.02), 
                                           distance=kwargs.get('distance', 30),
                                           height=kwargs.get('height', None))
        reference_peaks = list(itemgetter(*results['flat_index_peaks'])(reference_normalized))
        frequency_peaks = list(itemgetter(*results['flat_index_peaks'])(results['frequency']))
        ax1.scatter(frequency_peaks, reference_peaks, color='red', marker = 'v', s = 20, label='Detected Peaks')
        
        #all_reference_peaks = list(itemgetter(*all_peak_indices)(reference_normalized))
        #all_frequency_peaks = list(itemgetter(*all_peak_indices)(frequencies)) 
        #plt.scatter(all_frequency_peaks, all_reference_peaks, color='grey', marker = 'v', s = 20, label='All Detected Peaks', alpha=0.5)

        if method == 'second_order':
            second_order_peaks = list(itemgetter(*results['flat_index_peaks'])(results['second_order']))
            ax2.plot(results['frequency'], results['second_order'], color='green', label='Second Order Derivative')
            ax2.scatter(frequency_peaks, second_order_peaks, color='red', marker = 'v', s = 20, label='Detected Peaks')

    ax1.plot(results['frequency'], absorption_normalized, label='Absorption (calibrated)')
    ax1.plot(results['frequency'], reference_normalized, label='Reference (calibrated)')

    ax1.set_ylabel('Absorción (AU)')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_title(f'Procesamiento de Perfil Espectral', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Frecuencia Hz')
    ax2.set_ylabel('')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()  