#region Libraries
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

from src.core.profiles import AbsortionProfiles as ap
from src.core.analysis import AbsortionAnalyser as abs

from src.core.fitting import SpectrumFitter as sfit
#endregion

def fit_all(data_config:dict, preprocess_config: dict, exp_config:dict, fit_config: dict, init_params:dict, plot_config:dict, save: bool):
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,   # Tamaño de las etiquetas de los ejes
    "xtick.labelsize": 10,  # Tamaño de los números del eje X
    "ytick.labelsize": 10,  # Tamaño de los números del eje Y
    "legend.fontsize": 10,  # Tamaño del texto en la leyenda
    "axes.linewidth": 1.0,  # Grosor del marco de los ejes
})
    print("::::::       PRESENTACIÓN DE RESULADOS       ::::::")

    data = sdl.load_csv(data_config)
    vals_path = f'../src/database/{exp_config['atomic_transition']}.json'
    vals = dbl.load_json(vals_path)
    group_known_peaks = vals.get('peaks', [])
    time = data['time']*1e3

    #region gráfica de Datos Crudos
    print("\nDatos Crudos")
    plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
    
    plt.plot(time, data['absorption']*10-1, label='Absorción', color=plot_config['colors']['absorption'], linewidth=plot_config['linewidth_data'])
    plt.plot(time, data['reference'], label='Referencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])
    
    plt.xlabel(r'Tiempo $\Delta t$ (ms)')
    plt.ylabel(r'Señal I (V)')

    plt.legend(frameon=True, edgecolor='none') # Leyenda limpia sin bordes recargados
    if plot_config['grid']: plt.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)

    plt.tight_layout()
    if save: plt.savefig(f'plots/01.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    #endregion

    #region preprocesamiento de datos
    absorption_preprocess_results = pp.preprocess_data(data['time'], data['absorption'], preprocess_config)
    reference_preprocess_results = pp.preprocess_data(data['time'], data['reference'], preprocess_config)

    data['absorption_smooth'], data['absorption_normalized'] = absorption_preprocess_results['data_smooth'], absorption_preprocess_results['data_normalized']
    data['reference_smooth'], data['reference_normalized'] = reference_preprocess_results['data_smooth'], reference_preprocess_results['data_normalized']

    x1, x2 = preprocess_config['limits']
    abs_smooth_cut, ref_smooth_cut = data['absorption_smooth'].copy(), data['reference_smooth'].copy()
    mask = (data['time'] >= x1) & (data['time'] <= x2)
    abs_smooth_cut[mask], ref_smooth_cut[mask] = np.nan, np.nan
    #endregion

    #region gráfica de Datos Suavizados
    print("\nDatos Suavizados")
    plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
        
    plt.plot(time, data['absorption_smooth']*10-1, label='Absorción', color=plot_config['colors']['absorption'], linewidth=plot_config['linewidth_data'])
    plt.plot(time, data['reference_smooth'], label='Referencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])
        
    plt.xlabel(r'Tiempo $\Delta t$ (ms)')
    plt.ylabel(r'Señal I (V)')
    
    plt.legend(frameon=True, edgecolor='none') # Leyenda limpia sin bordes recargados
    if plot_config['grid']: plt.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if save: plt.savefig(f'plots/02.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    #endregion

    #region gráfica de Datos Suavizados con Ajuste
    print("\nDatos Suavizados con Ajuste")
    plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
            
    plt.plot(time, data['absorption_smooth']*10-1, label='Absorción', color=plot_config['colors']['absorption'], linewidth=plot_config['linewidth_data'])
    plt.plot(time, data['reference_smooth'], label='Referencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])

    plt.plot(time, absorption_preprocess_results['baseline_info']['baseline_fit']*10-1, label='Fondo de Absorción', color=plot_config['colors']['fits'], linewidth=plot_config['linewidth_data'], linestyle='--')
    plt.plot(time, reference_preprocess_results['baseline_info']['baseline_fit'], label='Fondo de Referencia', color=plot_config['colors']['fits'], linewidth=plot_config['linewidth_data'], linestyle='--')

    plt.plot(time,abs_smooth_cut*10-1, label='Corte de Absorción', color=plot_config['colors']['regions'], linewidth=plot_config['linewidth_data'])
    plt.plot(time,ref_smooth_cut,  label='Corte de Referencia', color=plot_config['colors']['regions'], linewidth=plot_config['linewidth_data'])

    plt.xlabel(r'Tiempo $\Delta t$ (ms)')
    plt.ylabel(r'Señal I (V)')
        
    plt.legend(frameon=True, edgecolor='none') # Leyenda limpia sin bordes recargados
    if plot_config['grid']: plt.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
        
    plt.tight_layout()
    if save: plt.savefig(f'plots/03.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    #endregion

    #region gráfica de Datos Normalizados
    print("\nDatos Normalizados")
    plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
            
    plt.plot(time, data['absorption_normalized'], label='Absorción', color=plot_config['colors']['absorption'], linewidth=plot_config['linewidth_data'])
    plt.plot(time, data['reference_normalized'], label='Referencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])
            
    plt.xlabel(r'Tiempo $\Delta t$ (ms)')
    plt.ylabel(r'Transmitancia $\mathscr{T}$ (a.u.)')
        
    plt.legend(frameon=True, edgecolor='none') # Leyenda limpia sin bordes recargados
    if plot_config['grid']: plt.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if save: plt.savefig(f'plots/04.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    #endregion

    #region gráfica de Búsqueda de Pozos y Picos
    print("\nBúsqueda de Pozos y Picos")
    calibration_results = tc.time_to_frequency(data, preprocess_config, group_known_peaks=group_known_peaks)
    data['frequency'] = calibration_results['frequency']*1e9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=plot_config['shared_figsize'], sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=plot_config['dpi'])

    ax1.plot(time, data['reference_normalized'], label='Referencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])
    ax1.plot(time, 1- calibration_results['reference_inverted'], label='Referecia Normalizada', color=plot_config['colors']['fits'], linewidth=plot_config['linewidth_data'], linestyle= '--')

    for i in calibration_results['valley_peaks']:
        ax1.axvline(x=time[i], linestyle='--', color=plot_config['colors']['peaks'], linewidth=plot_config['linewidth_data'])
        ax2.axvline(x=time[i], linestyle='--', color=plot_config['colors']['peaks'], linewidth=plot_config['linewidth_data'])

    for (left, right), deriv in zip(calibration_results['valley_bounds'], calibration_results['second_order']):
        ax1.axvspan(xmin=time[left], xmax=time[right], color=plot_config['colors']['regions'], edgecolor=None, alpha = 0.18)
        ax2.plot(time[left:right], deriv-0.5, label='Segunda Derivada Normalizada', color=plot_config['colors']['fits'], linewidth=plot_config['linewidth_data'])

    ax2.axhline(y=0, color='#000000', linestyle='-', linewidth=plot_config['linewidth_data'])
    ax1.set_ylabel(r'Transmitancia $\mathscr{T}$ (a.u.)')
    ax1.legend(frameon=True, edgecolor='none')
    if plot_config['grid']:
        ax1.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
        ax2.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
    
    ax2.set_ylabel(r'$\partial^2 \mathscr{T} / \partial t^2$')
    ax2.set_xlabel(r'Tiempo $\Delta t$ (ms)')    
    
    plt.tight_layout()
    if save: plt.savefig(f'plots/05.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    #endregion

    #region gráfica Ajuste de Tiempo a Frecuencia
    print("\nAjuste de Tiempo a Frecuencia")
    frequency_peaks = list(itemgetter(*calibration_results['flat_index_peaks'])(data['frequency']))
    frequency = data['frequency']*1e-9
    plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
            
    plt.plot(time, frequency, label='Ajuste Tiempo-Frecuencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])
    for peak_data in calibration_results['calibration_data']:
        plt.scatter(peak_data['time']*1e3, peak_data['known_freq'],color=plot_config['colors']['marker'], marker=plot_config['marker'], s= 150, linewidths = 1.0)

    plt.xlabel(r'Tiempo $\Delta t$ (ms)')
    plt.ylabel(r'Desintonía $\Delta $ (GHz)')
        
    plt.legend(frameon=True, edgecolor='none') # Leyenda limpia sin bordes recargados
    if plot_config['grid']: plt.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if save: plt.savefig(f'plots/06.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    #endregion

    #region gráfica de Datos en Frecuencia
    print("\nDatos Normalizados")    
    plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
            
    plt.plot(data['frequency'], data['absorption_normalized'], label='Absorción', color=plot_config['colors']['absorption'], linewidth=plot_config['linewidth_data'])
    plt.plot(data['frequency'], data['reference_normalized'], label='Referencia', color=plot_config['colors']['reference'], linewidth=plot_config['linewidth_data'])

    for peak in frequency_peaks:
        plt.axvline(x=peak, linestyle='--', color=plot_config['colors']['peaks'], linewidth=plot_config['linewidth_data'])

    plt.xlabel(r'Desintonía $\Delta $ (GHz)')
    plt.ylabel(r'Transmitancia $\mathscr{T}$ (a.u.)')
        
    plt.legend(frameon=True, edgecolor='none') # Leyenda limpia sin bordes recargados
    if plot_config['grid']: plt.grid(True, color=plot_config['colors']['grid'], linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if save: plt.savefig(f'plots/07.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    #endregion

def fit_spectra(data_config:dict, preprocess_config: dict, exp_config:dict, fit_config: dict, init_params:dict):
    data = sdl.load_csv(data_config)
    vals_path = f'../src/database/{exp_config['atomic_transition']}.json'
    vals = dbl.load_json(vals_path)
    group_known_peaks = vals.get('peaks', [])
    time = data['time']

    plt.figure(figsize=(10,6))

    plt.plot(time, data['absorption'], label='Absorption')
    plt.plot(time, data['reference'], label='Reference')
    if data['fabry_perot']: plt.plot(time, data['fabry_perot'], label='Fabry-Perot')

    plt.xlabel(r'Barrido Temporal $\Delta t$ (s)')
    plt.ylabel(r'Señal $I(\Delta t)$ (V)')
    plt.title('Raw Data')
    plt.legend()
    plt.grid()
    plt.show()
    
    _, data['absorption_normalized'] = pp.preprocess_data(time, data['absorption'], preprocess_config)
    _, data['reference_normalized'] = pp.preprocess_data(time, data['reference'], preprocess_config)
    if data.get('fabry_perot'):
        _, data['fabry_perot_normalized'] = pp.preprocess_data(time, data['fabry_perot'], preprocess_config)
    else:
        data['fabry_perot_normalized'] = None

    print('Data preprocessed.')

    plt.figure(figsize=(10, 6))
    plt.plot(time, data['absorption_normalized'], label='Absorption (normalized)')
    plt.plot(time, data['reference_normalized'], label='Reference (normalized)')
    if data['fabry_perot_normalized']: plt.plot(time, data['fabry_perot_normalized'], label='Fabry-Perot (normalized)')

    plt.axvline(x=preprocess_config['limits'][0], color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=preprocess_config['limits'][1], color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Preprocessed Data')
    plt.legend()
    plt.grid()
    plt.show()
    
    if not confirm_process("Preprocessing step completed. ¿Continue process? (y/n):"):
        print("Process stopped by user.")
        return
    
    calibration_results = tc.time_to_frequency(data, preprocess_config, group_known_peaks=group_known_peaks)
    data['frequency'] = calibration_results['frequency']*1e9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                      
    if preprocess_config['calibration_method'] == 'fabry_perot':
        ax2.plot(data['frequency'], data['fabry_perot_normalized'], color='green', label='Fabry-Perot (calibrated)')
        
    else:
        reference_peaks = list(itemgetter(*calibration_results['flat_index_peaks'])(data['reference_normalized']))
        frequency_peaks = list(itemgetter(*calibration_results['flat_index_peaks'])(data['frequency']))
        ax1.scatter(frequency_peaks, reference_peaks, color='red', marker = 'v', s = 20, label='Detected Peaks')

        if preprocess_config['calibration_method'] =='second_order':
            second_order_peaks = list(itemgetter(*calibration_results['flat_index_peaks'])(calibration_results['second_order']))
            ax2.plot(data['frequency'], calibration_results['second_order'], color='green', label='Second Order Derivative')
            ax2.scatter(frequency_peaks, second_order_peaks, color='red', marker = 'v', s = 20, label='Detected Peaks')

    ax1.plot(data['frequency'], data['absorption_normalized'], label='Absorption (calibrated)')
    ax1.plot(data['frequency'], data['reference_normalized'], label='Reference (calibrated)')

    ax1.set_ylabel('Absorption (AU)')
    ax1.set_xlabel('Frecuency (Hz)')
    ax1.set_title(f'Frequency-Time Calibration', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Frecuency (Hz)')
    ax2.set_ylabel('')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()  

    if not confirm_process("Frequency calibration step completed. ¿Continue process? (y/n):"):
        print("Process stopped by user.")
        return

    def absortion_profile(x, x0, T):
        pressure_params, _ = abs.get_pressure_params(T, vals)
        gamma_buffer = abs.get_gamma_buffer(T, exp_config['buffer_gas'], vals)
        N = abs.calculate_density(T, pressure_params)
        print(f'Atomic density expected: {N} atoms/m^3')
        gamma = abs.calculate_gamma(T, exp_config['buffer_pressure'], gamma_buffer, pressure_params, vals,  exp_config['with_collisions'],  exp_config['with_buffer'])

        return  ap.profile_wigner(x,0, T, N, exp_config['L'], exp_config['Q'],  gamma,  vals = vals)

    absortion_proposal = absortion_profile(data['frequency'], 0, init_params['T']['value'])
    plt.figure(figsize=(10, 6))
    plt.plot(data['frequency'], data['absorption_normalized'], label='Absorption (experimental)')
    plt.plot(data['frequency'],absortion_proposal, label='Absortion (proposal)')

    plt.ylabel('Absorption (AU)')
    plt.xlabel('Frecuency (Hz)')
    plt.title(f'Initial Temperature Verification', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.show() 

    if not confirm_process("Initial Temperature shown. ¿Continue process? (y/n):"):
        print("Process stopped by user.")
        return 
    
    fit_results = sfit.fit_spectrum(data, vals, exp_config, fit_config, init_params)
    fit_results.plot('-')
    plt.savefig(f'plots/{data_config["save_info"]}.png', dpi=300, bbox_inches='tight')

    print(fit_results.fit_report())
    plt.show()
    
    
    p_T = fit_results.params['T']
    p_N = fit_results.params['N']
    R2 = fit_results.rsquared

    val_T = p_T.value - 273.15
    str_T = (
        f"{val_T:.4f} \u00B1 {p_T.stderr:.4f}"
        if p_T.stderr is not None
        else f"{val_T:.4f}"
        )

    val_N = abs.calculate_density(val_T + 273.15, abs.get_pressure_params(val_T + 273.15, vals)[0]) * 1e-6 if fit_config['fit_mode'] == 'N_from_T' else p_N.value * 1e-6
    str_N = (
        f"{val_N:.4e} \u00B1 {p_N.stderr * 1e-6:.4e}"
        if p_N.stderr is not None
        else f"{val_N:.4e}"
        )

    print(f"""Process ended. Data gathered: 
          T [°C]: {str_T} 
          N [atoms/cm^3]: {str_N}
          R squared: {R2:.4f}""")
    
def confirm_process(mensage="¿Continue process? (y/n): "):
    while True:
        respuesta = input(mensage).strip().lower()
        if respuesta in ['y', 'yes']:
            return True
        elif respuesta in ['n', 'no']:
            return False
        print("No valid answer. Please enter 'y' o 'n'.")