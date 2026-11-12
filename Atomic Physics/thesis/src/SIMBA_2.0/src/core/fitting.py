import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model

from src.core.profiles import AbsortionProfiles
from src.core.analysis import AbsortionAnalyser  

class SpectrumFitter:
    @staticmethod
    def fit_wrapper(x, x0, T, N, vals, exp_config, fit_config):
        """
        Parámetros:
        -----------
        vals : dict                 -> Estructura estática con constantes y propiedades atómicas.
        exp_config: dict   -> ['L', 'Q', 'buffer_gas, 'buffer_pressure']
        fit_config: dict    -> ['profile_type', 'with_collisions', 'with_buffer']
        """
        fit_mode = fit_config.get('fit_mode', 'free')
        profile_type = fit_config['profile_type']
        pressure_params, _ = AbsortionAnalyser.get_pressure_params(T, vals)

        if fit_mode == 'N_from_T':
            N = AbsortionAnalyser.calculate_density(T, pressure_params)
        elif fit_mode == 'T_from_N':
            T = AbsortionAnalyser.calculate_temperature(N, pressure_params)

        gamma_buffer = AbsortionAnalyser.get_gamma_buffer(T, exp_config['buffer_gas'], vals)
        gamma = AbsortionAnalyser.calculate_gamma(T, exp_config['buffer_pressure'], 
                                                  gamma_buffer, pressure_params, vals, 
                                                  exp_config['with_collisions'], 
                                                  exp_config['with_buffer'])

        if profile_type == 'erfc':
            return AbsortionProfiles.profile_erfc(x, x0, T, N, exp_config['L'], exp_config['Q'], gamma, vals)
        if profile_type == 'wigner':
            return AbsortionProfiles.profile_wigner(x, x0, T, N, exp_config['L'], exp_config['Q'], gamma, vals)
        else:
            raise ValueError(f"Unknown profile type '{profile_type}'.")

    @classmethod
    def fit_spectrum(self, data:dict, vals:dict , exp_config:dict, fit_config:dict, init_params:dict):
        """
        Parámetros:
        -----------
        frequency : np.ndarray -> Variable temporal calibrada en frecuencia.
        vals : dict         -> Estructura estática con constantes y propiedades atómicas.
        exp_config: dict   -> ['L', 'Q', 'buffer_gas, 'buffer_pressure', 'with_collisions', 'with_buffer'].
        fit_config: dict    -> ['profile_type'('erfc' o 'wigner'), 'fit_method'].
        init_params: dict   -> Parámetros de iniciación y límites para cada variable dependiente.
            {
            'T':  {'value': 300.0, 'min': 270.0, 'max': 500.0, 'vary': True},
            'N':  {'value': 1.0e11, 'min': 1.0e9, 'max': 1.0e13, 'vary': True},
            'x0': {'value': 0.0,   'min': -10.0,  'max': 10.0,  'vary': False}  # Congelado temporalmente
            }
        """

        fit_mode = fit_config.get('fit_mode', 'free')

        spectra_model = lambda x, x0, T, N: SpectrumFitter.fit_wrapper(
            x, x0, T, N, 
            vals=vals,
            exp_config=exp_config,
            fit_config=fit_config
        )

        model = Model(spectra_model, independent_vars=['x'])
        params = model.make_params()

        for param_name, config in init_params.items():
            if param_name in params:
                params[param_name].set(
                    value = config['value'],
                    min = config['min'],
                    max = config['max'],
                    vary = config['vary']
                    )

        if fit_mode in ('fix_T', 'T_from_N'):
            params['T'].vary = False
            params['N'].vary = init_params['N']['vary']
        elif fit_mode in ('fix_N', 'N_from_T'):
            params['T'].vary = init_params['T']['vary']
            params['N'].vary = False
        elif fit_mode == 'free':
            params['T'].vary = init_params['T']['vary']
            params['N'].vary = init_params['N']['vary']
        else:
            raise ValueError(f"Unknown fit mode '{fit_mode}'.")
        try:
            result = model.fit(data['absorption_normalized'], x=data['frequency'], params=params, method=fit_config['fit_method'])
            return result
        
        except ValueError as e:
            raise ValueError(f"Mathematical error on spectra fitting: {e}")
            
        except Exception as e:
            raise Exception(f"An unexpected error has occurred during spectra fitting: {e}.")
