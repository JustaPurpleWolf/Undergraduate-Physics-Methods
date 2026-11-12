import  json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, Union,  List, Tuple

class SpectralDataLoader:
    @staticmethod
    def parse_spectral_data(filepath: str) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath, comment='#')
        except Exception as e:
            raise IOError(f"Error al leer el archivo {filepath}: {e}")
    
    @staticmethod
    def load_generic_single_channel(absortion_path: Union[str, Path],
                                     reference_path: Union[str, Path],
                                     fabry_perot_path: Optional[Union[str, Path]] = None):
        df_abs = SpectralDataLoader.parse_spectral_data(Path(absortion_path))
        df_ref = SpectralDataLoader.parse_spectral_data(Path(reference_path))

        data = {
            'time': df_abs.iloc[:, 0].to_numpy(),
            'absorption': df_abs.iloc[:, 1].to_numpy(),
            'reference': df_ref.iloc[:, 1].to_numpy(),
            'fabry_perot': None
        }

        if fabry_perot_path is not None:
            df_fp = SpectralDataLoader.parse_spectral_data(Path(fabry_perot_path))
            data['fabry_perot'] = df_fp.iloc[:, 1].to_numpy()
        return data

    @staticmethod
    def load_generic_multi_channel(filepath: Union[str, Path], 
                               col_time: int = 0, 
                               col_abs: int = 1, 
                               col_reference: int = 2, 
                               col_fabry_perot: Optional[int] = None):
        df = SpectralDataLoader.parse_spectral_data(Path(filepath))

        data = {
            'time': df.iloc[:, col_time].to_numpy(),
            'absorption': df.iloc[:, col_abs].to_numpy(),
            'reference': df.iloc[:, col_reference].to_numpy(),
            'fabry_perot': None
        }

        if col_fabry_perot is not None:
            data['fabry_perot'] = df.iloc[:, col_fabry_perot].to_numpy()
        return data
    
    @staticmethod
    def load_tektronix_tbs(absortion_path: Union[str, Path], 
                           reference_path: Union[str, Path], 
                           fabry_perot_path: Optional[Union[str, Path]] = None):
        df_abs = pd.read_csv(absortion_path, header=None)
        df_ref = pd.read_csv(reference_path, header=None)
        time_data = df_abs.iloc[:, 3].dropna().to_numpy(dtype=float)
        absorption_data = df_abs.iloc[:, 4].dropna().to_numpy(dtype=float)
        reference_data = df_ref.iloc[:, 4].dropna().to_numpy(dtype=float)

        data = {
            'time': time_data,
            'absorption': absorption_data,
            'reference': reference_data,
            'fabry_perot': None
        }

        if fabry_perot_path is not None:
            df_fp = pd.read_csv(fabry_perot_path, header=None)
            fabry_perot_data = df_fp.iloc[:, 4].dropna().to_numpy(dtype=float)
            data['fabry_perot'] = fabry_perot_data
        return data
    
    @staticmethod
    def load_csv(data_config: dict):
        """
        Parámetros:
        -----------
        data_config : dict  -> Estructura estática con configuración de carga de datos.
        {
            'model': 'generic_single' | 'generic_multi' | 'tektronix_tbs',
            'filepath': str,
            'reference_path': Optional[str],
            'fabry-perot_path': Optional[str],
            'col_time': Optional[int],
            'col_abs': Optional[int],
            'col_ref': Optional[int],
            'col_fp': Optional[int]
        }
        """
        model = data_config.get('model', None)
        if model == 'generic_single':
            abs_path = data_config.get('absortion_path')
            ref_path = data_config.get('reference_path', None)
            fp_path = data_config.get('fabry-perot_path', None)
            return SpectralDataLoader.load_generic_single_channel(abs_path, ref_path, fp_path)
        
        elif model == 'generic_multi':
            filepath = data_config.get('filepath')
            col_time = data_config.get('col_time', 0)
            col_abs = data_config.get('col_abs', 1)
            col_ref = data_config.get('col_ref', 2)
            col_fp = data_config.get('col_fp', None)
            return SpectralDataLoader.load_generic_multi_channel(filepath, col_time, col_abs, col_ref, col_fp)
        
        elif model == 'tektronix_tbs':
            abs_path = data_config.get('absortion_path')
            ref_path = data_config.get('reference_path', None)
            fp_path = data_config.get('fabry-perot_path', None)
            return SpectralDataLoader.load_tektronix_tbs(abs_path, ref_path, fp_path)
        
        else:
            raise ValueError(f"Modelo desconocido: {model}")

class DatabaseLoader:
    @staticmethod
    def get_absolute_path(filepath:str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        path = filepath.lstrip('./').replace('..', '')
        return os.path.join(root, path)
    
    @staticmethod
    def load_json(filepath: str):
        abs_path = DatabaseLoader.get_absolute_path(filepath)
        with open(abs_path, 'r+') as openfile:
            return json.load(openfile)