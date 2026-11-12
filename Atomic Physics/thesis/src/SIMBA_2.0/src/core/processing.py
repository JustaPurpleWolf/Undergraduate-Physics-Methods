import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, Tuple, List, Union

class Preprocessing:
    def smooth_data(data: np.ndarray, preprocess_config: dict) -> np.ndarray:
        window_length = preprocess_config.get('window_length', 11)
        polyorder = preprocess_config.get('polyorder', 3)
        if window_length % 2 == 0 or window_length <= 0:
            raise ValueError("window_length must be a positive odd integer.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")

        return savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=0)

    def remove_baseline(time: np.ndarray, data: np.ndarray, mask: np.ndarray, degree: int = 1) -> np.ndarray:
        baseline = data[mask]
        coeff = np.polyfit(time[mask], baseline, deg=degree)
        coeff, residuals, rank, singular_values, rcond = np.polyfit(time[mask], data[mask], deg=degree, full=True)
        baseline_fit = np.polyval(coeff, time)
        data_baseline_removed = data/baseline_fit

        fit_info = {
        "data_baseline_removed": data_baseline_removed,
        "baseline_fit": baseline_fit,
        "coefficients": coeff,
        "degree": degree,
        "residuals": residuals[0] if len(residuals) > 0 else 0.0,
        }
        return fit_info

    def normalize_data(data: np.ndarray, method: str = 'min-max') -> np.ndarray:
        if method == 'min-max':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val == 0:
                return data - min_val
            return (data - min_val) / (max_val - min_val)
        if method == 'max-abs':
            max_abs_val = np.max(np.abs(data))
            if max_abs_val == 0:
                return data
            return data / max_abs_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def preprocess_data(time: np.ndarray, data: np.ndarray, preprocess_config: dict) -> np.ndarray:
        
        if preprocess_config.get('smooth', False):
            data_smooth = Preprocessing.smooth_data(data, preprocess_config)
        else:
            data_smooth = data.copy()
        x1, x2 = preprocess_config['limits']
        mask = (time <= x1) | (time >= x2)
        baseline_info = Preprocessing.remove_baseline(time, data_smooth, mask, degree=preprocess_config['background_degree'])
        data_baseline_removed = baseline_info['data_baseline_removed']
        data_normalized = Preprocessing.normalize_data(data_baseline_removed)

        preprocess_results = {
        "data_smooth": data_smooth,
        "mask": mask,
        "baseline_info": baseline_info,
        "data_normalized": data_normalized
        }
        return preprocess_results
    
class TimeCalibration:
    @staticmethod
    def time_to_frequency(data:dict, preprocess_config: dict, **kargs) -> np.ndarray:
        method = preprocess_config['calibration_method']
        results = {
        'method_used': method,
        'success': False,
        }

        if  method == 'fabry_perot':
            if data['fabry_perot'] is None or preprocess_config['known_spacing'] is None:
                raise ValueError(f"For {method} method, 'signal' and 'known_spacing' must be provided.")
            
            results['frequency'] = TimeCalibration.calibrate_with_fabry_perot(data['time'], data['fabry_perot'], preprocess_config['known_spacing']) 
        
        elif method in ('hyperfine', 'second_order', 'doppler-hyperfine'):
            group_known_peaks = kargs.get('group_known_peaks')
            
            if group_known_peaks is None or data['reference_normalized'] is None:
                raise ValueError(f"For {method} method, 'reference' and 'group_known_peaks' must be provided.")
            
            if method == 'hyperfine':
                group_index_peaks, group_time_peaks = TimeCalibration.find_signal_peaks(data['time'], data['reference_normalized'], group_known_peaks, preprocess_config)

            elif method == 'second_order':
                second_order, (group_index_peaks, group_time_peaks) = TimeCalibration.find_second_order_peaks(data['time'], data['reference_normalized'], group_known_peaks, preprocess_config)
                results['second_order'] = second_order

            else: 
                ref_inv_norm, valley_peaks, valley_bounds, calibration_data, group_index_peaks, group_time_peaks, second_order, cropped_sections = TimeCalibration.find_doppler_hyperfine_peaks(data['time'], data['reference_normalized'], group_known_peaks, preprocess_config)

                results['reference_inverted'] = ref_inv_norm
                results['valley_peaks'] = valley_peaks
                results['valley_bounds'] = valley_bounds
                results['calibration_data'] = calibration_data
                results['second_order'] = second_order
                results['cropped_sections'] = cropped_sections
            
            results['flat_index_peaks'] = TimeCalibration.flatten(group_index_peaks)
            try:
                results['frequency'] = TimeCalibration.calibrate_with_reference(data['time'], group_known_peaks, group_time_peaks, degree=preprocess_config.get('calibration_degree', 1))
            except:
                raise ValueError(f"Calibration failed for method {method}. Check the provided known peaks and reference signal.")
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        results['success'] = True
        return results
    
    @staticmethod
    def flatten(nested_array: Union[np.ndarray, List[List[float]]]) -> List[float]:
        return [item for group in nested_array for item in group]

    @staticmethod
    def filter_and_group_peaks(time:np.ndarray , indexes: np.ndarray, known_peaks: np.ndarray)-> Tuple[List[np.ndarray], List[np.ndarray]]:
        time_candidates = time[indexes]
        flat_known_peaks = [f for group in known_peaks for f in group]

        t_first, t_second = time_candidates[0], time_candidates[1]
        f_first, f_second = flat_known_peaks[0], flat_known_peaks[1]

        slope_approx = (f_second - f_first) / (t_second - t_first)
        offset_approx = f_first - slope_approx * t_first

        grouped_time_peaks = []
        grouped_index_peaks = []
        
        for group in known_peaks:
            group_time = []
            group_index = []
            for f in group:
                t_estimated = (f - offset_approx) / slope_approx
                closest_time = time_candidates[np.argmin(np.abs(time_candidates - t_estimated))]
                group_time.append(closest_time)
                group_index.append(indexes[np.argmin(np.abs(time_candidates - t_estimated))])
            grouped_time_peaks.append(np.array(group_time))
            grouped_index_peaks.append(np.array(group_index))
        return grouped_index_peaks, grouped_time_peaks
    
    @staticmethod
    def find_signal_peaks(time: np.ndarray, reference: np.ndarray, known_peaks: np.ndarray, preprocess_config: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        flat_known_peaks = TimeCalibration.flatten(known_peaks)
        expected_peaks = len(flat_known_peaks)

        prominence = preprocess_config.get('prominence', 0.02)
        distance = preprocess_config.get('distance', 10)
        height = preprocess_config.get('height', None)

        indexes, _ = find_peaks(reference, prominence=prominence, distance=distance, height = height)
        if len(indexes) < expected_peaks:
            raise ValueError(f"Expected at least {expected_peaks} peaks, but found {len(indexes)}.")
        
        return TimeCalibration.filter_and_group_peaks(time = time, indexes= indexes, known_peaks=known_peaks)

    @staticmethod
    def find_second_order_peaks(time: np.ndarray, reference: np.ndarray, known_peaks: np.ndarray, preprocess_config: dict) -> Tuple[np.ndarray, Tuple[List[np.ndarray], List[np.ndarray]]]:
        flat_known_peaks = TimeCalibration.flatten(known_peaks)
        expected_peaks = len(flat_known_peaks)

        prominence = preprocess_config.get('prominence', 0.2)
        distance = preprocess_config.get('distance', 10)
        height = preprocess_config.get('height', None)

        deriv2 = savgol_filter(reference, window_length=11, polyorder=3, deriv=2)
        second_order = -deriv2/deriv2.max()
        second_order[0], second_order[-1] = 0,0
        indexes, _ = find_peaks(second_order, prominence=prominence, distance=distance, height=height)
        if len(indexes) < expected_peaks:
            raise ValueError(f"Expected at least {expected_peaks} peaks, but found {len(indexes)}.")

        return second_order, TimeCalibration.filter_and_group_peaks(time = time, indexes= indexes, known_peaks=known_peaks)

    @staticmethod
    def find_doppler_hyperfine_peaks(time: np.ndarray, reference: np.ndarray, known_peaks: np.ndarray, preprocess_config: dict) -> Tuple[np.ndarray, Tuple[List[np.ndarray], List[np.ndarray]]]:

        """
        Adaptive Doppler-Hyperfine Calibration (ADHC)
        """
        # 1. Identificación de los 4 valles Doppler. Suavizar agresivamente la señal con una ventana grande para "borrar" los picos hiperfinos y detectar los 4 mínimos absolutos locales del fondo, encontrar los límites de cada valle para la busqueda recursiva de los picos hiperfinos.
        ref_inv_norm, valley_peaks, valley_bounds = TimeCalibration.detect_doppler_valleys(time, reference, num_valleys=len(known_peaks), sigma_fraction=preprocess_config['sigma_fraction'], threshold_ratio=preprocess_config['threshold_ratio'])

        # 2. Recorte de secciones independientes. 
        cropped_sections = TimeCalibration.crop_doppler_sections(time, reference, valley_bounds, known_peaks)

        # 3.  Búsqueda adaptativa de picos. Usar la segunda derivada invertida dentro de un bucle de convergencia donde se ajusta la prominencia hasta que la cantidad de picos devuelta coincida.
        group_index_peaks = []
        group_time_peaks = []
        calibration_data = []
        second_order = []
        for sec in cropped_sections:
            local_peak_indexes, second_deriv = TimeCalibration.find_adaptive_section_peaks(sec)
            sec["local_peak_indexes"] = local_peak_indexes
            sec["second_deriv"] = second_deriv

            second_order.append(second_deriv)
            sec["detected_indexes"] = sec["bounds_indexes"][0] + local_peak_indexes
            sec["detected_times"] = sec["time"][local_peak_indexes]

            group_index_peaks.append(sec["detected_indexes"])
            group_time_peaks.append(sec["detected_times"])

            for i, t in enumerate(sec["detected_times"]):
                calibration_data.append({
                    "valley_index": sec["valley_index"],
                    "time": t,
                    "known_freq": sec["known_freqs"][i]
                })

        return ref_inv_norm, valley_peaks, valley_bounds, calibration_data, group_index_peaks, group_time_peaks, second_order, cropped_sections

    @staticmethod
    def calibrate_with_fabry_perot(time: np.ndarray, fabry_perot: np.ndarray, known_spacing: float) -> np.ndarray:
        
        peaks = find_peaks(fabry_perot)
        if len(peaks) < 2:
            raise ValueError("Not enough peaks found in Fabry-Perot signal for calibration.")
        peak_times = time[peaks]
        time_spacing = np.diff(peak_times)
        average_time_spacing = np.mean(time_spacing)
        calibrated_frequency = (time - peak_times[0]) * (known_spacing / average_time_spacing)
        return calibrated_frequency

    @staticmethod
    def calibrate_with_reference(time: np.ndarray, group_known_peaks: np.ndarray, group_time_peaks: np.ndarray, degree: int = 1) -> np.ndarray:
        flat_known_peaks = []
        flat_time_peaks = []
        for i, group in enumerate(group_known_peaks):
            if len(group) != len(group_time_peaks[i]):
                raise ValueError(f"Length of known peaks and time peaks must match for group {i}.")
            for j, peak in enumerate(group):
                flat_known_peaks.append(peak)
                flat_time_peaks.append(group_time_peaks[i][j])
        
        flat_known_peaks = np.array(flat_known_peaks)
        flat_time_peaks = np.array(flat_time_peaks)

        coeff = np.polyfit(flat_time_peaks, flat_known_peaks, deg=degree)
        return np.polyval(coeff, time)

    @staticmethod
    def detect_doppler_valleys(time:np.ndarray, reference: np.ndarray, num_valleys, sigma_fraction=0.03, threshold_ratio=0.03) -> List[Tuple[int, int]]:
        sigma = int(len(reference)*sigma_fraction)
        ref_smooth = gaussian_filter1d(reference, sigma = sigma)
        ref_inv = -ref_smooth
        ref_inv_norm = (ref_inv - np.min(ref_inv))/(np.max(ref_inv) - np.min(ref_inv))

        valley_peaks, _ = find_peaks(ref_inv_norm, distance=max(1, len(reference)//(num_valleys*10)), prominence=0.05)
        
        if len(valley_peaks) > num_valleys:
            prominences = ref_inv_norm[valley_peaks]
            sorted_indices = np.argsort(prominences)[-num_valleys:]
            valley_peaks = np.sort(valley_peaks[sorted_indices])

        valley_bounds = TimeCalibration.get_doppler_bounds_per_peak(ref_inv_norm, valley_peaks, threshold_ratio=threshold_ratio)
        
        return ref_inv_norm, valley_peaks, valley_bounds

    @staticmethod
    def get_doppler_bounds_per_peak(ref_inv_norm, peaks, threshold_ratio=0.03):
        bounds = []
        n_peaks = len(peaks)

        for i, p in enumerate(peaks):
            p_height = ref_inv_norm[p]
            cutoff = p_height * threshold_ratio  # Umbral para detectar línea base
        
        # -------------------------------------------------------------
        # 1. LÍMITE IZQUIERDO
        # -------------------------------------------------------------
            if i == 0:
            # Primer pico: retroceder hasta caer al umbral de línea base
                left = 0
                for idx in range(p, -1, -1):
                    if ref_inv_norm[idx] <= cutoff:
                        left = idx
                        break
            else:
                prev_p = peaks[i - 1]
                rel_min = prev_p + np.argmin(ref_inv_norm[prev_p:p])
                min_val = ref_inv_norm[rel_min]
            
                if min_val > cutoff:
                # Picos solapados: comparten el mínimo local
                    left = rel_min
                else:
                # Picos separados: retroceder desde el pico 'p' hasta el umbral
                    left = rel_min
                    for idx in range(p, rel_min, -1):
                        if ref_inv_norm[idx] <= cutoff:
                            left = idx
                            break

        # -------------------------------------------------------------
        # 2. LÍMITE DERECHO
        # -------------------------------------------------------------
            if i == n_peaks - 1:
            # Último pico: avanzar hasta caer al umbral de línea base
                right = len(ref_inv_norm) - 1
                for idx in range(p, len(ref_inv_norm)):
                    if ref_inv_norm[idx] <= cutoff:
                        right = idx
                        break
            else:
                next_p = peaks[i + 1]
                rel_min = p + np.argmin(ref_inv_norm[p:next_p])
                min_val = ref_inv_norm[rel_min]
            
                if min_val > cutoff:
                # Picos solapados: comparten el mínimo local
                    right = rel_min
                else:
                # Picos separados: avanzar desde el pico 'p' hasta el umbral
                    right = rel_min
                    for idx in range(p, rel_min):
                        if ref_inv_norm[idx] <= cutoff:
                            right = idx
                            break

            bounds.append((left, right))
        return bounds

    @staticmethod
    def crop_doppler_sections(time: np.ndarray, reference: np.ndarray, bounds: List[Tuple[int, int]], known_peaks: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        sections = []
        for i, (left, right) in enumerate(bounds):
            
            sub_time = time[left:right]
            sub_reference = reference[left:right]

            target_freqs = known_peaks[i]
            target_count = len(target_freqs)

            sections.append({
                "valley_index": i,
                "bounds_indexes": (left, right),
                "time": sub_time,
                "reference": sub_reference,
                "known_freqs": target_freqs,
                "target_count": target_count
            })
        
        return sections

    @staticmethod
    def find_adaptive_section_peaks(section: dict, window_length=15, polyorder=3, max_iter=50) -> Tuple[np.ndarray, np.ndarray]:
        sub_time = section["time"]
        sub_reference = section["reference"]
        target_count = section["target_count"]

        ref_d2 = savgol_filter(sub_reference, window_length=window_length, polyorder=polyorder, deriv=2)
        ref_d2_inv = -ref_d2
        ref_d2_norm_inv = (ref_d2_inv - np.min(ref_d2_inv)) / (np.max(ref_d2_inv) - np.min(ref_d2_inv))
        ref_d2_norm_inv[0], ref_d2_norm_inv[-1] = ref_d2_norm_inv[1], ref_d2_norm_inv[-2]  # Evitar picos en los bordes

        if target_count == 0:
            return np.array([np.argmax(ref_d2_inv)]), ref_d2_norm_inv

        prominence = 0.1
        step = 0.5
        distance = max(1, len(sub_reference) // (target_count * 100))

        for _ in range(max_iter):
            peaks, _ = find_peaks(ref_d2_norm_inv, prominence=prominence, distance=distance)
            if len(peaks) == target_count:
                return np.sort(peaks), ref_d2_norm_inv
            elif len(peaks) > target_count:
                prominence *= (1+ step)
            else:
                prominence *= (1 - step/2)

        all_peaks, props = find_peaks(ref_d2_norm_inv, prominence=0.1, distance=distance)
        if len(all_peaks) >= target_count:
            prominences = props.get('prominences', ref_d2_norm_inv[all_peaks])
            sorted_indices = np.argsort(prominences)[-target_count:]
            return np.sort(all_peaks[sorted_indices]), ref_d2_norm_inv

        return all_peaks, ref_d2_norm_inv