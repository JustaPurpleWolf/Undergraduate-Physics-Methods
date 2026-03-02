import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pydicom
import os
from lmfit import Model

class Radiograph:
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.ds = pydicom.dcmread(path)
        self.image = self.ds.pixel_array.astype(np.float64)
        self.max_bit_val = (2**(self.ds.BitsStored if 'BitsStored' in self.ds else 16)) - 1

    def invert(self, fondo=0):
        att = self.max_bit_val - fondo
        self.image = self.max_bit_val - self.image + att
        return self

    def crop(self, x1, y1, x2, y2):
        self.image = self.image[y1:y2, x1:x2]
        return self

    def measure_weighted(self, label="Muestra", thickness=0.0, sheets=1, weights=None):
        if weights is None:
            weights = np.ones_like(self.image)
        
        v1 = np.sum(weights)
        mean_w = np.sum(weights * self.image) / v1
        
        if v1 > 1:
            variance_w = np.sum(weights * (self.image - mean_w)**2) / (v1 - 1)
        else:
            variance_w = 0.0
        
        std_w = np.sqrt(variance_w)
        
        data = {
            "Filename": self.filename,
            "Label": label,
            "Thickness_mm": thickness,
            "Sheet_Count": sheets,
            "Mean_Intensity": mean_w,
            "Std_Dev": std_w,
            "Max_Bit": self.max_bit_val
        }
        
        return mean_w, std_w, data

class dicom:
    @staticmethod
    def load(path):
        return Radiograph(path)
    
    @staticmethod
    def save_to_csv(data_list, filename="calibration_thickness.csv", overwrite=False):
        df = pd.DataFrame(data_list)
        
        if overwrite:
            df.to_csv(filename, mode='w', header=True, index=False)
            print(f"Archivo '{filename}' sobrescrito con {len(df)} registros.")
        else:
            file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
            df.to_csv(filename, mode='a', header=not file_exists, index=False)
            print(f"Se han añadido {len(df)} registros a '{filename}'.")

def beer_lambert_model(z, mu, z0):
    return np.exp(-(mu*z + z0))

def plot_calibration_legacy(csv_file, unit_thickness =0.10, mu = None, berger= None, MaxBit = 4095, xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize':10})
    df = pd.read_csv(csv_file)

    fig, bx = plt.subplots(1, figsize=(10,4), dpi = 100)
    
    x = df['Sheet_Count']
    g = x*unit_thickness
    y = df['Mean_Intensity']/MaxBit
    y_err = df['Std_Dev']/MaxBit

    if mu:
        if xlim:
            n = np.linspace(0,xlim, 1000)
        else: 
            n = np.linspace(0,x.max(), 1000)
        t = n*unit_thickness
        I = np.exp(-mu*t)
        bx.plot(t, I, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
        if berger:
            I_berger = (1+berger*t)*np.exp(-mu*t)
            bx.plot(t, I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Aproximación de Berger  ($C$={berger/mu:.2f})', zorder=2)
        
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14,y = 0.93)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    bx.errorbar(g, y, yerr=y_err, ecolor='black', fmt='s', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Datos Experimentales', zorder=3)
    
    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.savefig('legacy/calibration_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, berger = None, berger_al=None, 
             MaxBit = 4095, xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(8,4), dpi = 100)
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    
    x1 = df_og['Sheet_Count']
    g1 = x1*unit_thickness
    y1 = df_og['Mean_Intensity']/MaxBit
    y1_err = df_og['Std_Dev']/MaxBit

    x2 = df_sf['Sheet_Count']
    g2 = x2*unit_thickness
    y2 = df_sf['Mean_Intensity']/MaxBit
    y2_err = df_sf['Std_Dev']/MaxBit

    gal = df_al['Thickness [mm]']
    gal_err = df_al['Delta [mm]']
    yal = df_al['Mean [12bit]']/MaxBit
    yal_err = df_al['StdDev [12bit]']/MaxBit
    
    bx.errorbar(g1, y1, yerr=y1_err, fmt='bs', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='blue', markerfacecolor='white', 
                label='Papel, Datos Crudos', zorder=1,alpha= 0.3)
    
    
    bx.errorbar(g2, y2, yerr=y2_err, fmt='b^', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='blue', markerfacecolor='white', 
                label='Papel, Corrección de Fondo', zorder=3)
    
    bx.errorbar(gal, yal, xerr= gal_err, yerr=yal_err, fmt='go', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='green', markerfacecolor='white', 
                label='Aluminio, Datos Crudos', zorder=3)
    
    if mu:
        if xlim:
            n = np.linspace(0,xlim, 1000)
        else: 
            xmax = max(x1.max(), x2.max())
            n = np.linspace(0,xmax, 1000)
        t = n*unit_thickness
        if berger:
            I_berger = (1+berger*t)*np.exp(-mu*t)
            bx.plot(t, I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Papel, Aproximación de Berger ($C$={berger/mu:.2f})', zorder=2)
        I = np.exp(-mu*t)
        bx.plot(t, I, color='blue', linewidth = 0.8, label=f'Papel, Ley de Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
            
    if mu_al:
        t = np.linspace(0, gal.max(),1000)
        if berger_al:
            I_berger = (1+berger_al*t)*np.exp(-mu_al*t)
            bx.plot(t, I_berger, color='green', linewidth = 0.8, linestyle = ':', label=f'Aluminio, Ley de Beer-Lambert ($\mu$={mu_al:.6f})', zorder=2)   
        I = np.exp(-mu_al*t)
        bx.plot(t, I, color='green', linewidth = 0.8, label=f'Aluminio, Aproximación de Berger ($C$={berger_al/mu_al:.2f})', zorder=2)

    
    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.tight_layout()
    plt.savefig('legacy/calibration_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_quo(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, berger = None, berger_al= None, MaxBit = 4095,
             label = '',figure=1, caption='', var = ''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(10,4), dpi = 100)
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    
    x1 = df_og['Sheet_Count']
    g1 = x1*unit_thickness
    y1 = df_og['Mean_Intensity']/MaxBit

    x2 = df_sf['Sheet_Count']
    g2 = x2*unit_thickness
    y2 = df_sf['Mean_Intensity']/MaxBit

    gal = df_al['Thickness [mm]']
    yal = df_al['Mean [12bit]']/MaxBit
    
    I1 = np.exp(-mu*g1)
    I2 = np.exp(-mu*g2)
    I_berger = (1+berger*g2)*np.exp(-mu*g2)
    
    Ial = np.exp(-mu_al*gal)
    I_berger_al = (1+berger_al*gal)*np.exp(-mu_al*gal)

    if var == 'paper':
        bx.plot(g1, y1/I1, color='blue', linewidth = 0.8, label=f'Papel; Datos Crudos, Beer-Lambert ($\mu$={mu:.6f})', zorder=2)   
        bx.plot(g2, y2/I2, color='blue', linewidth = 0.8, linestyle = '--', label=f'Papel; Corrección de Fondo, Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
        bx.plot(g2, y2/I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Papel; Corrección de Fondo, Berger ($C$={berger/mu:.2f})', zorder=2)

    if var == 'al':
        bx.plot(gal, yal/Ial, color='green', linewidth = 0.8, label=f'Aluminio; Datos Crudos, Beer-Lambert ($\mu$={mu_al:.6f})', zorder=2)
        bx.plot(gal, yal/I_berger_al, color='green', linewidth = 0.8, linestyle = ':', label=f'Aluminio; Datos Crudos, Berger ($C$={berger_al/mu_al:.2f})', zorder=2)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Cociente de Intensidades [AU]', fontweight='bold')
    #bx.set_ylim(0.9,1.1)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.tight_layout()
    plt.savefig('legacy/calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()

def plot_reference(csv_file, mu_al = False, z0 = 0, berger_al= False, MaxBit = 4095, xlim = False, label = '',figure=1, caption='', only_fit= False):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize':10})
    df_al = pd.read_csv(csv_file)

    fig, [bx, ax] = plt.subplots(2,1, figsize=(10,10), dpi = 100)

    gal = df_al['Thickness [mm]']
    gal_err = df_al['Delta [mm]']
    yal = df_al['Mean [12bit]']/MaxBit
    yal_err = df_al['StdDev [12bit]']/MaxBit
    
    bx.errorbar(gal, yal, xerr= gal_err, yerr=yal_err, fmt='ks', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Datos Experimentales', zorder=3)
    
    if mu_al:
        t = np.linspace(0, gal.max(),1000)
        Ial = beer_lambert_model(gal, mu_al, z0)
        I = beer_lambert_model(t, mu_al, z0)
        if not only_fit:
            bx.plot(t, I, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu_al:.4f}, $z_0$={z0:.4f})', zorder=2)
            ax.plot(gal, yal/Ial, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu_al:.4f}, $z_0$={z0:.4f})', zorder=2)
        
        if berger_al:
            I_berger_al = (1+berger_al*gal)*beer_lambert_model(gal, mu_al, z0)
            I_berger = (1+berger_al*t)*beer_lambert_model(t, mu_al, z0)
            bx.plot(t, I_berger, color='green', linewidth = 0.8, linestyle = ':',label=f'Aproximación de Berger ($C$={berger_al/mu_al:.2f})', zorder=2)
            ax.plot(gal, yal/I_berger_al, color='green', linewidth = 0.8, linestyle = ':', label=f'Aproximación de Berger ($C$={berger_al/mu_al:.2f})', zorder=2)

    model = Model(beer_lambert_model)
    params = model.make_params()

    params['mu'].set(value=mu_al, min= 0, max=1.0)
    params['z0'].set(value=z0, min= 0, max=1.0)

    result = model.fit(yal[:-1], params, z=gal[:-1], method='dual_annealing')
    
    mu_fit = result.params['mu'].value
    mu_error = result.params['mu'].stderr
    z0_fit = result.params['z0'].value
    z0_error = result.params['z0'].stderr

    I_fit = beer_lambert_model(t, mu_fit, z0_fit)
    I_fit_al = beer_lambert_model(gal, mu_fit, z0_fit)
    
    bx.plot(t, I_fit, color='blue', linewidth = 0.8, label=f'Ajuste Matemático ($\mu$={mu_fit:.4f}, $z_0$={z0_fit:.4f})', zorder=2)
    ax.plot(gal, yal/I_fit_al, color='blue', linewidth = 0.8, label=f'Ajuste Matemático ($\mu$={mu_fit:.4f}, $z_0$={z0_fit:.4f})', zorder=2)
    
    limite = 0.005
    distancia = np.abs(yal/I_fit_al - 1)
    indices = np.where(distancia < limite)[0]
    if indices.size > 0:
        indice = indices[-1]
        umbral = gal[indice]
        bx.axvline(x=umbral, color='green', linestyle='--', linewidth=0.8, label=f'Corte {umbral:.4f}mm')
        ax.axvline(x=umbral, color='green', linestyle='--', linewidth=0.8, label=f'Corte {umbral:.4f}mm')

    ax.axhspan(1-limite, 1+limite, 
           color='gray', alpha=0.2, label='Tolerancia $\pm$0.005', zorder=0)
    
    plt.suptitle(f'Curva de Atenuación: Aluminio.',fontweight = 'bold', fontsize = 14,y = 0.86)
    bx.set_title(f'Figura {figure}a. Datos, Simulación y Ajuste Matemático',  pad=10, fontsize = 12)
    ax.set_title(f'Figura {figure}b. Cociente de Error',  pad=10, fontsize = 12)
    
    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    ax.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    ax.set_ylabel(r'Cociente de Intensidad [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )
    
    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80, hspace = 0.3)
    plt.savefig('plots/reference_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    with open("reports/reference_curve_"+label+".txt", "w") as f:
        f.write(result.fit_report())
        f.write("\n\n--- ANÁLISIS DE UMBRAL ---")
        f.write(f"\nCentro definido: 1.0")
        f.write(f"\nTolerancia: 0.005")
        f.write(f"\nPunto de ruptura detectado en el grosor: {umbral:.4f} mm")
        f.write(f"\nValor del cociente en ese punto: {yal[indice]/I_fit_al[indice]:.4f}")
        
def plot_calibration(csv_file, mu = False, z0 = 0, berger= False, MaxBit = 4095, xlim = False, label = '',figure=1, caption='', only_fit= False):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize':10})
    df = pd.read_csv(csv_file)

    fig, [bx, ax] = plt.subplots(2,1, figsize=(10,10), dpi = 100)

    g = df['Thickness_mm']
    y = df['Mean_Intensity']/MaxBit
    y_err = df['Std_Dev']/MaxBit
    
    bx.errorbar(g, y, yerr=y_err, fmt='ks', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Datos Experimentales', zorder=3)
    
    if mu:
        t = np.linspace(0, g.max(),1000)
        Ig = beer_lambert_model(g, mu, z0)
        I = beer_lambert_model(t, mu, z0)
        if not only_fit:
            bx.plot(t, I, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu:.4f}, $z_0$={z0:.4f})', zorder=2)
            ax.plot(g, y/Ig, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu:.4f}, $z_0$={z0:.4f})', zorder=2)
        
        if berger:
            I_berger_g = (1+berger*g)*beer_lambert_model(g, mu, z0)
            I_berger = (1+berger*t)*beer_lambert_model(t, mu, z0)
            bx.plot(t, I_berger, color='green', linewidth = 0.8, linestyle = ':',label=f'Aproximación de Berger ($C$={berger/mu:.2f})', zorder=2)
            ax.plot(g, y/I_berger_g, color='green', linewidth = 0.8, linestyle = ':', label=f'Aproximación de Berger ($C$={berger/mu:.2f})', zorder=2)

    model = Model(beer_lambert_model)
    params = model.make_params()

    params['mu'].set(value=mu, min= 0, max=1.0)
    params['z0'].set(value=z0, min= 0, max=1.0)

    result = model.fit(y[:5], params, z=g[:5], method='dual_annealing')
    
    mu_fit = result.params['mu'].value
    mu_error = result.params['mu'].stderr
    z0_fit = result.params['z0'].value
    z0_error = result.params['z0'].stderr

    I_fit = beer_lambert_model(t, mu_fit, z0_fit)
    I_fit_g = beer_lambert_model(g, mu_fit, z0_fit)
    
    bx.plot(t, I_fit, color='blue', linewidth = 0.8, label=f'Ajuste Matemático ($\mu$={mu_fit:.4f}, $z_0$={z0_fit:.4f})', zorder=2)
    ax.plot(g, y/I_fit_g, color='blue', linewidth = 0.8, label=f'Ajuste Matemático ($\mu$={mu_fit:.4f}, $z_0$={z0_fit:.4f})', zorder=2)
    
    limite = 0.005
    distancia = np.abs(y/I_fit_g - 1)
    indices = np.where(distancia < limite)[0]
    if indices.size > 0:
        indice = indices[-1]
        umbral = g[indice]
        bx.axvline(x=umbral, color='green', linestyle='--', linewidth=0.8, label=f'Corte {umbral:.4f}mm')
        ax.axvline(x=umbral, color='green', linestyle='--', linewidth=0.8, label=f'Corte {umbral:.4f}mm')

    ax.axhspan(1-limite, 1+limite, 
           color='gray', alpha=0.2, label='Tolerancia $\pm$0.005', zorder=0)
    
    plt.suptitle(f'Curva de Atenuación: Hojas de Papel.',fontweight = 'bold', fontsize = 14,y = 0.86)
    bx.set_title(f'Figura {figure}a. Datos, Simulación y Ajuste Matemático',  pad=10, fontsize = 12)
    ax.set_title(f'Figura {figure}b. Cociente de Error',  pad=10, fontsize = 12)
    
    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    ax.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    ax.set_ylabel(r'Cociente de Intensidad [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )
    
    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80, hspace = 0.3)
    plt.savefig('plots/calibration_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    with open("reports/calibration_curve_"+label+".txt", "w") as f:
        f.write(result.fit_report())
        f.write("\n\n--- ANÁLISIS DE UMBRAL ---")
        f.write(f"\nCentro definido: 1.0")
        f.write(f"\nTolerancia: 0.005")
        f.write(f"\nPunto de ruptura detectado en el grosor: {umbral:.4f} mm")
        f.write(f"\nValor del cociente en ese punto: {y[indice]/I_fit_g[indice]:.4f}")
