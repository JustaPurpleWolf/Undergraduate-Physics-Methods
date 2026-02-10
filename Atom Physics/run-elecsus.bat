@echo off
set CONDA_ACTIVATE_PATH="C:\Users\Personal\anaconda3\Scripts\activate.bat"
set ENV_NAME=elecsus-env
set PROJECT_DIR="C:\Users\Personal\Documents\GitHub\Undergraduate-Physics-Methods\Atom Physics\ElecSus-dev"

start cmd /k "%CONDA_ACTIVATE_PATH% & conda activate %ENV_NAME% & cd /d %PROJECT_DIR% & pip list & python -m elecsus.elecsus_gui & pause"