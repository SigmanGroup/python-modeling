# python-modeling
This repository contains the Sigman lab workflow for linear modeling, primarily driven by bidirectional stepwise MLR, as well as tools for feature curation.

# Installation
A conda environment is provided in the modeling_env_***.yml files. To create the environment, run \
`conda env create --file=modeling_env_win.yml`\
or\
`conda env create --file=modeling_env_mac.yml`\
depending on your opperating system and set the modeling environment as the kernel for the Jupyter notebook.

# Requirements
If you'd prefer to create your own environment, here is a list of known dependencies:
1. numpy
2. matplotlib
3. pandas
4. scipy
5. seaborn
6. statsmodel
7. scikit-learn
8. openpyxl
9. boruta_py

# Usage
The full linear modeling workflow can be run using the NAME_CHANGED.ipynb notebook. Input data should be stored in the InputData folder and formatted similarly to the example file NAME_CHANGED.xlsx with a row of x# identifiers, a row of parameter names, and the parameter values. Experimental outputs can be in the same or different sheet.

For cross-term generation or Boruta feature selection, run the feature_curation.ipynb notebook, which outputs an excel sheet that can be fed into the modeling script with minor modeification explained in the final cell.