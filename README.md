# python-modeling
This repository contains the Sigman lab workflow for linear modeling, primarily driven by bidirectional stepwise MLR, as well as tools for feature curation.

# Installation
A conda environment is provided in the environment.yml files. To create the environment, run

  ```bash
  conda env create --file=environment.yml --name=modeling
  ```

# Requirements
If you'd prefer to create your own environment, here is a list of known dependencies:
  - numpy=1.26.4
  - matplotlib=3.8.4
  - pandas=2.2.2
  - scipy=1.13.1
  - python=3.12.4
  - seaborn=0.13.2
  - statsmodels=0.14.2
  - scikit-learn=1.4.2
  - openpyxl=3.1.2
  - boruta_py=0.3
  - ipykernel=6.19.2

# Usage
The full linear modeling workflow can be run using the Mattlab_modeling_v6.0.0.ipynb notebook. Input data should be stored in the InputData folder and formatted similarly to the example file Multi-Threshold Analysis Data.xlsx with a row of x# identifiers, a row of parameter names, and the parameter values. Experimental outputs can be in the same or different sheet.

For cross-term generation or Boruta feature selection, run the feature_curation.ipynb notebook, which outputs an excel sheet that can be fed into the modeling script with minor modeification explained in the final cell.
