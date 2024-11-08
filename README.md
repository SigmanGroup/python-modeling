# python-modeling
This repository contains the Sigman lab workflow for linear modeling, primarily driven by bidirectional stepwise MLR, as well as threshold analysis and tools for feature curation.

# Installation
Two conda environments (modeling_env.yml and feature_curation_env.yml) are provided for use with their respective notebooks.
They can be installed by running the following code:

  ```bash
  conda env create --file=modeling_env.yml --name=modeling
  ```
  and
  ```bash
  conda env create --file=feature_curation_env.yml --name=feature_curation
  ```

# Usage
The full linear modeling workflow can be run using the Mattlab_modeling_v6.0.0.ipynb notebook. Input data should be stored in the InputData folder and formatted similarly to the example file Multi-Threshold Analysis Data.xlsx with a row of parameter names and the parameter values. Experimental outputs can be in the same sheet or different sheet. There should be nothing to the right of the parameters in the parameters file.

For cross-term generation or Boruta feature selection, run the feature_curation.ipynb notebook, which outputs an excel sheet that can be fed into the modeling script with minor modification explained in the final cell.

In the modeling notebook, most functions can be accessed by running the cells in sequence. If jumping between cells, make sure to run a train/validation/test split cell before running a modeling cell.