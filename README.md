# Paper Supplement: On Feature Removal for Explainability in Dynamic Environments

Supplement Material for this research paper: https://doi.org/10.14428/esann/2023.ES2023-148

The experiments and results presented in the paper are available in the `projects/Removal` module.
The `projects/Removal` module contains the following submodules:

- `agrawal_stream.py`: Script to run the experiments on the agrawal stream.
- `gaussian_stream.py`: Script to run the experiments on the gaussian stream.
- `observational_interventional_agrawal.py`: Script to create the illustrative example of the observational and interventional feature removals.
- `mdi.py`: Module that contains the implementation of the MDI explainer.

The implementation of the explainers can be found in the `iXAI` module. 

## Running the experiments
1. Install requirements by running `pip install -r requirements.txt`.
2. Navigate to iXAI root directory (`cd iXAI`) and install the ixai package by running `pip install .`.
3. Navigate to the `projects/Removal` directory (`cd projects/Removal`) and run the experiments.
