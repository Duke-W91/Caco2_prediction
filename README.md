# ADMET Evaluation in Drug Discovery. 21. Application and Industrial Validation of Machine Learning Algorithms for Caco-2 Permeability Prediction
![image](https://github.com/Duke-W91/Caco2_prediction/blob/main/img/Graphical%20Abstract.png)

## Data

We collected experimental values of Caco-2 permeability from three publicly available datasets, resulting in 7861 compounds being included in this research. After data curation, a total of 5654 compounds with structural diversity were obtained.
We additionally collected 271 compounds reported in 2022 and 2023 from the ChEMBL database as the external set (please see the jupyter notebook contained in the folder).
In curated_caco2_data folder,  the results of the MMP analysis were also included.

## Model

*descriptor_model: contain the codes for training four descriptor-based models (XGBoost, SVM, GB, and RF).
*combinednet and dmpnn: contain the codes for training the CombinedNet and DMPNN.
*saved_models: contain the best-performing models obtained in this research.


## Requirements

* python                    3.7.12
* torch                     1.13.1
* numpy                     1.21.6 
* pandas                    1.2.3
* xgboost                   1.0.2
* scikit-learn              0.21.3
* mmpdb                     2.1
* rdkit                     2023.3.1
