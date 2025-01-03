# Discret2DeepDive
## Extracting Knowledge using Machine Learning for Anomaly Detection and Root-Cause Diagnosis

## First steps 
At the start, take a look at the instructions in the notebook file named "prepare_datasets.ipynb". 
Once executed, you can access the preprocessed datasets for training the model and evaluating our approach. 
Also, make sure to run the "tank_generation.ipynb" file to create the simulated Three-Tank dataset.

The resources for finding the datasets are as follows: 
* BeRfiPl Dataset - Benchmark for diagnosis, reconfiguration and planning (https://github.com/j-ehrhardt/benchmark-for-diagnosis-reconf-planning), where our dataset is generated via the Data Creation repository: https://github.com/PhillipOverloeper/DiscretizationOfTimeSeries/tree/main/Berfipl%20Data%20Creation 
* Siemens SmartAutomation SmA proceess plant: https://github.com/thomasbierweiler/FaultsOf4-TankBatchProcess
* Three Tank Simulation Dataset - Simulation out of the paper "Learning Physical Concepts in CPS: A Case Study with a Three-Tank System": https://www.sciencedirect.com/science/article/pii/S2405896322004840



## Model Training of the CatVAE and sequential RNN-CatVAE
The parameters can be found in the folder /VAE_training_hparams/ for every single evaluation dataset. <br>

## Evaluation 
Within the following folders, the evaluation of the specific sections of the paper can be reviewed. <br> 

**Evaluation of CatVAE** (folder path: notebooks_evaluation_CatVAE): Discretisations can be carried out in the notebooks with the pre-trained models and compared on the basis of the plots.

**Evaluation of Discret2DeepDive** (folder path: notebooks_evaluation_discret2deepdive): 
Discretisations can be carried out in the notebooks with the pre-trained models and compared on the basis of the plots. <br>
Based on the simulated data set of the three-tank model, various anomalies were simulated as described in the preprocessing step. <br>
The notebook "04_diagnosis_tank.ipynb" can be executed, in which the various diagnoses are carried out.
