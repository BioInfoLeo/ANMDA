# ANMDA

## Introduction

ANMDA is an anti-noise based computational model for predicting potential miRNA-disease associations. 

In order to resist the noise hiding in the data, ANMDA uses k-means to pick negative samples, subsamples for noise smoothing and applies light gradient boosting machine to predict miRNA-disease associations.

## Preparation

Python 3.5(Anaconda)

## Data & code

Step1. Construct the features of miRNAs and diseases.

Step2. Construct the data by positive and negative samples(use k-means).

Step3. Construct the ANMDA and use cross validation to get the AUROC.

Step4. Adjust the hyper-parameters of ANMDA.

Step5. Use the ANMDA on cases.

Input data in Step1 are in input_data.zip.
Output data in Step2, i.e.,input data in Step3 is in data.zip.

The code of Step1 & Step2 is consturct_data.py.
The code of Step3 & Step4 is test.py.
The code of Step5 is case.py.

The result of 100 times 5-fold cross validation is result.csv.

## Other data & code

In noisy_data_test, data_clean.csv and data_noisy.csv which are extracted from raw_data.csv are used to show the noise hiding in the data. 

The code is noisy_data_test.py.
