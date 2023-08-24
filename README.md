# CDNL

## Introduction

<!-- 
This repository contains the source code for the paper
```
Combined-task deep network based on LassoNet feature selection for predicting the comorbidities of acute coronary syndrome
```
-->
CNDL is designed to enhance the accuracy and prediction capability for ACS comorbidities by simultaneously considering multiple tasks and incorporating informative biomarkers into the prediction process.

## Access

CDNL is free for non-commerical use only.

## Requirements

Python 3

PyTorch

scikit-learn

## Usage

The folder './src/featureSelection' contains files related to the identification of crucial biomarkers for comorbidities of acute coronary syndrome (ACS), with the main file being 'coronary_feature_selection.py'.

The folder './src/improved_LBTW' contains files related to the prediction of ACS comorbidities, with the main file being 'pcba_run.py'.


## Data

The data was collected from a cross-sectional study conducted at a tertiary hospital in China in the cardiology department between October 2019 and June 2022. Due to privacy concerns, we are unable to disclose the raw data. We have included sample data in the "toyData" folder for demonstration purposes.

## Reference

[1] I. Lemhadri, F. Ruan, L. Abraham, R. Tibshirani, Lassonet: A neural network with feature sparsity, The Journal of Machine Learning Re- search 22 (1) (2021) 5633–5661.

[2] S. Liu, Y. Liang, A. Gitter, Loss-balanced task weighting to reduce negative transfer in multi-task learning, in: Proceedings of the AAAI conference on artificial intelligence, Vol. 33, 2019, pp. 9977–9978.


<!-- 
## Cite

This project is developed based on Non-NIID, if you find this repository useful, please cite paper:

```
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
```
-->

## Developer

Xiaolu Xu 

lu.xu@lnnu.edu.cn

Liaoning Normal University.