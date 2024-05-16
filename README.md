# MPOS-RVFL: Imbalanced Real-time Fault Diagnosis Based on Minority-Prioritized Online Semi-supervised Random Vector Functional Link Network

## Overview
Industrial real-time fault diagnosis is vital to ensure efficient and safe production. In the literature, existing methods usually do not systematically consider some realistic constraints in dealing with the above problem, such as real-time model update, sample imbalance, and high cost of labeling. In this paper, a minority-prioritized online semi-supervised random vector functional link network approach, termed MPOS-RVFL, is proposed to cope with the above issues. Specifically, the pseudo-labeling technique is introduced to fully exploit the information from unlabeled samples in the online data stream. In this context, the approach incorporating minority anchors prioritization, minority weight, and pseudo-label is developed to enhance the model's capability in accurately identifying minority samples. Several experiments with a real-world gearbox fault dataset are conducted to verify the practicality of MPOS-RVFL. The results demonstrate that the proposed method outperforms the existing state-of-the-art approaches.

## The flow chart of MPOS-RVFL fault diagnosis scheme
![image](https://github.com/Jay0804/MPOS-RVFL/assets/114797941/c256d785-58ad-4ce4-ac25-a61735499422)

## Usage
This repository includes four main files:
1. **clf_MPOSRVFL.py**: This file contains the implementation of the Minority-Prioritized Online Semi-supervised Random Vector Functional Link Network (MPOS-RVFL). 
2. **utils.py**: This file contains the parameter settings for MPOS-RVFL.
3. **main.py**: This is the main file that demonstrates the usage of the proposed method for imbalanced real-time fault diagnosis under different imbalanced ratios.
4. **imbalance_data**: This file contains 19 imbalanced offline data and online data streams with their corresponding real labels. The fault combination is teeth_crack and gear_wear. The full version of the data set used in the paper is available at: https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets

## Partial Experimental Results
![image](https://github.com/Jay0804/MPOS-RVFL/assets/114797941/2fff5e9e-ecc4-4903-a37e-bb893ae6508d)


## Contact
Welcome to communicate with us: 1120200786@bit.edu.cn

## Acknowledgments
We extend our sincere gratitude to our THUFDD Group, led by Prof. Xiao He and Prof. Donghua Zhou, for their invaluable support and contributions to the development of this scheme.

**Disclaimer**: This scheme is provided as-is without any warranty. Use at your own risk.

