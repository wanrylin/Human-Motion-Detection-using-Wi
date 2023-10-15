# Human Motion Detection using Wi-Fi 2022
This is my Master project in NUS.<br>
My superviser is Prof.Cuo Yongxin homepage:https://www.ece.nus.edu.sg/stfpage/eleguoyx/.


## Introduction
In the rapid advancement of commercial Wi-Fi devices, this project explores a Wi-Fi based motion detection method for "through the wall" scenarios. Leveraging channel state information (CSI) changes due to indoor body movements, the proposed system can detect and classify seven unique human motions. It employs a conjugate multiplication algorithm for noise removal, Principal Component Analysis for denoising and feature extraction, and a support vector machine for movement detection. A specially designed channel clean algorithm mitigates multipath interference, while a 1D Convolutional Neural Network classifies motions. Experimentally, the system boasts 98% accuracy in movement detection and over 95% in motion classification.

## Main contribution
(1)To the best of my acknowledgment, this system is the first one to detect 7 human motions in through the wall scenario. <br>
(2)Proposed a new channel clean algorithm to reduce the the multipath interference in through the wall scenario.<br>
(3)Achieve 98% accuracy in movement detection and 95% accuracy in motion classification.<br>

## Overview of the system
<img src="https://github.com/wanrylin/Human-Motion-Detection-using-WiFi/blob/main/figure/Master%20project.png" alt="overview of system" width="800"><br>
In this project, human motion is monitored using Channel-State-Information (CSI). CSI data provides amplitude and phase time sequences for each subcarrier, forming the foundation for an amplitude and phase-based system. A Support-Vector-Machine (SVM) is trained to determine the presence of human movement, while a random forest differentiates walking from other motions. Additionally, two 1-Dimensional Convolutional Neural Networks are utilized: one to identify the direction of movement and another to classify five distinct types of motion. The entire system's workflow is depicted in the figure.

## Condition and assumption
(1)Wireless transmission protocol of the WiFi device: IEEE 802.1n <br>
(2)Operating frequency band: 2.4GHz<br>
(3)Number of transmitting antennas: 2<br>
(4)Number of receiving antennas: 2<br>
(5)Type of antennas: omni antenna<br>
(6)Application scenario: indoor through concrete wall<br>
(7)Sampling rate: 200Hz<br>

## System Design
### 1 CSI data capturing
In this project, the PicoScenes[^1] platform, installed on two Ubuntu desktops, was employed. This platform doesn't support simultaneous monitoring and CSI output. Instead, it records CSI data in .csi files, which are decoded using the PicoScenes Matlab toolbox. For enhanced accessibility, I extracted CSI data and saved it in .csv files. The complex CSI data encompasses amplitude and phase information. Each output package from the PicoScenes platform, saved in a structured format, contains data from two antennas. I initiated the process by extracting this data into a matrix and recording the package sequence, critical due to occasional data packet loss during transmission. The sequence helped identify missing packets, necessitating its preservation for subsequent data interpolation.

### 2 CSI phase denoising


### 3 Movement detection


### 4 Motion seperation


### 5 Motion classification

#### Channel clean

#### Feature process

#### 1DCNN classifier



## Result and conclusion
### Result


### Conclusion




## Reference 
[^1]:Z. Jiang, T. H. Luan, X. Ren, D. Lv, H. Hao, J. Wang, K. Zhao, W. Xi, Y. Xu, and R. Li, “Eliminating the Barriers: Demystifying Wi-Fi Baseband Design and Introducing the PicoScenes Wi-Fi Sensing Platform,” IEEE Internet of Things Journal, pp. 1-1, 2021.
