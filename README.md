
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
This project reveals that denoised CSI phase data is more stable and less noise-sensitive than amplitude, making it preferable for movement detection. However, as discussed in chapter 4.1, network cards induce a random phase offset in CSI data, overshadowing phase changes caused by human movement. Previous research cited as Indotrack[^2] introduced an algorithm to counteract this randomness.<br>
Commodity Wi-Fi devices lack tight time synchronization between receivers and transmitters, leading to a random phase offset, $e^{j\theta_{offset}}$, in each CSI sample as shown in equation:
```math
\begin{split}
x(f,t_0 + t) &= e^{j\theta_{offset}}(\sum_{m=1}^{M}A_me^{-j2\pi f \tau_m} +A_p(t)e^{-j2\pi f_p \tau_p} )\\
& = e^{j\theta_{offset}}(x_s(f,t_0) + A_p(t)e^{-j2\pi f_p \tau_p} )
\end{split}
```
This scenario complicates the observation of the Doppler effect. To address these issues, the following steps are proposed:
<b>1 Conjugate multiplication:</b> Utilizing the fact that Wi-Fi cards' antennas share the same RF oscillator, conjugate multiplication between two antennas' CSI is possible, effectively eliminating random phase offsets as per equation:
```math
\begin{split}
x_{cm}(f,t_0 + t) &= x_1(f,t_0 + t)\bar{x}_2(f,t_0 + t)\\
&=(x_{1,s}(f,t_0) + A_{1,p}(t)e^{-j2\pi f_{1,p} \tau_{1,p}} )(\bar{x}_{2,s}(f,t_0) + A_{2,p}(t)e^{j2\pi f_{2,p} \tau_{2,p}} )\\
& = \underbrace{x_{1,s}(f,t_0)\bar{x}_{2,s}(f,t_0)}_{(1)} + \underbrace{x_{1,s}(f,t_0)A_{2,p}(t)e^{j2\pi f_{2,p} \tau_{2,p}}}_{(2)} \\
&+ \underbrace{\bar{x}_{2,s}(f,t_0)A_{1,p}(t)e^{-j2\pi f_{1,p} \tau_{1,p}}}_{(3)} + 
\underbrace{A_{1,p}(t)e^{-j2\pi f_{1,p} \tau_{1,p}}A_{2,p}(t)e^{j2\pi f_{2,p} \tau_{2,p}}}_{(4)} 
\end{split}
```
<b>2 Remove static component:</b> The static-path component product (1) is considered constant over short periods and is subtracted from the conjugate multiplication to prevent interference with Doppler monitoring.<br>
<b>3 Adjust the power of each antenna:</b> By modifying the power of static components for each antenna (subtracting $\alpha$ and adding $\beta$), the Doppler effect information is preserved, and the correct phase becomes prominent in the spectrum, as equation explains:
```math
\begin{split}
&x_{cm}(f,t_0 + t) - x_{1,s}(f,t_0)\bar{x}_{2,s}(f,t_0) = x_1(f,t_0 + t)\bar{x}_2(f,t_0 + t) - x_{1,s}(f,t_0)\bar{x}_{2,s}(f,t_0)\\
&\approx (x_{1,s}(f,t_0) - \alpha)A_{2,p}(t)e^{j2\pi f_{2,p} \tau_{2,p}} +
(\bar{x}_{2,s}(f,t_0) + \beta)A_{1,p}(t)e^{-j2\pi f_{1,p} \tau_{1,p}}\\
&\approx (\bar{x}_{2,s}(f,t_0) + \beta)A_{1,p}(t)e^{-j2\pi f_{1,p} \tau_{1,p}}
\end{split}
```
These strategies effectively remove random CSI phase offsets. The denoising algorithm's efficacy is evident when comparing pre- and post-denoising CSI data, as illustrated in following figure.
<p float="left">
  <img src="https://github.com/wanrylin/Human-Motion-Detection-using-WiFi/blob/main/figure/phase%20correction0.png" width="300" />
  <img src="https://github.com/wanrylin/Human-Motion-Detection-using-WiFi/blob/main/figure/phase%20correction1.png" width="300" /> 
</p>
This denoising algorithm successfully eliminates random phase offsets caused by the network card.

### 3 Movement detection
After denoising and segmenting, CSI phase data from Receiver 1 (Rx1) is processed, adhering to the IEEE 802.11n protocol, which includes 57 subcarriers each with unique CSI phase data. The experiment reveals that each subcarrier's sensitivity to human movement varies due to frequency differences. Some subcarriers are highly sensitive to movement but are also prone to noise interference, while others show less sensitivity to both movement and noise.

Previous research indicates that in the absence of human movement, CSI values across subcarriers appear random and uncorrelated. However, when human movement occurs, these values become strongly correlated. The amplitude of CSI measurements varies across subcarriers, influencing their sensitivity in detecting human motion. Although there's a correlation in variations across all subcarriers, the degree of this correlation varies.

To extract key features and minimize noise, I integrated all subcarriers using Principal Component Analysis (PCA), a method for transforming high-dimensional data into a lower-dimensional format while preserving most variability. The first 10 PCA components, representing over 99% of the dataset's characteristics, are retained for feature extraction. This approach also reveals the relationships between subcarriers. The CSI phase data, initially a $200 \times 57$ matrix, is dimensionally reduced to a $200 \times 10$ matrix through PCA. Following prior work, the first PCA component, typically noise-dominant, is excluded. Instead, the mean and variance of the first 10 PCA components are calculated and used as inputs for the Support Vector Machine (SVM). This method effectively illustrates the data's directional trends and core structures.

Additionally, the static environment in the study encompasses two scenarios: an empty sensing area or a non-moving individual within it. To account for both, the training dataset includes three situations, labeling only significant human movements as positive. This augmented dataset, which incorporates static human presence, aids the SVM in better distinguishing significant human movements.

Classification involves two stages: training and testing. In the training stage, a set of labeled samples is used to establish models for classification. Each sample in the training set, represented by a pair $(r_i,c_i)$, consists of a feature vector $r_i=(r_{i1},r_{i2},\ldots,r_{il})$ and a class label $c_i \in \{1, 0\}$ indicating the presence or absence of human movement. These samples are utilized to train Support Vector Machine (SVM) classifiers. In the testing phase, the classifier is presented with an unlabeled sample $(r, c)$, where $r \in \mathbb{R}^l$ is the CSI feature vector, and the goal is to ascertain the label $c$, determining human movement in the sensing area.

Prior to classification, normalizing the CSI features is crucial for enhancing accuracy and convergence speed. Normalization of each feature is achieved by calculating the minimum $r_{\text{min}, j} = \min_i r_{ij}$ and maximum $r_{\text{max}, j} = \max_i r_{ij}$ values for the $j$-th feature. The normalized value of $r_{ij}$ is then computed as follows:
```math
\begin{equation}
r_{ij} = \frac{r_{ij} - r_{\text{min}, j}}{r_{\text{max}, j} - r_{\text{min}, j}}
\end{equation}
```

For classification, C-Support Vector Classification (C-SVC) is adopted, which solves the optimization problem:
```math
\begin{equation}
\min \quad \frac{1}{2}\|\omega\|^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.} \quad \begin{cases}
c_i(\omega^T r_i + b) \geq 1 - \xi_i,\\
C > 0, \xi_i \geq 0,
\end{cases}
\end{equation}
```
where $\omega \in \mathbb{R}^l$ is the orientation vector of the separating hyperplane, $b \in \mathbb{R}$ its offset, and $C$ a regularization parameter. The $\xi_i$'s are slack variables allowing misclassification. The classification function is then given by:
```math
\begin{equation}
f(r) = \text{sign}\left(\sum_{i=1}^{n}c_i\alpha_i K(r_i,r) + b\right),
\end{equation}
```
where $\alpha_i$ are Lagrange multipliers and $K(r_i, r)$ is the kernel function. The Radial Basis Function (RBF) is chosen as the kernel:
```math
\begin{equation}
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right),
\end{equation}
```
with $\gamma > 0$ as the kernel parameter. A positive $f(r)$ indicates human presence, while a non-positive value implies absence. For movement detection, a binary classification using C-SVC is employed, requiring tuning of $C$ and $\gamma$ through grid searching and cross-validation. In this system, $C$ is set to 0.1.


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
[^2]:X. Li, D. Zhang, Q. Lv, J. Xiong, S. Li, Y. Zhang, and H. Mei, “Indotrack: Device-free indoor human tracking with commodity wi-fi”, Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 1, no. 3, pp. 1–22, 2017.
