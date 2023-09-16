<!--A curated list of resources for Image and Video Deblurring-->
<!-- PROJECT LOGO -->

<p align="center">
  <h2 align="center">Awesome-Deep-Learning-Technologies-in-Events</h2>
  <p align="center">A curated list of resources for deep learning with events camera.
    <br />
    <br />
    <br />
    <a href="https://github.com/yunfanLu/Awesome-Event-Guided-Enhancement/pulls/new">Suggest new item</a>
    <br />
    <a href="https://github.com/yunfanLu/Awesome-Event-Guided-Enhancement/issues/new">Report Bug</a>
  </p>
</p>

## DDL Time:

2023-01-15

<h2>Table of contents</h2>

- [1  Event Representation](#1--event-representation)
  - [1.0 Event Streaming](#10-event-streaming)
  - [1.1 Event Count Image](#11-event-count-image)
  - [1.2 Event Voxel Grids](#12-event-voxel-grids)
  - [1.3 Surface of Active Event (SAE)](#13-surface-of-active-event-sae)
  - [1.4 Histograms of Averaged Time Surfaces (HATS)](#14-histograms-of-averaged-time-surfaces-hats)
  - [1.5 MatrixLSTM](#15-matrixlstm)
  - [1.6 Event Spike Tensor](#16-event-spike-tensor)
  - [1.7 Binary Event Image](#17-binary-event-image)
  - [1.8 Event Histogram](#18-event-histogram)
  - [1.9 Event Image](#19-event-image)
  - [1.10 Timestamp Image](#110-timestamp-image)
  - [1.11 DiT \& DiSR](#111-dit--disr)
  - [Open Sources Projects](#open-sources-projects)
- [2 Event Only](#2-event-only)
  - [2.1 Image / Video Reconstruction](#21-image--video-reconstruction)
    - [(1) Reconstruction and SR](#1-reconstruction-and-sr)
  - [2.2 Optical Flow Estimation](#22-optical-flow-estimation)
  - [2.3 Classification](#23-classification)
  - [2.4 Detection](#24-detection)
  - [2.5 Segmentation](#25-segmentation)
- [3 Event + RGB](#3-event--rgb)
  - [3.1 Event Guided Image / Video SR](#31-event-guided-image--video-sr)
  - [3.2 Event Guided Image / Video Deblurring](#32-event-guided-image--video-deblurring)
  - [3.3 Event Guided Image / Video HDR](#33-event-guided-image--video-hdr)
    - [(1) Low Light](#1-low-light)
  - [3.4 Event Guided Image / Video Denoising](#34-event-guided-image--video-denoising)
  - [3.5 Event Guided Video Interpolation](#35-event-guided-video-interpolation)
  - [3.6 Depth estimation](#36-depth-estimation)
- [4 Domain Specific](#4-domain-specific)
  - [4.1 NeRF \& 3D reconstruction](#41-nerf--3d-reconstruction)
  - [4.2 Human Pose and Shape](#42-human-pose-and-shape)
  - [4.3 Body and Eye Tracking](#43-body-and-eye-tracking)
  - [4.4 Face](#44-face)
  - [4.5 Compression](#45-compression)
  - [4.4 SAI](#44-sai)
- [5 Robotic Vision](#5-robotic-vision)
  - [5.1 Object Detection and Tracking](#51-object-detection-and-tracking)
  - [5.2 Simultaneous Localization and Mapping (SLAM)](#52-simultaneous-localization-and-mapping-slam)
    - [5.2.1 Front-End of SLAM](#521-front-end-of-slam)
      - [5.2.1.1 Feature Detection for Tracking](#5211-feature-detection-for-tracking)
      - [5.2.1.2 Optical Flow for Tracking](#5212-optical-flow-for-tracking)
    - [5.2.2 Back-End of SLAM](#522-back-end-of-slam)
      - [5.2.2.1 Ego-motion estimation](#5221-ego-motion-estimation)
      - [5.2.2.2 3D Reconstruction](#5222-3d-reconstruction)
    - [5.2.3 End-to-End SLAM](#523-end-to-end-slam)
  - [5.3 Multi-sensor](#53-multi-sensor)
- [6 New Direction](#6-new-direction)
- [7 Discussion](#7-discussion)
- [8 Conclusion](#8-conclusion)
- [Supplementary Material](#supplementary-material)
  - [Event Dataset](#event-dataset)
  - [Simulation](#simulation)


# 1  Event Representation
## 1.0 Event Streaming

## 1.1 Event Count Image
| Publication | Title                                                                                          | Highlight                                                                      |
| ----------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| IJCV 2021   | Learning to Reconstruct HDR Images from Events, with Applications to Depth and Flow Prediction | stacking based on time (SBT) and stacking based on the number of events (SBN). |
| IROS 2018   | The multi vehicle stereo event camera dataset: An event camera dataset for 3D perception       | Per polarity sum                                                               |
## 1.2 Event Voxel Grids
| Publication | Title                                                                                                  | Highlight |
| ----------- | ------------------------------------------------------------------------------------------------------ | --------- |
| T-PAMA 2020 | Time-Ordered Recent Event (TORE) Volumes for Event Cameras                                             |           |
| NC 2019     | Asynchronous Event-Based Motion Processing: From Visual Events to Probabilistic Sensory Representation |           |
| ICCV 2019   | Unsupervised event-based learning of optical ﬂow, depth, and egomotion                                 |           |
| CVPR 2019   | Eventsto-video: Bringing modern computer vision to event cameras                                       |           |


## 1.3 Surface of Active Event (SAE)
| Publication             | Title                                                                                                                   | Highlight                                                                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| T-RAL 2020              | Efficient Spatial-Temporal Normalization of SAE Representation for Event Camera                                         |                                                                                        |
| CVPR 2019               | Speed invariant time surface for learning to detect corner points with event-based cameras                              | Speed Invariant Time Surface + SAE                                                     |
| ISACS 2019              | N-HAR: A neuromorphic event-based human activity recognition system using memory surfaces                               | Memory Surface                                                                         |
| Neurosci 2019           | Investigation of event-based surfaces for high-speed detection, unsupervised feature extraction, and object recognition | Time-Window Normalization + SAE, Time-surfaces                                         |
| 3DV 2018                | ACE: An efficient asynchronous corner tracker for event cameras                                                         | Sort Normalization + SAE, Min-Max Normalization + SAE                                  |
| T-PAMI 2017             | Hots: A hierarchy of event-based time-surfaces for pattern recognition                                                  | Exponential Decay Normalization + SAE, Time-surfaces, Exponential of newest timestamps |
| Ph.D. dissertation 2016 | Event-based feature detection, recognition and classification                                                           | Exponential Decay Normalization + SAE, Time-surfaces                                   |
| T-NLL 2014              | The event-based visual ﬂow                                                                                              | First work                                                                             |

## 1.4 Histograms of Averaged Time Surfaces (HATS)
| Publication | Title                                                                                   | Highlight                                |
| ----------- | --------------------------------------------------------------------------------------- | ---------------------------------------- |
| CVPR 2018   | Hats: Histograms of averaged time surfaces for robust event-based object classification | First Work, Aggregated newest timestamps |

## 1.5 MatrixLSTM
| Publication | Title                                                                | Highlight         |
| ----------- | -------------------------------------------------------------------- | ----------------- |
| ECCV 2020   | A differentiable recurrent surface for asynchronous event-based data | Learned with LSTM |

## 1.6 Event Spike Tensor
| Publication | Title                                                                    | Highlight        |
| ----------- | ------------------------------------------------------------------------ | ---------------- |
| ICCV 2019   | End-to-end learning of representations for asynchronous event-based data | Learned with MLP |

## 1.7 Binary Event Image
| Publication | Title                                                                  | Highlight                 |
| ----------- | ---------------------------------------------------------------------- | ------------------------- |
| T-NNLS 2018 | Spatial and temporal downsampling in event-based visual classification | Binarized event occurence |

## 1.8 Event Histogram
| Publication | Title                                                                                | Highlight    |
| ----------- | ------------------------------------------------------------------------------------ | ------------ |
| CVPR 2018   | Event-based vision meets deep learning on steering prediction for self-driving cars. | Event counts |

## 1.9 Event Image
| Publication | Title                                                            | Highlight                          |
| ----------- | ---------------------------------------------------------------- | ---------------------------------- |
| CVPR 2019   | Event-based robust gait recognition using dynamic vision sensors | Event counts and newest timestamps |

## 1.10 Timestamp Image

| Publication | Title                                                                                                         | Highlight         |
| ----------- | ------------------------------------------------------------------------------------------------------------- | ----------------- |
| ICIP 2020   | Performance improvement of deep learning based gesture recognition using spatiotemporal demosaicing technique | Newest timestamps |

## 1.11 DiT & DiSR
| Publication | Title                                                                          | Highlight         |
| ----------- | ------------------------------------------------------------------------------ | ----------------- |
| ICCV 2021   | N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras | Dataset Image Net |

## Open Sources Projects

| Project | URL                              | Description                                                        |
| ------- | -------------------------------- | ------------------------------------------------------------------ |
| Chain   | https://github.com/eleboss/chain | The effect about different representation for event classification |

# 2 Event Only

## 2.1 Image / Video Reconstruction
| Publication | Title                                                                                                                                              | Highlight |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| CVPR 2023   | EvShutter: Transforming Events for Unconstrained Rolling Shutter Correction |      🆕     |
| CVPR 2023   | Learning Event Guided High Dynamic Range Video Reconstruction |      🆕     |
| CVPR 2021   | Image Reconstruction From Neuromorphic Event Cameras Using Laplacian-Prediction and Poisson Integration With Spiking and Artificial Neural Networks |           |
| CVPR 2021   | Event-Based Synthetic Aperture Imaging With a Hybrid Network                                                                                       |           |
| CVPR 2021   | Learning To Reconstruct High Speed and High Dynamic Range Videos From Events                                                                       |           |
| CVPR 2021   | Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy                                 |           |
| IJCV 2021   | Learning to Reconstruct HDR Images from Events, with Applications to Depth and Flow Prediction                                                     |           |
| ICCP 2021   | EventGAN: Leveraging Large Scale Image Datasets for Event Cameras                                                                                  |           |
| ECCV 2020   | Reducing the Sim-to-Real Gap for Event Cameras                                                                                                     |           |
| ECCV 2020   | Event Enhanced High-Quality Image Recovery                                                                                                         |           |
| WACV 2020   | Fast Image Reconstruction with an Event Camera                                                                                                     |           |
| T-PAMI 2020 | High Frame Rate Video Reconstruction Based on an Event Camera                                                                                      |           |
| T-PAMI 2019 | High Speed and High Dynamic Range Video with an Event Camera                                                                                       |           |
| CVPR 2019   | Events-to-Video: Bringing Modern Computer Vision to Event Cameras                                                                                  |           |
| IJCV 2018   | Real-Time Intensity-Image Reconstruction for Event Cameras Using Manifold Regularisation                                                           |           |

### (1) Reconstruction and SR
| Publication         | Title                                                                                                                            | Highlight                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| CVPR 2023           | High-Fidelity Event-Radiance Recovery via Transient Event Frequency                                                              | High-fidelity Event-Radiance Recovery 🆕     |
| ICCV 2021           | EvIntSR-Net: Event Guided Multiple Latent Frames Reconstruction and Super-resolution                                             | Latent frame reconstruction; MISR            |
| ECCV 2020           | Event Enhanced High-Quality Image Recovery                                                                                       | eSL-Net; Sparse learning                     |
| CVPR 2020           | Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging                             | GEF; Joint filtering                         |
| CVPR 2021           | Turning Frequency to Resolution: Video Super-resolution via Event Cameras                                                        | E-VSR; frequency                             |
| ICCV 2021           | EvIntSR-Net: Event Guided Multiple Latent Frames Reconstruction and Super-resolution                                             |                                              |
| CVPR 2021           | EventZoom: Learning To Denoise and Super Resolve Neuromorphic Events                                                             | Denoise                                      |
| CVPR 2021           | Turning Frequency to Resolution: Video Super-Resolution via Event Camera                                                         | Video SR + RGB                               |
| CVPR 2020           | Learning to Super Resolve Intensity Images from Events                                                                           |                                              |
| CVPR 2020           | EventSR: From Asynchronous Events to Image Reconstruction, Restoration, and Super-Resolution via End-to-End Adversarial Learning | EventSR; End2end; Adversarial learning + RGB |
| TAPMI 2020          | E2SRI: Learning to Super-Resolve Intensity Images from Events                                                                    | SR-Net; RCNN + RGB                           |
| Neurocomputing 2019 | Super-resolution of spatiotemporal event-stream image                                                                            |                                              |


## 2.2 Optical Flow Estimation


| Publication | Title                                                                                                                               | Highlight |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------- |
| CVPR 2021   | Spike Timing-Based Unsupervised Learning of Orientation, Disparity, and Motion Representations in a Spiking Neural Network          |           |
| Arxiv 2021  | Formulating Event-based Image Reconstruction as a Linear Inverse Problem using Optical Flow                                         |           |
| ECCV 2020   | Jointly learning visual motion and confidence from local patches in event cameras                                                   |           |
| T-PAMI 2020 | Real-time high speed motion prediction using fast aperture-robust event-driven visual flow                                          |           |
| CVPR 2020   | SOFEA: A Non-Iterative and Robust Optical Flow Estimation Algorithm for Dynamic Vision Sensors                                      |           |
| CVPR 2020   | Single Image Optical Flow Estimation with an Event Camera                                                                           |           |
| ECCV 2020   | Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks                                     |           |
| 3DV 2021    | E-RAFT: Dense Optical Flow from Event Cameras                                                                                       |           |
| T-PAMI 2020 | Distance Surface for Event-Based Optical Flow                                                                                       |           |
| IROS 2020   | Unsupervised Learning of Dense Optical Flow, Depth and Egomotion with Event-Based Sensors                                           |           |
| CVPR 2020   | Single Image Optical Flow Estimation With an Event Camera                                                                           |           |
| T-PAMI 2019 | Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception |           |
| CVPR 2019   | Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion                                                             |           |
| CVPR 2019   | Secrets of Event-based Optical Flow                                                                                                 |           |
| CVPR 2019   | Event-Based Motion Segmentation by Motion Compensation                                                                              |           |
| CVPR 2019   | Event Cameras, Contrast Maximization and Reward Functions: An Analysis                                                              |           |
| ICCV 2019   | End-to-End Learning of Representations for Asynchronous Event-Based Data                                                            |           |
| BMVC 2018   | Adaptive Time-Slice Block-Matching Optical Flow Algorithm for Dynamic Vision Sensors                                                |           |
| RSS 2018    | EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras                                                         |           |
| CVPR 2018   | A Unifying Contrast Maximization Framework for Event Cameras, with Applications to Motion, Depth and Optical Flow Estimation        |           |
| T-BCS 2018  | Spiking Optical Flow for Event-based Sensors Using IBM's TrueNorth Neurosynaptic System                                             |           |
| IEEE 2017   | Event-based real-time optical flow estimation                                                                                       |           |
| CVPR 2016   | Simultaneous Optical Flow and Intensity Estimation From an Event Camera                                                             |           |




## 2.3 Classification


| Publication | Title                                                                            | Highlight |
| ----------- | -------------------------------------------------------------------------------- | --------- |
| T-PAMI 2022 | Time-Ordered Recent Event (TORE) Volumes for Event Cameras |  |
| Arxiv 2022  | Event Transformer                 |  |
| Arxiv 2022  | Exploiting Spatial Sparsity for Event Cameras with Visual Transformers                   |  |
| RA-L 2022   | VMV-GCN: Volumetric Multi-View Based Graph CNN for Event Stream Classificati                   |  |
| CVPR 2022   | A Voxel Graph CNN for Object Classification with Event Cameras                   |           |
| CVPR 2022   | Ev-TTA: Test-Time Adaptation for Event-Based Object Recognition                  |           |
| CVPR 2022   | AEGNN: Asynchronous Event-based Graph Neural Networks                            |           |
| Frontiers in Neuroscience 2021 | ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks |  |
| T-IP 2021   |  Learning from Images: A Distillation Learning Framework for Event Cameras      |           |
| T-CSVT 2021   |  MVF-Net: A Multi-view Fusion Network for Event-based Object Classification      |           |
| CVPRW 2021   |   V2e: From video frames to realistic DVS events         |           |
| ICCV 2021   | Temporal-wise Attention Spiking Neural Networks for Event Streams Classification |           |
| RA-L 2021   | DA4Event: Towards Bridging the Sim-to-Real Gap for Event Cameras Using Domain Adaptation |           |
| ICCV 2021   | N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras   |           |
| ICCV 2021   | Graph-based Asynchronous Event Processing for Rapid Object Recognition |           |
| ECCV 2020  | A Differentiable Recurrent Surface for Asynchronous Event-Based Data  |   |
| CVPR 2020  | Video to Events: Recycling Video Datasets for Event Cameras  |   |
| ECCV 2020   | Event-based Asynchronous Sparse Convolutional Networks  |      |
| T-PAMI 2019 | High Speed and High Dynamic Range Video with an Event Camera |  |
| T-PAMI 2019 | DART: Distribution aware retinal transform for event-based cameras |  |
| CVPR 2019 | Events-to-video: Bringing modern computer vision to event cameras |  |
| ICCV 2019 | Graph-based object classification for neuromorphic vision sensing |  |
| ICCV 2019  | End-to-End Learning of Representations for Asynchronous Event-Based Data |    |
| CVPR 2018 | HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification |  |

## 2.4 Detection
| Publication | Title                                                                                                             | Highlight |
| ----------- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| ICRA 2023    | RGB-Event Fusion for Moving Object Detection in Autonomous Driving                                               |    🆕     |
| TIM 2023    | Motion Robust High-Speed Light-Weighted Object Detection With Event Camera                                        |    🆕     |
| CVPR 2023   | Recurrent Vision Transformers for Object Detection With Event Cameras                                             |    🆕     |
| CVPR 2023   | Hierarchical Neural Memory Network for Low Latency Event Processing                                               |    🆕     |
| IROS 2018   | Towards Event-Driven Object Detection with Off-the-Shelf Deep Learning                                            |           |
| CVPR 2018   | Pseudo-Labels for Supervised Learning on Dynamic Vision Sensor Data, Applied to Object Detection Under Ego-Motion |           |
| ACCVW 2018  | PCA-RECT: An Energy-Efficient Object Detection Approach for Event Cameras                                         |           |
| IROA 2018   | Event-Based Moving Object Detection and Tracking                                                                  |           |
| TPAMI 2019  | DART: Distribution Aware Retinal Transform for Event-Based Cameras                                                |           |
| ICME 2019   | Event-Based Vision Enhanced: A Joint Detection Framework in Autonomous Driving                                    |           |
| CVPRW 2019  | Asynchronous Convolutional Networks for Object Detection in Neuromorphic Cameras                                  |           |
| ECCV 2020   | Event-Based Asynchronous Sparse Convolutional Networks                                                            |           |
| NIPS 2020   | Learning to Detect Objects with a 1 Megapixel Event Camera                                                        |           |
| ICIP 2021   | An Attention Fusion Network For Event-Based Vehicle Object Detection                                              |           |
| TIP  2021   | Asynchronous Spatio-Temporal Memory Network for Continuous Event-Based Object Detection                           |           |
| ICCVW 2021  | Moving Object Detection for Event-Based Vision Using Graph Spectral Clustering                                    |           |
| WACV 2022   | Learned Event-based Visual Perception for Improved Space Object Detection                                         |           |
| ICRA 2022   | Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions                               |           |
| MFI 2022    | Global-local Feature Aggregation for Event-based Object Detection on EventKITTI                                   |           |
| Access 2021 | Detection of Binary Square Fiducial Markers Using an Event Camera                                                 |           |

## 2.5 Segmentation
| Publication | Title                                                                                                                          | Highlight |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------ | --------- |
| TIP 2023   | Event-Based Semantic Segmentation With Posterior Attention                                                                     |    🆕     |
| CVPR 2023   | Hierarchical Neural Memory Network for Low Latency Event Processing                                                            |    🆕     |
| CVPRW 2019  | EV-SegNet: Semantic Segmentation for Event-Based Cameras                                                                       |           |
| ICDSP 2021  | VESS: Variable Event Stream Structure for Event-based Instance Segmentation Benchmark                                          |
| IROS 2021   | ISSAFE: Improving Semantic Segmentation in Accidents by Fusing Event-based Data                                                |            |
| ICNOS 2021  | Superevents: Towards Native Semantic Segmentation for Event-based Cameras                                                      |
| CVPR 2021   | Dual Transfer Learning for Event-Based End-Task Prediction via Pluggable Event to Image Translation                            |            |
| CVPRW 2021  | SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos                                   |            |
| CVPR 2021   | EvDistill: Asynchronous Events to End-task Learning via Bidirectional Reconstruction-guided Cross-modal Knowledge Distillation |            |
| TPAMI 2022  | Globally-Optimal Contrast Maximisation for Event Cameras                                                                       |            |
| IPMV 2022   | Event Camera Survey and Extension Application to Semantic Segmentation                                                         |            |
| ECCV 2022   | ESS: Learning Event-Based Semantic Segmentation from Still Images                                                              |            |
| TITS 2022   | Exploring Event-Driven Dynamic Context for Accident Scene Segmentation                                                         |            |
| NCS  2022   | Beyond Classification: Directly Training Spiking Neural Networks for Semantic Segmentation                                     |            |

## Motion Segmentation
| Publication | Title                                                                                                                          | Highlight |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------ | --------- |
| IROS  2018  | Event-Based Moving Object Detection and Tracking                                                                               |       |
| CVPR  2018  |  A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation |       | 
| IROS  2019  | EV-IMO: Motion Segmentation Dataset and Learning Pipeline for Event Cameras                                                    |       | 
| ICCV  2019  | Event-Based Motion Segmentation by Motion Compensation                                                                         |       |
| CVPR 2020   | Video to Events:Recycling Video Datasets for Event Cameras                                                                     |       |
| CVPR 2020   | Learning Visual Motion Segmentation using Event Surfaces                                                                       |       |
| ECCV 2020   | Jointly learning visual motion and confidence from local patches in event cameras                                              |       |
| IROS 2021   | Event-based Motion Segmentation by Cascaded Two-Level Multi-Model Fitting                                                      |       |
| MMsys 2021  | EMotion segmentation and tracking for integrating event cameras                                                                |       |
| IROS 2021   | SpikeMS: Deep Spiking Neural Network for Motion Segmentation                                                                   |       |
| TNNLS 2021  | Event-Based Motion Segmentation With Spatio-Temporal Graph Cuts                                                                |       |
| ICRA 2021   | 0-MMS: Zero-Shot Multi-Motion Segmentation With A Monocular Event Camera                                                       |       |
| AAAI 2022   | ProgressiveMotionSeg: Mutually Reinforced Framework for Event-Based Motion Segmentation                                        |       |

# 3 Event + RGB 

## 3.1 Event Guided Image / Video SR


| Publication | Title                                                                                                                            | Highlight                              |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| CVPR 2023   | Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution | INR 🆕 |
| CVPR 2020   | EventSR: From Asynchronous Events to Image Reconstruction, Restoration, and Super-Resolution via End-to-End Adversarial Learning | EventSR; End2end; Adversarial learning |
| TAPMI 2020  | E2SRI: Learning to Super-Resolve Intensity Images from Events                                                                    | SR-Net; RCNN                           |


## 3.2 Event Guided Image / Video Deblurring
| Publication | Title                                                               | Highlight                |
| ----------- | ------------------------------------------------------------------- | ------------------------ |
| IJCV 2023   | Event-guided Multi-patch Network with Self-supervision for Non-uniform Motion Deblurring              | 🆕 |
| CVPR 2023   | Event-Based Frame Interpolation With Ad-Hoc Deblurring              | 🆕 |
| ECCV 2020   | Event Enhanced High-Quality Image Recovery                          | eSL-Net; Sparse learning |
| ECCV 2022   | Event-guided Deblurring of Unknown Exposure Time Videos             | ETES; Cross-modal        |
| T-CAI       | Robust motion compensation for event cameras with smooth constraint |                          |
| MMM 2021    | Fine-Grained Video Deblurring with Event Camera                     |                          |
| Access 2020 | Hybrid Deblur Net: Deep Non-Uniform Deblurring With Event Camera    |                          |
| ECCV 2020   | Learning Event-Driven Video Deblurring and Interpolation            |                          |
| ECCV 2020   | Event Enhanced High-Quality Image Recovery                          |                          |
| CVPR 2020   | Learning Event-Based Motion Deblurring                              |                          |
| T-CAI       | Robust motion compensation for event cameras with smooth constraint |                          |



## 3.3 Event Guided Image / Video HDR
| Publication | Title                                                                                                                            | Highlight                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| CVPR 2022   | Multi-Bracket High Dynamic Range Imaging With Event Cameras                                                                      |                            |
| CVPR 2021   | Learning To Reconstruct High Speed and High Dynamic Range Videos From Events                                                     |                            |
| IJCV 2021   | Learning to Reconstruct HDR Images from Events, with Applications to Depth and Flow Prediction                                   |                            |
| ICCV 2021   | An Asynchronous Kalman Filter for Hybrid Event Cameras                                                                           |                            |
| CVPR 2019   | Event-based High Dynamic Range Image and Very High Frame Rate Video Generation using Conditional Generative Adversarial Networks | Conditional GAN            |
| TAPMI 2019  | High Speed and High Dynamic Range Video with an Event Camera                                                                     | Recurrent network          |
| CVPR 2021   | Learning to Reconstruct High Speed and High Dynamic Range Videos from Events                                                     | RCNN; Temporal consistency |
| CVPR 2020   | Neuromorphic Camera Guided High Dynamic Range Imaging                                                                            |                            |

### (1) Low Light
| Publication | Title                                   | Highlight |
| ----------- | --------------------------------------- | --------- |
| AAAI 2023   | Low-Light Video Enhancement with Synthetic Event Guidance |     🆕       |
| ECCV 2020   | Learning to See in the Dark with Events |           |

## 3.4 Event Guided Image / Video Denoising

| Publication | Title                                                                                                | Highlight |
| ----------- | ---------------------------------------------------------------------------------------------------- | --------- |
| CVPR 2021   | EventZoom: Learning To Denoise and Super Resolve Neuromorphic Events                                 | Denoise   |
| CVPR 2020   | Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging |           |


## 3.5 Event Guided Video Interpolation

| Publication | Title                                                                                               | Highlight                 |
| ----------- | --------------------------------------------------------------------------------------------------- | ------------------------- | 
| CVPR 2023   | Event-Based Blurry Frame Interpolation Under Blind Exposure                                         |            🆕             |
| CVPR 2023   | Event-Based Video Frame Interpolation With Cross-Modal Asymmetric Bidirectional Motion Fields       |            🆕             |
| CVPR 2023   | Event-Based Frame Interpolation With Ad-Hoc Deblurring                                              |            🆕             |
| CVPR 2022   | TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation                      |                           |
| CVPR 2022   | E-CIR: Event-Enhanced Continuous Intensity Recovery                                                 |                           |
| CVPR 2022   | Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion | Dataset                   |
| CVPR 2022   | Unifying Motion Deblurring and Frame Interpolation with Events                                      |                           |
| ECCV 2022   | Video Interpolation by Event-driven Anisotropic Adjustment of Optical Flow                          |                           |
| ICCV 2021   | Training Weakly Supervised Video Frame Interpolation with Events                                    | Weakly Supervised         |
| CVPR 2021   | Time Lens: Event-based Video Frame Interpolation                                                    | First Work; Dataset       |
| CVPR 2021   | EFI-Net: Video Frame Interpolation from Fusion of Events and Frames                                 | Dataset; Open Dataset     |
| ECCV 2020   | Learning Event-Driven Video Deblurring and Interpolation                                            | Deblurring; Interpolation |
| ICCV-W 2019 | Event-driven Video Frame Synthesis                                                                  |                           |

## 3.6 Depth estimation
| Publication | Title                                                               | Highlight |
| ----------- | ------------------------------------------------------------------- | --------- |
| CVPR 2023   | Hierarchical Neural Memory Network for Low Latency Event Processing                                                            |    🆕     |
| ICCV 2021   | Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds |           |

# 4 Domain Specific


## 4.1 NeRF & 3D reconstruction 

| Publication | Title                                                                                     | Highlight |
| ----------- | ----------------------------------------------------------------------------------------  | --------- |
| WACV 2023  | Ev-NeRF: Event Based Neural Radiance Field                                                                |    🆕     |
| RAL  2023  | E-NeRF: Neural Radiance Fields from a Moving Event Camera                                                 |    🆕     |
| CVPR  2023  | EventNeRF: Neural Radiance Fields from a Single Colour Event Camera                                      |    🆕     |
| IJCV  2018  | EMVS: Event-Based Multi-View Stereo—3D Reconstruction with an Event Camera in Real-Time                  |           |
| Arxiv 2020  | E3D: Event-Based 3D Shape Reconstruction                                                                 |           |
| ECCV  2022  | EvAC3D: From Event-based Apparent Contours to 3D Models via Continuous Visual Hulls                      |           |
| Arxiv 2022  | Event-based Non-Rigid Reconstruction from Contours                                                       |           |
| Arxiv 2022  | Event-Based Dense Reconstruction Pipeline                                                                |           |
| ECCV  2016  | Real-Time 3D Reconstruction and 6-DoF Tracking with an Event Camera                                      |           |
| ICAR  2019  | Multi-View 3D Reconstruction with Self-Organizing Maps on Event-Based Data                               |           |
| IEEE  2018  | Ultimate slam? combining events, images, and imu for robust visual slam in hdr and high-speed scenarios  |           |
| 3DV   2021  | ESL: Event-based Structured Light                                                                        |           |
| ECCV  2020  | Stereo Event-Based Particle Tracking Velocimetry for 3D Fluid Flow Reconstruction                        |           |
| ICCV  2019  | Learning an Event Sequence Embedding for Dense Event-Based Deep Stereo                                   |           |

## 4.2 Human Pose and Shape

| Publication | Title                                                                                | Highlight  |
| ----------- | ------------------------------------------------------------------------------------ | ---------- |
| Arxiv 2022  | Efficient Human Pose Estimation via 3D Event Point Cloud                             |            |
| Arxiv 2022  | A Temporal Densely Connected Recurrent Network for Event-based Human Pose Estimation |            |
| ICCV 2021   | EventHands: real-time neural 3D hand pose estimation from an event stream            | Hand Pose  |
| CVPR 2021   | Lifting Monocular Events to 3D Human Poses                                           |            |
| ICCV 2021   | EventHPE: Event-based 3D Human Pose and Shape Estimation                             | Human Pose |
| CVPR 2020   | EventCap: Monocular 3D Capture of High-Speed Human Motions Using an Event Camera     | Human Pose |
| WACV 2019   | Space-Time Event Clouds for Gesture Recognition: From RGB Cameras to Event Cameras   | Hand Pose  |
| CVPR 2019   | DHP19: Dynamic Vision Sensor 3D Human Pose Dataset                                   |            |
| Arxiv 2019  | EventGAN: Leveraging Large Scale Image Datasets for Event Cameras                    |            |
| ICCV  2021  | EventHPE: Event-Based 3D Human Pose and Shape Estimation                                                 |           |

## 4.3 Body and Eye Tracking

| Publication | Title                                                                             | Highlight |
| ----------- | --------------------------------------------------------------------------------- | --------- |
| CAAI2020    | Object tracking on event cameras with offline–online learning                     |           |
| NN 2021     | Real-Time Face & Eye Tracking and Blink Detection using Event Cameras             |           |
| T-CG 2021   | Event Based, Near-Eye Gaze Tracking Beyond 10,000Hz                               | Dataset   |

## 4.4 Face
| Publication | Title                                  | Highlight |
| ----------- | -------------------------------------- | --------- |
| Sensor 2020 | Face pose alignment with event cameras |           |



## 4.5 Compression
| Publication | Title                                       | Highlight |
| ----------- | ------------------------------------------- | --------- |
| T-SPL 2020  | Lossless Compression of Event Camera Frames |           |

## 4.4 SAI

| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| CVPR 2021   | Event-Based Synthetic Aperture Imaging With a Hybrid Network |           |


# 5 Robotic Vision
## 5.1 Object Detection and Tracking

This section focuses on event-based detection/tracking tasks for Robotics implementation.
| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| CVPR 2022   | Frame-Event Alignment and Fusion Network for High Frame Rate Tracking      |     🆕      | 
| RAL 2022   | EV-Catcher: High-Speed Object Catching Using Low-Latency Event-Based Neural Networks      | DL        | 
| IEEE Robot & Automat 2021| Fast motion understanding with spatiotemporal neural networks and dynamic vision sensors |DL |
| arxiv 2021  | EVReflex: Dense Time-to-Impact Prediction for Event-based Obstacle Avoidance.             | DL        | 
| ICRA 2020   | EVDodgeNet: Deep Dynamic Obstacle Dodging with Event Cameras                              | DL        |
| CAAI 2020   | Object tracking on event cameras with offline–online learning                             | DL        | 

## 5.2 Simultaneous Localization and Mapping (SLAM)

Foundational Event-based SLAM (no-deep learning) for pure discussion.
| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| Sensor 2022 | Visual Odometry with an Event Camera Using Continuous Ray Warping and Volumetric Contrast Maximization |  no DL         |
| CVPR-W 2021 | Comparing Representations in Tracking for Event Camera-based SLAM                          |  no DL         |
| IEEE robot 2021 | Event-based Stereo Visual Odometry                                                     |  no DL         |
| RAL 2018    | Ultimate SLAM? Combining Events, Images, and IMU for Robust Visual SLAM in HDR and High Speed Scenarios|  no DL         |
| RAL 2017    | "EVO: A Geometric Approach to Event-Based 6-DOF Parallel Tracking and Mapping in Real Time |  no DL         |

### 5.2.1 Front-End of SLAM
#### 5.2.1.1 Feature Detection for Tracking

| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| CVPR 2023   | [Data-driven Feature Tracking for Event Cameras](https://github.com/uzh-rpg/deep_ev_tracker) |    🆕     |
| PAMI 2021   | luvHarris: A Practical Corner Detector for Event-Cameras                                   | no DL     |
| CVPR-W 2019 | Fast Event-based Corner Detection                                                          | no DL     |
| IROS 2019   | Fa-harris: A fast and asynchronous corner detector for event cameras                       | no DL     |
| CVPR-W 2019 | Speed Invariant Time Surface for Learning to Detect Corner Points with Event-Based Cameras | ML        |
| ECCV 2018   | Asynchronous, Photometric Feature Tracking using Events and Frames                         | ML     |
| 3DV 2018    | ACE: An efficient asynchronous corner tracker for event cameras                            | no DL     |

#### 5.2.1.2 Optical Flow for Tracking

| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| CVPR 2023 | Tangentially Elongated Gaussian Belief Propagation for Event-Based Incremental Optical Flow Estimation                            | 🆕 | 
| ECCV 2022 | Secrets of Event-Based Optical Flow                            | no DL | 
| 3DV 2021 | E-RAFT: Dense Optical Flow from Event Cameras                   | DL | 
| T-PAMI 2020| Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception | DL | 
| CVPR 2020| Single Image Optical Flow Estimation with an Event Camera | no DL |  
| CVPR 2020| Single Image Optical Flow Estimation With an Event Camera  | DL  | 
| ECCV 2020 | Spike-FlowNet: Event-Based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks | DL | 
| ECCV 2020 | Jointly Learning Visual Motion and Confidence from Local Patches in Event Cameras| DL | 
| Robotics: S&S XIV 2018| EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras | DL | 

### 5.2.2 Back-End of SLAM

In the above summarised Optical Flow, we have obtained the pixel motion. The ego-motion estimation we discussed here focuses on explicitly regressing the 6-DOF motion from event data through an end-to-end trainable pipeline. In traditional SLAM, we can easily estimate depth and ego motion through triangulation and the followed local pose estimation e.g. PnP. When adopting a learning-based approach, the optical flow, depth and ego-motion estimation usually be united in single, consistent frameworks, which provides an opportunity for dense mapping in the SLAM system. 

#### 5.2.2.1 Ego-motion estimation

CVPR and IROS present the united framework for optical flow, depth and ego-motion estimation. The recent EAGAN adopting Transformer to boost the performance of optical flow task to SOTA, yet no optimization is done to depth estimation. Moreover, the network show increse in parameter while the motion estimation is even kicked out of the network.

| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| CVPR  2022| Progressive Spatio-Temporal Alignment for Efficient Event-Based Motion Estimation  | 🆕 | 
| ARXIV 2022| EAGAN: Event‐based attention generative adversarial networks for optical flow and depth estimation | DL | 
| IROS 2020| Unsupervised Learning of Dense Optical Flow, Depth and Egomotion with Event-Based Sensors | DL | 
| CVPR 2019| Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion | DL | 
| T-IRAL 2016 | Accurate Angular Velocity Estimation With an Event Camera     | no DL |


#### 5.2.2.2 3D Reconstruction
| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| IET Comput. Vis. 2022| Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds | DL | 
| 3DV 2020 | Learning Monocular Dense Depth from Events                                      | DL | 

### 5.2.3 End-to-End SLAM
| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| ICRAS 2022 | Event-Based Dense Reconstruction Pipeline                                               | DL reconstruction     | 
| ESWA 2021  | A 6-DOFs event-based camera relocalization system by CNN-LSTM and image denoising       | DL pose+relocalization| 
| CVPR-W 2019| Real-Time 6DOF Pose Relocalization for Event Cameras With Stacked Spatial LSTM Networks | DL pose+relocalization| 
| ARXIV 2019 | V2CNet: A Deep Learning Framework to Translate Videos to Commands for Robotic Manipulation | DL Task-specific (control)| 

## 5.3 Multi-sensor

| Publication | Title                              | Highlight |
| ----------- | ---------------------------------- | --------- |
| IET Comput. Vis. 2022| Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds | DL | 
| CVPR-W 2021 | How To Calibrate Your Event Camera |  no DL         |

# 6 New Direction
| Publication | Title                                                        | Highlight |
| ----------- | ------------------------------------------------------------ | --------- |
| CVPR 2023| Event-based Shape from Polarization |Shape from Polarization 🆕 | 
| CVPR2023 | Event-Guided Person Re-Identification via Sparse-Dense Complementary Learning | Person Re-Identification 🆕| 
| CVPR 2023| Adaptive Global Decay Process for Event Cameras | 🆕| 
| CVPR 2023| Learning Adaptive Dense Event Stereo From the Image Domain | 🆕| 


# 7 Discussion


# 8 Conclusion




# Supplementary Material

## Event Dataset

| Publication   | Title                                                                                                 | Highlight    |
| ------------- | ----------------------------------------------------------------------------------------------------- | ------------ |
| IEEE RAL 2022 | VECtor: A Versatile Event-Centric Benchmark for Multi-Sensor SLAM                                     | multi-sensor |
| CVPR 2022     | Video to -Events: Recycling Video Datasets for Event Cameras                                          |              |
|               | DSEC: A Stereo Event Camera Dataset for Driving Scenarios                                             |              |
| T-CG 2021     | Event Based, Near-Eye Gaze Tracking Beyond 10,000Hz                                                   | Eye Tracking |
| IEEE 2021     | DSEC: A Stereo Event Camera Dataset for Driving Scenarios                                             |              |
| CVPR 2019     | CED: Color Event Camera Dataset                                                                       | First Color  |
| CVPR 2019     | DHP19: Dynamic Vision Sensor 3D Human Pose Dataset                                                    |              |
| IEEE 2018     | The Multivehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception               |              |
| CVPR 2018     | Hats: Histograms of averaged time surfaces for robust event-based object classification               | N-Cars       |
| Neurosci 2015 | Converting static image datasets to spiking neuromorphic datasets using saccades                      | Caltech101   |
| IROS 2016     | The event-camera dataset and simulator: Event-based data for pose estimation,visual odometry,and slam | DVS 260      |

## Simulation
| Publication | Title                                                       | Highlight |
| ----------- | ----------------------------------------------------------- | --------- |
| ECCV 2020   | Reducing the Sim-to-Real Gap for Event Cameras              |           |
| CVPR 20202  | Video to Events: Recycling Video Datasets for Event Cameras |           |
