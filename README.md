# [AAAI 2024] Mining Gaze for Contrastive Learning toward Computer-assisted Diagnosis

by Zihao Zhao\*, Sheng Wang\*, Qian Wang, Dinggang Shen<br/>

<!-- <div style="display: flex; justify-content: center; align-items: center;">
    <img src="asset/overview.png" alt="Image" style="width: 60%;">
</div> -->

<div align="center">
  <img src="asset/overview.png" style="width: 60%">
</div>
<!-- <p style="align-self: flex-start;">The illustration of our proposed McGIP. For contrastive pre-training, a positive pair is typically only constructed between a image and its augmented version. In our McGIP, the images with similar gaze patterns when read by a radiologist are also considered as positive pairs and be pulled closer in the latent space.</p> -->
The illustration of our proposed McGIP. For contrastive pre-training, a positive pair is typically only constructed between a image and its augmented version. In our McGIP, the images with similar gaze patterns when diagnosed by a radiologist are also considered as positive pairs and be pulled closer in the latent space.

---

## Introduction

In this paper, we introduce a plug-and-play module called McGIP. This module efficiently constructs positive sample pairs for contrastive learning in medical image analysis based on Gaze similarity.

- We provide the code for integrating McGIP into the contrastive learning framework, available [here](src/Contrastive+McGIP).
- Furthermore, we offer code to evaluate different schemes for comparing gaze similarity in medical images, available [here](src/GazeSimilarityEval/).

This integration enhances the performance of contrastive learning, leading to improved results.

## Repository Structure

This repository contains the following:

1. **Contrastive+McGIP Folder**: In this folder, you can find the contrastive learning code integrated with the McGIP module. These codes demonstrate how to incorporate McGIP into an existing contrastive learning framework to achieve superior performance.

2. **GazeSimilarityEval Folder**: In this folder, we provide code to compare different schemes for measuring gaze similarity in medical images. We designed three different schemes tailored to various situations in medical image analysis to assist you in evaluating doctors' gaze similarity.
