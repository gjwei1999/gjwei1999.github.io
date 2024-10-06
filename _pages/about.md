---
permalink: /
title: "Hello! Welcome to Jiawei Guo's homepage!"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Education
------
* Ph.D. in Physics, Carnegie Mellon University, Aug. 2021 - (exp) May 2026
* B.S. in Physics, Shandong University, Sep. 2017 - June 2021
* Visiting International Student, Duke University, Aug. 2019 - May 2020

Qualification
------
* 4+ years research experience in physics, data science, and machine learning.
* 4+ years of programming experience with Python, C++ and C.

Research Experience
------
My research focuses on firstly using machine learning techniques, particularly deep reinforcement learning to assist experiment control and secondly advanced data analysis with statistical methods and high-performance computing.

**AI/ML Assisted Experiment Control**\
Mar. 2024-Present
* The objective is to develop and train a reinforcement learning model for real-time control of the radiator rotation, ensuring the production of a polarized photon beam with the precise energy required for the GlueX experiment.
* Developed an automated data collection pipeline in Python, processing 500MB time-series experiment control data into CSV format, optimizing data accessibility and facilitating streamlined analysis and model training.
* Conducted exploratory and correlation analysis to identify most relevant variables influencing photon energy.
* Trained and optimized a surrogate model using Gaussian Process Regression to map photon energy from relevant variables and integrated the model into the custom RL environment built with the OpenAI gymnasium.
* Implemented DDPG and TD3 algorithms using TensorFlow to train reinforcement learning agents. Enhanced agent performance by the refining observation space, reward function, and fine-tuning model hyperparameters.

**GlueX Experiment Software Engineering and Data Analysis**\
Aug. 2021-Present
* Established a processing pipeline in C++ and processed over 25TB experimental data using computing clusters at CMU.
* Implemented a statistical weighting method, applied event-by-event, to disentangle the contributions of different decay processes in the data, effectively separating signal from background channels and improving the data purity.
* Performed partial wave analysis based on maximum likelihood estimation (MLE) with gradient descent optimization and parallel computing with MPI and GPUs, optimizing models to extract physics insights.
* Led the study of mathematical ambiguity in the MLE analysis, demonstrating that multiple parameter sets can yield the same likelihood. Derived criteria for the occurrence of ambiguity, which were verified by Monte Carlo simulations.
* Developing LASSO regularization technique in C++ to enhance model selection for the partial wave analysis.

**Algorithm Development for PandaX-4T Supernova Early Warning**\
Nov. 2020-Jun. 2021
* Developed an object-oriented sliding window algorithm in C++ for the prompt detection of supernova bursts.
* Implemented Monte Carlo simulation to assess the algorithm’s performance in classifying supernova burst signals amidst Poisson-distributed background noise, ensuring accurate detection capabilities.
* Achieved a 99.73% true positive rate and limited the false positive frequency to once a week with optimized parameters. 


Selected Course Project
------
**Image Captioning : Exploring CNN + CNN And CNN + Transformer Model**\
Course: 10-701 Introduction to Machine Learning
* Built 2 models with PyTorch, one using CNN+CNN for vision and language, the other using CNN+Transformer.
* Applied hierarchical attention mechanism for CNN+CNN model, achieving results comparable to LSTM-based models.
* Trained models with 10K images from MSCOCO dataset, demonstrating the CNN+Transformer model outperforms CNN-and LSTM-based models by 10%, with a BLEU-1 score around 70. See our final report [here](https://gjwei1999.github.io/files/10701_final_report.pdf).

