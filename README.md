# Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness

Welcome to the official code repository for our NeurIPS 2023 submission paper "Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness". This repository contains the PyTorch implementation of our proposed method.

## Introduction

Adversarial attacks, which manipulate feature maps during neural network inference, pose a significant challenge to model accuracy. Our research identifies that low-frequency components, compared to their high-frequency counterparts, maintain more substantial information during such attacks.

We introduce an innovative strategy that leverages graph convolutional networks (GCNs) with low-pass filters for feature denoising, a methodology distinct from previous denoising methods that often overlook the inter-feature map data and the correlations between feature maps.

Our approach innovatively integrates a graph convolution-based denoising block. By reconstructing the graph with correlated feature map data, we have managed to bolster the model's robustness against various adversarial attacks. This method, compatible with a wide range of neural network architectures, presents a versatile solution for mitigating adversarial noise in feature maps.

## Implementation

The provided PyTorch code includes the implementation of our graph convolution-based denoising block, as well as the method for reconstructing the graph with correlated feature map data. We also provide example scripts for training and testing models using our proposed method.

## Directory Structure

The repository is structured as follows:

- `models/`: This directory contains the different network architectures used in our paper, including our proposed model with graph convolution-based denoising block.
- `README.md`: The file you're currently reading.
- `configs.yml`: This configuration file sets the parameters for adversarial training.
- `configs_simple.yml`: This configuration file sets the parameters for standard training.
- `configs_test.yml`: This configuration file sets the parameters for testing the model's robustness against attacks.
- `test_net.py`: This script is used for testing the standard training network's robustness against adversarial attacks.
- `train_free.py`: This script is used for adversarial training.
- `train_simple.py`: This script is used for standard training.
- `utils.py`: This file includes utility functions used across the project.
- `utils_AT.py`: This file includes utility functions specific to adversarial training.
- `validation.py`: This script is used for validating the trained models.

## Testing and Training Scripts

## Training Models

To train a network, open `train_simple.py` and set the desired network architecture. For example, you can use a wide residual network with depth 32 and widen factor 10 by setting `net = WRN32_10()`. 

If you want to use our proposed method with the denoising block, you can set `net = WRN32_10_GNN(block_pos=[1,2,3])`. The `block_pos` parameter determines the positions in the network where the denoising block is added after group convolution. You can adjust these positions according to your needs. 

Remember to adjust the configuration files (`configs.yml`, `configs_simple.yml`, and `configs_test.yml`) to match the settings of your training environment and the specific parameters of the models you are training.

### test_net.py

The `test_net.py` script is used to evaluate the robustness of the models trained using standard methods against adversarial attacks. The script includes four types of attacks that we utilized in our research: 

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent with 20 iterations (PGD-20)
- Projected Gradient Descent with 100 iterations (PGD-100)
- AutoAttack

These attacks provide a comprehensive evaluation of the model's robustness under different adversarial conditions.

### train_free.py

The `train_free.py` script is used for adversarial training of the models. In this training process, models are directly exposed to adversarial attacks as part of their learning journey. The goal of this approach is to familiarize the model with a range of adversarial attack patterns, thereby fortifying its ability to resist such attacks during inference.

After the adversarial training, we evaluate the robustness of the adversarially trained models against the same four types of attacks mentioned earlier: 

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent with 20 iterations (PGD-20)
- Projected Gradient Descent with 100 iterations (PGD-100)
- AutoAttack

This rigorous testing process provides a comprehensive assessment of how our adversarially trained models perform under various adversarial conditions, thereby validating the effectiveness of our training approach.

## Conclusion

In this study, we ventured to innovate by integrating Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs) through the introduction of a novel GCN-based denoising block. This unique design markedly amplifies the robustness of CNNs against adversarial attacks and highlights the untapped potential of leveraging GCNs for image processing tasks.

For more information about our work, please refer to our paper. If you encounter any issues when using our code, feel free to raise an issue on this GitHub repository. We appreciate any feedback or contributions to improve this project.

```
