# Rethinking Unlearning for Transfer Learning

Official PyTorch implementation of Rethinking Unlearning for Transfer Learning.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Datasets Preparation](#datasets-preparation)
- [Training and Unlearning](#training-and-unlearning)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Introduction

Machine unlearning (MU) has become a critical area of research due to data regulations like GDPR, addressing the trustworthiness and safety of AI foundation models. These models, trained on vast web data, often raise privacy and copyright concerns. While existing MU methods mainly target smaller datasets (e.g., CIFAR-10, MNIST), their impact on transfer learning(commonly used in large-scale models) remains underexplored. Our study investigates the practical effects of MU methods on transfer learning. We apply unlearning techniques to forget selected classes on ImageNet pre-trained models and evaluate their classification accuracy using k-nearest neighbors on the Office-Home dataset, comparing unlearned models with retrained models.

<p align="center"><img src="images/picture1.png" alt="graph" width="90%"></p>

## Features

- **Machine Unlearning**: Focuses on efficiently removing specific data or knowledge from trained AI models while minimizing performance degradation, ensuring adaptability for real-world applications and compliance with regulatory requirements.
- **Privacy-Compliant**: Implements machine unlearning to comply with data regulations like GDPR by addressing privacy and copyright concerns.
- **Scalable Unlearning**: Adapts unlearning methods to foundation models and large-scale datasets beyond smaller benchmarks like CIFAR-10 and MNIST.
- **Transfer Learning Evaluation**: Measures the impact of unlearning on transfer learning using k-nearest neighbors across downstream tasks.
- **Comparative Analysis**: Compares unlearned models with retrained models for classification accuracy, k-nearest neighbors and CKA ensuring practical insights.

## Requirements


```bash
pip install -r requirements.txt
```

## Datasets Preparation

All datasets used in our experiments are publicly available. Please refer to the [`DATA.md`](DATA.md) file for detailed instructions on how to set up each dataset.

## Training and Unlearning

1. Get the origin model.
    ```bash
    python main_train.py --dataset ${dataset} --arch ${model architechture} --imagenet_arch --save_dir ${save_dir} --epochs ${epochs for training} --lr ${learning rate for training} --save_dir ${file to save the orgin model}
    ```

    A simple example for ResNet-50 on ImageNet.
    ```bash
    python main_train.py --dataset imagenet --arch resnet50 --imagenet_arch --save_dir ./result --lr 0.1 --epochs 182
    ```

2. Generate Saliency Map
    ```bash
    python generate_mask.py --save_dir ${saliency_map_path} --model_path ${original model path} --class_to_replace ${classes to forget} --unlearn_epochs 1
    ```

3. Unlearn
    * Our method
    ```bash
    python main_forget.py --dataset imagenet --num_classes 1000 --arch resnet50 --imagenet_arch --save_dir ${save_dir} --model_path ${original model path} --unlearn SPKD --class_to_replace ${classes to forget} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    A simple example for unlearning ResNet-50 on ImageNet using SPKD.
    ```bash
    python main_forget.py --dataset imagenet --num_classes 1000 --arch resnet50 --imagenet_arch --save_dir ./result/ --model_path ${original model path} --unlearn SPKD --unlearn_epochs 15 --unlearn_lr 1e-5 --batch_size 128
    ```

    * Retrain
    ```bash
    python main_forget.py --dataset imagenet --num_classes 1000 --arch resnet50 --imagenet_arch --save_dir ${save_dir} --model_path ${original model path} --unlearn retrain --class_to_replace ${classes to forget} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * FT
    ```bash
    python main_forget.py --dataset imagenet --num_classes 1000 --arch resnet50 --imagenet_arch --save_dir ${save_dir} --model_path ${original model path} --unlearn FT --class_to_replace ${classes to forget} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * GA
    ```bash
    python main_forget.py --dataset imagenet --num_classes 1000 --arch resnet50 --imagenet_arch --save_dir ${save_dir} --model_path ${original model path} --unlearn GA --class_to_replace 4500 --class_to_replace ${classes to forget} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * l1-sparse
    ```bash
    python -u main_forget.py --dataset imagenet --num_classes 1000 --arch resnet50 --imagenet_arch --save_dir ${save_dir} --model_path ${original model path} --unlearn FT_prune --class_to_replace ${classes to forget} --alpha ${alpha} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * SalUn
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --class_to_replace ${classes to forget} --model_path ${original model path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
    ```

## Citation

If you find this repository helpful in your research, please cite:

```bibtex
@inproceedings{-,
    title={},
    author={},
    booktitle={},
    year={},
  }
```

## License

This project is licensed under the terms of the MIT license.

## Contact

For any questions, please reach out to:

- **Youngkyun Kim**: [ygkim08@korea.ac.kr](mailto:ygkim08@korea.ac.kr)
- **Yongwoo Kim**: [yongwookim@korea.ac.kr](mailto:yongwookim@korea.ac.kr)

We appreciate your interest in our work.
