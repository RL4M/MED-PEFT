# Less Could Be Better: Parameter-efficient Fine-tuning Advances Medical Vision Foundation Models
This repository includes an official implementation of paper: [Less Could Be Better: Parameter-efficient Fine-tuning Advances Medical Vision Foundation Models](TODO:arxivlink).

Some code is borrowed from [MRM](https://github.com/RL4M/MRM-pytorch), [LoRA](https://github.com/microsoft/LoRA), and [MAE](https://github.com/facebookresearch/mae).

## 1 Environmental preparation and quick start
**Environmental requirements**
- Ubuntu 20.04 LTS.

- Python 3.10.8

If you are using anaconda/miniconda, we provide an easy way to prepare the environment for pre-training and finetuning of classification:

      conda env create -f environment.yaml
      pip install -r requirements.txt

## 2 Pre-trained weights preparation
Download the pre-trained weight of [MRM](https://drive.google.com/file/d/1JwZaqvsSdk1bD3B7fsN0uOz-2Fzz1amc/view?usp=sharing) and [CXR_MAE](https://drive.google.com/file/d/1v-IzAz8ZPvorHNtHmJn4O9Ih6E4J2PT9/view?usp=drive_link), putting them into the directory of [pretrained_weights](/pretrained_weights)


## 3 Data preparation (take NIH ChestX-ray 14 dataset as the example)
- Download NIH ChestX-ray 14 dataset and split [train/valid/test](DatasetsSplits/NIH_ChestX-ray) set. The directory should be organized as follows:
```
      NIH_ChestX-ray/
            all_classes/
                  xxxx1.png
                  xxxx2.png
                  ...
                  xxxxn.png
            train_1.txt
            trian_10.txt
            train_list.txt
            val_list.txt
            test_list.txt
```	
- Specify the ``dataset_path`` in [ft_lora_mrm.sh](/ft_lora_mrm.sh)

## 4 Start fine-tuning (take 1% data as the example)

- Specify ``pretrained_path`` in [ft_lora_mrm.sh](/ft_lora_mrm.sh) as ``pretrained_weights/MRM.pth``.

- Start training by running
```
      chmod a+x ft_lora_mrm.sh
      ./ft_lora_mrm.sh
```
## 5 Recommended LoRA fine-tuning hyperparameters
### 5.1 For MRM pre-trained weights

|     NIH     |     epochs           |     optimizer    |     learning rate    |   lora rank  |
|-------------|----------------------|------------------|----------------------|--------------|
|     1%      |     100              |     sgd          |     0.05             |      4       |
|     10%     |     100              |     sgd          |     0.05             |      8       |
|     100%    |     100              |     sgd          |     0.2              |      32      |

|   CheXpert  |     epochs           |     optimizer    |     learning rate    |   lora rank  |
|-------------|----------------------|------------------|----------------------|--------------|
|     1%      |     100              |     sgd          |     0.02             |      4       |
|     10%     |     100              |     sgd          |     0.05             |      16      |
|     100%    |     100              |     sgd          |     0.01             |      32      |

|     RSNA    |     epochs           |     optimizer    |     learning rate    |   lora rank  |
|-------------|----------------------|------------------|----------------------|--------------|
|     1%      |     100              |     sgd          |     0.01             |      4       |
|     10%     |     100              |     sgd          |     0.01             |      32      |
|     100%    |     100              |     sgd          |     0.01             |      32      |


### 5.2 For CXR_MAE pre-trained weights

|     NIH     |     epochs           |     optimizer    |  base learning rate  |   lora rank  |
|-------------|----------------------|------------------|----------------------|--------------|
|     1%      |     400              |     adamw        |         3e-3         |      4       |
|     10%     |     200              |     adamw        |         1e-3         |      8       |
|     100%    |     100              |     adamw        |         3e-3         |      32      |

|   CheXpert  |     epochs           |     optimizer    |  base learning rate  |   lora rank  |
|-------------|----------------------|------------------|----------------------|--------------|
|     1%      |     400              |     adamw        |         3e-3         |      4       |
|     10%     |     100              |     adamw        |         5e-3         |      8       |
|     100%    |     100              |     adamw        |         3e-3         |      32      |


|     RSNA    |     epochs           |     optimizer    |  base learning rate  |   lora rank  |
|-------------|----------------------|------------------|----------------------|--------------|
|     1%      |     100              |     adamw        |         3e-3         |      4       |
|     10%     |     100              |     adamw        |         5e-3         |      8       |
|     100%    |     100              |     adamw        |         5e-3         |      32      |


## 6 Links to download datasets
- [NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)

- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/#:~:text=What%20is%20CheXpert%3F,labeled%20reference%20standard%20evaluation%20sets.)

- [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)


## 7 Datasets splits
In the directory [DatasetsSplits](DatasetsSplits), we provide dataset splits that may be helpful for organizing the datasets.

We give the train/valid/test splits of [NIH ChestX-ray](DatasetsSplits/NIH_ChestX-ray), [CheXpert](DatasetsSplits/CheXpert), and [RSNA Pneumonia](DatasetsSplits/RSNA_Pneumonia).
