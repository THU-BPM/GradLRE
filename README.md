# GradLRE: Gradient Imitation Reinforcement Learning for Low resource Relation Extraction.

This project provides tools for "[GradLRE: Gradient Imitation Reinforcement Learning for Low resource Relation Extraction.](https://arxiv.org/abs/2109.06415)" in EMNLP as a long paper.

Details about low resource relation are in the paper and the implementation is based on the PyTorch library. 

## Quick Links
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

For training, a GPU is recommended to accelerate the training speed.

### PyTroch

The code is based on PyTorch 1.6+. You can find tutorials [here](https://pytorch.org/tutorials/).

### Dependencies

The code is written in Python 3.7. Its dependencies are summarized in the file ```requirements.txt```. 
```
torch==1.6.0
numpy==1.18.5
scikit_learn==0.23.2
transformers==3.5.1
tqdm==4.48.2
```
You can install these dependencies like this:
```
pip3 install -r requirements.txt
```
## Usage
* Run the full model on SemEval dataset with default hyperparameter settings<br>

```python3 src/train.py```<br>

* If you need data augmentation to generate unlabeled data in low resource scenarios, please run with the following parameter<br>

```python3 src/train.py --use_aug True```<br>
 

## Data
### Format
Each dataset is a folder under the ```./data``` folder:
```
./data
└── SemEval
    ├── train_sentence.json
    ├── train_label_id.json
    ├── dev_sentence.json
    ├── dev_label_id.json
    ├── test_sentence.json
    └── test_label_id.json

```
### Download

* SemEval: SemEval 2010 Task 8 data (included in ```data/SemEval```)<br>
* TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))<br>

Then use the scripts from ```data/data_prepare.py``` to further preprocess the data. For SemEval, the script split the original training data into two sets. For TACRED, the script first perform some preprocessing to ensure the same format as SemEval.
 
 
## Acknowledgements
https://github.com/huggingface/transformers

https://github.com/INK-USC/DualRE

## Contact

If you have any problem about our code, feel free to contact: hxm19@mails.tsinghua.edu.cn

## Reference

If the code is used in your research, hope you can cite our paper as follows:
```
@inproceedings{hu2021gradient,
  abbr = {EMNLP},
  title = {Gradient Imitation Reinforcement Learning for Low Resource Relation Extraction},
  author = {Hu, Xuming and Zhang, Chenwei and Yang, Yawen and Li, Xiaohe and Lin, Li and Wen, Lijie and Yu, Philip S.},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year = {2021},
  pdf = {https://arxiv.org/pdf/2109.06415.pdf},
  code = {https://github.com/THU-BPM/GradLRE}
}
```