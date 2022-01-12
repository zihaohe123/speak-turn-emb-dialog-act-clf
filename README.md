## Speaker Turn Modeling for Dialogue Act Classification
This repo implements this [paper](https://aclanthology.org/2021.findings-emnlp.185/).

# Installation
1. unzip data.zip
3. Install [Pytorch](https://pytorch.org/get-started/locally/) and [Huggingface Transformers](https://huggingface.co/docs/transformers/installation).


# Usage
To train the model on different datasets, simply run the corresponding file
```angular2html
python run_swda.py
python run_mrda.py
python run_dyda.py
```

The hyperparameters can be set in the three scripts and should be fairly understandable.


# Citation
```angular2html
@inproceedings{he2021speaker,
  title={Speaker Turn Modeling for Dialogue Act Classification},
  author={He, Zihao and Tavabi, Leili and Lerman, Kristina and Soleymani, Mohammad},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={2150--2157},
  year={2021}
}
```