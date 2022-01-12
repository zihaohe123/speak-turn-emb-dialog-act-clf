# Speaker Turn Modeling for Dialogue Act Classification
This repo implements this [paper](https://aclanthology.org/2021.findings-emnlp.185/).

## Installation
1. unzip data.zip
3. Install [Pytorch](https://pytorch.org/get-started/locally/) and [Huggingface Transformers](https://huggingface.co/docs/transformers/installation).


## Usage
To train the model on different datasets, simply run the corresponding file
```angular2html
python run_swda.py
python run_mrda.py
python run_dyda.py
```

The hyperparameters can be set in the three scripts and should be fairly understandable.

## Train the model on other datasets
1. Create a folder data/{dataset_name}
2. Put your train/val/test data as <em>train.csv</em>, <em>val.csv</em>, and <em>test.csv</em> under this folder
   1. For any of the three files, each row represents an utterance, and it must have the following columns:
      1. conv_id. the id of this conversation
      2. speaker_id. the id of the speaker. the speaker id should be binary and indicates the turn of the speaker in this conversation. for dyadic conversations the original speaker ids should already be binary. in the case of multi-party conversations and speaker ids are non-binary, please refer to Section 3.3 of our paper on how to make the labels binary. if speaker ids are not available, just put all zeros. 
      3. text. the text of the utterance.
      4. act. the dialogue act label.
      5. topic. the topic label. if not available, just put all zeros.
3. Create a script as run_{dataset_name}.py. You can reuse most of the parameter settings in run_swda/mrda/dyda.py. If the conversations are very long (have a lot of utterances), consider slicing it into smaller chunks by specifying <em>chunk_size</em> to a non-zero value. 
   1. Set <em>copurs</em> as your {dataset_name}. 
   2. Set <em>nclass</em> as the number of dialogue act classes in your dataset.
4. Run the script
```angular2html
python run_{dataset_name}.py
```
5. In order to obtain the best performance, you may need to try different <em>batch_size</em>, <em>chunk_size</em> (32, 64, 128, 192, 256, 512 and etc.), <em>lr</em> (1e-4, 5e-5, 1e-5 and etc.), and <em>nfinetune</em> (1, 2).


## Citation
```angular2html
@inproceedings{he2021speaker,
  title={Speaker Turn Modeling for Dialogue Act Classification},
  author={He, Zihao and Tavabi, Leili and Lerman, Kristina and Soleymani, Mohammad},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={2150--2157},
  year={2021}
}
```