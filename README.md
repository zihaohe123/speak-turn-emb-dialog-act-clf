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
      2. speaker. the id of the speaker. the speaker id should be binary and indicates the turn of the speaker in this conversation. for dyadic conversations the original speaker ids should already be binary. in the case of multi-party conversations and speaker ids are non-binary, please refer to Section 3.3 of our paper on how to make the labels binary. if speaker ids are not available, just put all zeros. 
      3. text. the text of the utterance.
      4. act. the dialogue act label.
      5. topic. the topic label. if not available, just put all zeros.
3. Create a script as run_{dataset_name}.py. You can reuse most of the parameter settings in run_swda/mrda/dyda.py. If the conversations are very long (have a lot of utterances), consider slicing it into smaller chunks by specifying <em>chunk_size</em> to a non-zero value. 
   1. Set <em>copurs</em> to your {dataset_name}. 
   2. Set <em>nclass</em> to the number of dialogue act classes in your dataset.
4. Run the script
```angular2html
python run_{dataset_name}.py
```
5. In order to obtain the best performance, you may need to try different <em>batch_size</em>, <em>chunk_size</em> (32, 64, 128, 192, 256, 512 and etc.), <em>lr</em> (1e-4, 5e-5, 1e-5 and etc.), and <em>nfinetune</em> (1, 2).



## Test the trained model to a new dataset
1. Decide the pretraining dataset <em>pre_corpus</em> in {SWDA, MRDA, DyDA}. Choose the one that is most similar to your own dataset.
2. Train the model on the pretraining dataset using <em>run_pre_corpus.py</em>. Rename the saved checkpoint to <em>model.pt</em>.
3. Prepare your own dataset as described in Step 1 & 2 in "Train the model on other datasets". Encode the dialogue act labels of your own dataset using the mapping shown in the top comments of <em>dataset.py</em>. If you don't have training data and validation data, just prepare your test.csv file.
4. Make a copy of <em>run_pre_corpus.py</em> and change the following parameters.
   1. Set <em>corpus</em> to your {dataset_name}. 
   2. Set <em>mode</em> to <em>inference</em>.
5. Run the new script.
6. The predictions of the model (a list of predicted labels of all the utterances) will be saved in <em>preds_on_new.pkl</em>.

## Some tips
1. The code will save the processed data to pickle files under processed_data in order to prevent do the processing every time. If you made any changes to the processing in datasets.py, please delete the cached pickle files before you run the code.



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