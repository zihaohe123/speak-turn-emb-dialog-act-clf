import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import pickle


class DialogueActData(Dataset):
    def __init__(self, corpus, phase, chunk_size=0):
        os.makedirs('processed_data', exist_ok=True)
        if phase == 'train':
            data_path = f'processed_data/{corpus}_{chunk_size}_{phase}.pkl'
        else:
            data_path = f'processed_data/{corpus}_{phase}.pkl'

        print(f'Tokenizing {phase}....')
        if os.path.exists(data_path):
            input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_ = pickle.load(open(data_path, 'rb'))
        else:

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            df = pd.read_csv(f'data/{corpus}/{phase}.csv')
            max_conv_len = df['conv_id'].value_counts().max()
            if (chunk_size == 0 and phase == 'train') or phase != 'train':
                chunk_size = max_conv_len

            texts_all = df['text'].tolist()
            encodings_all = tokenizer(texts_all, truncation=True, padding=True)
            input_ids_all = np.array(encodings_all['input_ids'])
            attention_mask_all = np.array(encodings_all['attention_mask'])

            input_ids_ = []
            attention_mask_ = []
            labels_ = []
            chunk_lens_ = []
            speaker_ids_ = []
            topic_labels_ = []

            conv_ids = df['conv_id'].unique()
            for conv_id in conv_ids:
                mask_conv = df['conv_id'] == conv_id
                df_conv = df[mask_conv]
                input_ids = input_ids_all[mask_conv]
                attention_mask = attention_mask_all[mask_conv]
                speaker_ids = df_conv['speaker'].values
                labels = df_conv['act'].values
                topic_labels = df_conv['topic'].values

                chunk_indices = list(range(0, df_conv.shape[0], chunk_size)) + [df_conv.shape[0]]
                for i in range(len(chunk_indices) - 1):
                    idx1, idx2 = chunk_indices[i], chunk_indices[i + 1]

                    chunk_input_ids = input_ids[idx1: idx2].tolist()
                    chunk_attention_mask = attention_mask[idx1: idx2].tolist()
                    chunk_labels = labels[idx1: idx2].tolist()
                    chunk_speaker_ids = speaker_ids[idx1: idx2].tolist()
                    chunk_topic_labels = topic_labels[idx1: idx2].tolist()
                    chunk_len = idx2 - idx1

                    if idx2 - idx1 < chunk_size:
                        length1 = idx2 - idx1
                        length2 = chunk_size - length1
                        encodings_pad = [[0] * len(input_ids_all[0])] * length2
                        chunk_input_ids.extend(encodings_pad)
                        chunk_attention_mask.extend(encodings_pad)
                        labels_padding = np.array([-1] * length2)
                        chunk_labels = np.concatenate((chunk_labels, labels_padding), axis=0)
                        speaker_ids_padding = np.array([2] * length2)
                        chunk_speaker_ids = np.concatenate((chunk_speaker_ids, speaker_ids_padding), axis=0)
                        topic_labels_padding = np.array([99] * length2)
                        chunk_topic_labels = np.concatenate((chunk_topic_labels, topic_labels_padding), axis=0)

                    input_ids_.append(chunk_input_ids)
                    attention_mask_.append(chunk_attention_mask)
                    labels_.append(chunk_labels)
                    chunk_lens_.append(chunk_len)
                    speaker_ids_.append(chunk_speaker_ids)
                    topic_labels_.append(chunk_topic_labels)

        pickle.dump((input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_), open(data_path, 'wb'))

        print('Done')

        self.input_ids = input_ids_
        self.attention_mask = attention_mask_
        self.labels = labels_
        self.chunk_lens = chunk_lens_
        self.speaker_ids = speaker_ids_
        self.topic_labels = topic_labels_

    def __getitem__(self, index):
        item = {
            'input_ids': torch.tensor(self.input_ids[index]),
            'attention_mask': torch.tensor(self.attention_mask[index]),
            'labels': torch.tensor(self.labels[index]),
            'chunk_lens': torch.tensor(self.chunk_lens[index]),
            'speaker_ids': torch.tensor(self.speaker_ids[index], dtype=torch.long),
            'topic_labels': torch.tensor(self.topic_labels[index], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.labels)


def data_loader(corpus, phase, batch_size, chunk_size=0, shuffle=False):
    dataset = DialogueActData(corpus, phase, chunk_size=chunk_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
