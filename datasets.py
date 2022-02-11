import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import pickle

### Dialogue act label encoding, SWDA
# {'qw^d': 0, '^2': 1, 'b^m': 2, 'qy^d': 3, '^h': 4, 'bk': 5, 'b': 6, 'fa': 7, 'sd': 8, 'fo_o_fw_"_by_bc': 9,
#              'ad': 10, 'ba': 11, 'ng': 12, 't1': 13, 'bd': 14, 'qh': 15, 'br': 16, 'qo': 17, 'nn': 18, 'arp_nd': 19,
#              'fp': 20, 'aap_am': 21, 'oo_co_cc': 22, 'h': 23, 'qrr': 24, 'na': 25, 'x': 26, 'bh': 27, 'fc': 28,
#              'aa': 29, 't3': 30, 'no': 31, '%': 32, '^g': 33, 'qy': 34, 'sv': 35, 'ft': 36, '^q': 37, 'bf': 38,
#              'qw': 39, 'ny': 40, 'ar': 41, '+': 42}

### Topic label encoding, SWDA
# {'CARE OF THE ELDERLY': 0, 'HOBBIES AND CRAFTS': 1, 'WEATHER CLIMATE': 2, 'PETS': 3,
#              'CHOOSING A COLLEGE': 4, 'AIR POLLUTION': 5, 'GARDENING': 6, 'BOATING AND SAILING': 7,
#              'BASKETBALL': 8, 'CREDIT CARD USE': 9, 'LATIN AMERICA': 10, 'FAMILY LIFE': 11, 'METRIC SYSTEM': 12,
#              'BASEBALL': 13, 'TAXES': 14, 'BOOKS AND LITERATURE': 15, 'CRIME': 16, 'PUBLIC EDUCATION': 17,
#              'RIGHT TO PRIVACY': 18, 'AUTO REPAIRS': 19, 'MIDDLE EAST': 20, 'FOOTBALL': 21,
#              'UNIVERSAL PBLIC SERV': 22, 'CAMPING': 23, 'FAMILY FINANCE': 24, 'POLITICS': 25, 'SOCIAL CHANGE': 26,
#              'DRUG TESTING': 27, 'COMPUTERS': 28, 'BUYING A CAR': 29, 'WOODWORKING': 30, 'EXERCISE AND FITNESS': 31,
#              'GOLF': 32, 'CAPITAL PUNISHMENT': 33, 'NEWS MEDIA': 34, 'HOME REPAIRS': 35, 'PAINTING': 36,
#              'FISHING': 37, 'SOVIET UNION': 38, 'CHILD CARE': 39, 'IMMIGRATION': 40, 'JOB BENEFITS': 41,
#              'RECYCLING': 42, 'MUSIC': 43, 'TV PROGRAMS': 44, 'ELECTIONS AND VOTING': 45, 'FEDERAL BUDGET': 46,
#              'MOVIES': 47, 'AIDS': 48, 'HOUSES': 49, 'VACATION SPOTS': 50, 'VIETNAM WAR': 51, 'CONSUMER GOODS': 52,
#              'RECIPES/FOOD/COOKING': 53, 'GUN CONTROL': 54, 'CLOTHING AND DRESS': 55, 'MAGAZINES': 56,
#              'SVGS & LOAN BAILOUT': 57, 'SPACE FLIGHT AND EXPLORATION': 58, "WOMEN'S ROLES": 59,
#              'PUERTO RICAN STTEHD': 60, 'TRIAL BY JURY': 61, 'ETHICS IN GOVERNMENT': 62, 'FAMILY REUNIONS': 63,
#              'RESTAURANTS': 64, 'UNIVERSAL HEALTH INS': 65}


### Dialogue act label encoding, MRDA
# {'S':0, 'B':1, 'D':2, 'F':3, 'Q':4}

### Dialogue act label encoding, DyDA
# {1:0, 2:1, 3:2, 4:3}

### Topic label encoding, DyDA
# {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}

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
