import os

# BERT/Roberta
# finetune=2, batchsize=3, chunksize=0(35)
# finetune=1, batchsize=5, chunksize=0


if __name__ == '__main__':
    corpus = 'dyda'
    mode = ['train', 'inference'][0]
    batch_size = 10
    batch_size_val = batch_size
    emb_batch = 0
    epochs = 100    # default 100
    gpu = ['0,1', '', '0,1,2,3', '0,1,2,3,4,5,6,7'][0]   # default 0,1
    lr = 1e-4   # default 1e-4
    nlayer = 2  # default 1
    chunk_size = 0
    dropout = 0.5   # default 0.5
    nfinetune = 1  # default 2
    nclass = 4
    speaker_info = ['none', 'emb_cls'][1]  # whether to use speaker embeddings or not
    topic_info = ['none', 'emb_cls'][1]     # whether to use topic embeddings or not
    os.makedirs(f'results_{corpus}', exist_ok=True)

    file_name = f"results_{corpus}/{corpus}_chunk={chunk_size}_nlayer={nlayer}.txt"

    if nfinetune != 2:
        file_name = file_name[:-4] + f'_nfinetune={nfinetune}.txt'

    if speaker_info != 'none':
        file_name = file_name[:-4] + f'_speaker={speaker_info}.txt'

    if topic_info != 'none':
        file_name = file_name[:-4] + f'_tinfo={topic_info}.txt'

    if lr != 1e-4:
        file_name = file_name[:-4] + f'_lr={lr}.txt'

    if dropout != 0.5:
        file_name = file_name[:-4] + f'_dropout={dropout}.txt'

    if not gpu:
        n_gpu = 0
    else:
        n_gpu = len(gpu.split(','))
    if n_gpu != 2:
        file_name = file_name[:-4] + f'_ngpu={n_gpu}.txt'

    # bash_file = ''
    # with open(f'run_{corpus}.sh') as f:
    #     for line in f:
    #         str_line = list(line)
    #         if 'gpu' in line:
    #             pos = line.index('=')
    #             str_line[pos+1:pos+2] = str(n_gpu)
    #         bash_file += ''.join(str_line)
    # with open(f'run_{corpus}.sh', 'w') as f:
    #     f.write(bash_file)

    command = f"python -u engine.py --corpus={corpus} --mode={mode} --gpu={gpu} --batch_size={batch_size} " \
              f"--batch_size_val={batch_size_val} --epochs={epochs} " \
              f"--lr={lr} --nlayer={nlayer} --chunk_size={chunk_size} --dropout={dropout} " \
              f"--nfinetune={nfinetune}  --speaker_info={speaker_info} " \
              f"--topic_info={topic_info} --nclass={nclass} --emb_batch={emb_batch}" \
              # f" > {file_name}"

    print(command)
    os.system(command)
