# Lipreading
Tensorflow implementation of various models for lip reading on GRID, LRS and etc.

## Dependencies

- Tensorflow 1.9.0+. 1.10.0 is recommended.

code structure:

```
    |--lipreading
    |----dataset        Dataset input function for estimator.
    |----model          Various models. New model should inherit from `base_estimator.BaseEstimator`.
    |----bin            Executable files for train/eval/predict on datasets.
```

**before running scripts, the following scripts should run first:**

```
source preprare_env.sh
```

**TODO LIST:**

dataset: 
* [x] add tf-records file generating scripts.
* [ ] data augment during training: random flip, drop/duplicate frames. 
* [ ] add lrw and lrs dataset
* [ ] add training hooks such as earlyStop, learning rate schedule

seq2seq:
* [x] add seq2seq-attention model

tcn: 
* [x] implement tcn
* [ ] implement tcn seq2seq
* [x] implement transformer
* [x] exp: adjust parameter of transformer
* [ ] merge tcn and transformer

utilize lip region:
* [ ] wait for ideas


## 1. dataset

### 1.1 GRID
[GRID corpus] (http://spandh.dcs.shef.ac.uk/gridcorpus/)

### 1.2 prepare dataset
put tfrecords in `./data/tf-records/`: 

- overlapped: overlapped_train.tfrecord, overlapped_test.tfrecord. 255 samples of each speaker are randomly selected to test set, the others in training set.
- unseen: unseen_train.tfrecord, unseen_test.tfrecord. Speaker 1,2,20,22 are selected to test set, the others in training set.

The format of tfrecord:

| key_name | type | Length | Content |
| --- | --- | --- | --- |
| video | bytes list (string) | Variable | List of image string of one image. Each element is image encoded with JPEG. |
| label | bytes (string) | 1 | String label. A blank ‘ ‘ is padded in the begining and end of the label. e.g. ' bin red in d nine again ' |
| align | int64 | Variable | The begin frame idx of each word + the end frame idx of the last word. len(align) = len(video word count) + 1 |

### 1.3 train/eval

#### 1.3.1 Lipnet: CNN+RNN+CTC

```
usage: grid_tcnCtc.py [-h] [--save_steps SAVE_STEPS] [--model_dir MODEL_DIR]
                      [--eval_steps EVAL_STEPS] [--ckpt_path CKPT_PATH]
                      [-gpu GPU] [-bw BEAM_WIDTH]
                      mode type
positional arguments:
  mode                  either train, eval, predict
  type                  either `unseen` or `overlapped`

optional arguments:
  -h, --help            show this help message and exit
  --save_steps SAVE_STEPS
  --model_dir MODEL_DIR
                        directory to save checkpoints
  --eval_steps EVAL_STEPS
                        steps to eval
  --ckpt_path CKPT_PATH
                        checkpoints to evaluate/predict
  -gpu GPU, --gpu GPU   gpu id to use
  -bw BEAM_WIDTH, --beam_width BEAM_WIDTH
```

Current Best result:

|model| type | steps |cer | wer | notes |
| --- | --- | --- | --- | --- | --- | 
| rnn+ctc | overlapped | 30337 | 1.55% | 4.05% | |
| tcn+ctc | overlapped | 87420 | 1.36% |3.61% | | 
| transformer | overlapped | 66960 | 1.24% | 2.59% | wer already reached 3.58% @ 13.95K steps|
| seq2seq | overlapped | 61850 | 1.64% | 3.14% | |

monitor in tensorboard: `tensorboard --logdir data/ckpts/tcn_ctc_overlapped/ --port 10301` and browse `http://localhost:10301` in chrome. Note to modify the port to a free port and ip to the real host ip.

##### Overlapped
- train: `python ./lipreading/bin/grid_ctc.py train overlapped --save_steps 465 --eval_steps=190 --model_dir ./data/ckpts/tcn_ctc_overlapped -gpu 0 -bw 4 -use_tcn`
- eval: `python ./lipreading/bin/grid_ctc.py eval overlapped --eval_steps=190 --ckpt_path ./data/ckpts/tcn_ctc_overlapped/model.ckpt-30337 -bw 4 -gpu 1 -use_tcn`

##### Unseen
- train: `python ./lipreading/bin/grid_ctc.py train unseen --save_steps 580 --eval_steps=80 --model_dir ./data/ckpts/tcn_ctc_unseen -gpu 0 -bw 4 -use_tcn`
- eval: `python ./lipreading/bin/grid_ctc.py eval unseen --eval_steps=80 --ckpt_path ./data/ckpts/tcn_ctc_unseen/model.ckpt-30337 -bw 4 -gpu 1 -use_tcn`

### 1.3.2 LipNet+Transformer

#### Overlapped
- train and eval: `python ./lipreading/bin/grid_transformer.py train overlapped --save_steps 465 --eval_steps=190 --model_dir ./data/ckpts/transformer_overlapped/ --gpu 0`

### 1.3.3 LipNet+seq2seq

#### Overlapped
- train and eval: `python ./lipreading/bin/grid_seq2seq.py train overlapped --save_steps 465 --eval_steps=190 --model_dir ./ckpt/ --gpu 0`
- eval: `python ./lipreading/bin/grid_seq2seq.py eval overlapped --eval_steps=190 --ckpt_path ./ckpt/model.ckpt-60450 --gpu 0`
