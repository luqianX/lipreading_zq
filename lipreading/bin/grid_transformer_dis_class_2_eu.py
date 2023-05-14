#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: grid_transformer.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/15
#   description:
#
#================================================================

import tensorflow.contrib.keras as keras
import os
import argparse
import tensorflow as tf
from collections import defaultdict
import sys
sys.path.append("/data/users/qianxilu/lipreading_paper/")
from lipreading.dataset.grid_dataset_dis_2 import grid_tfrecord_input_fn
#from lipreading.model.transformer_estimator_dis_class_2 import TransformerEstimator
from lipreading.model.transformer_estimator_dis_class_2_eu import TransformerEstimator

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='either train, eval, predict')
    parser.add_argument('type', help='either `unseen` or `overlapped`')

    # dataset path
    parser.add_argument(
        '--data_dir',
        help='directory of the tfrecord files',
        default=os.path.join(CURRENT_FILE_DIRECTORY,
                             '../../data/tf-records/GRID/mask'))

    # train
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='steps interval to save checkpoint')
    parser.add_argument('--model_dir', help='directory to save checkpoints')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help=
        'batch size. If train with multi_gpu, each of which will be fed with batch_size samples'
    )
    # eval
    parser.add_argument(
        '--eval_steps', type=int, default=None, help='steps to eval')
    # eval and predict
    parser.add_argument(
        '--ckpt_path', help='checkpoints to evaluate/predict', default=None)

    # misc
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='')
    parser.add_argument('-bw', '--beam_width', type=int, default=4)
    parser.add_argument(
        '--use_mask', action='store_true', help='wether to add mask to input')
    return parser.parse_args()


def transformer_params():
    """get default transformer params.
    Returns: Dict

    """
    return defaultdict(
        lambda: None,

        # Model params
        initializer_gain=1.0,  # Used in trainable variable initialization.
        vocab_size=29,  # Number of tokens defined in the vocabulary file.           token_num  
        hidden_size=256,  # Model dimension in the hidden layers.                    embed_dim: Dimension of token embedding.
        num_hidden_layers=
        3,  # Number of layers in the encoder and decoder stacks.
        num_heads=8,  # Number of heads to use in multi-headed attention.            head_num
        #filter_size=1024,  # Inner layer dimension in the feedforward network.       hidden_dim: Hidden dimension of feed forward layer.
        filter_size=1024,  # Inner layer dimension in the feedforward network.       hidden_dim: Hidden dimension of feed forward layer.

        # Dropout values (only used when training)
        layer_postprocess_dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,

        # Training params
        label_smoothing=0.2,
        learning_rate=2.0,
        learning_rate_decay_rate=1.0,
        learning_rate_warmup_steps=16000, 

        # Optimizer params
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.997,
        optimizer_adam_epsilon=1e-09,

        # Default prediction params
        extra_decode_length=50,
        beam_size=4,
        alpha=0.6,  # used to calculate length normalization in beam search
        # allow_ffn_pad=True,
    )


def main():
    args = arg_parse()
    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)

    multi_gpu = len(args.gpu.split(',')) > 1
    # build estimator
    run_config = TransformerEstimator.get_runConfig(
        args.model_dir,
        args.save_steps,
        multi_gpu=multi_gpu,
        keep_checkpoint_max=200)

    model_parms = transformer_params()
    model_parms.update({
        # lipnet
        'feature_len': 512,

        # learn
        'batch_size': args.batch_size,
    })

    model = TransformerEstimator(model_parms, run_config)
    
    # build input
    train_file = os.path.join(args.data_dir,
                            '/data/users/qianxilu/lipreading_paper/data/tfrecord/unseen_train_dis_grid_2_900.tfrecord'.format(args.type))

    test_file = os.path.join(args.data_dir,
                              '/data/users/qianxilu/lipreading_paper/data/tfrecord/unseen_test_dis_grid_2_lip.tfrecord'.format(args.type))


    train_input_params = {
        'num_epochs': 10000,
        'batch_size': args.batch_size,
        'num_threads': 4,
        'use_mask': args.use_mask,
        'file_name_pattern': train_file
    }
    eval_input_params = {
        'num_epochs': 1,
        'batch_size': 32,
        'num_threads': 4,
        'use_mask': args.use_mask,
        'file_name_pattern': test_file
    }
    print('train_input_params: {}'.format(train_input_params))
    train_input_fn = lambda: grid_tfrecord_input_fn(mode=tf.estimator.ModeKeys.TRAIN, **train_input_params)
    eval_input_fn = lambda: grid_tfrecord_input_fn(mode=tf.estimator.ModeKeys.EVAL, **eval_input_params)

    #begin train,eval,predict
    if args.mode == 'train':
        model.train_and_evaluate(
            train_input_fn,
            eval_input_fn,
            eval_steps=args.eval_steps,
            throttle_secs=200)
    elif args.mode == 'eval':
        res = model.evaluate(
            eval_input_fn,
            steps=args.eval_steps,
            checkpoint_path=args.ckpt_path)
        print(res)
    elif args.mode == 'predict':
        model.predict(eval_input_fn, checkpoint_path=args.ckpt_path)
    else:
        raise ValueError(
            'arg mode should be one of "train", "eval", "predict"')


if __name__ == "__main__":
    main()
