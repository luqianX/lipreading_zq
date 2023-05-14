#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: transformer_estimator.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/14
#   description:
#
#================================================================

import os
import string
import tensorflow as tf
import tensorflow.contrib.keras as keras

from ..util import label_util
from .base_estimator import BaseEstimator

from .transformer_official.utils import metrics
from .transformer_official.model.transformer import Transformer
from .transformer_official.utils.tokenizer import PAD, PAD_ID, EOS, EOS_ID, RESERVED_TOKENS
from .res_decoder import Decoder as re_decoder
from .classfication_only_class_2 import Classfication as classfica

#PAD = "P"
#PAD_ID = 0
#EOS = "E"
#EOS_ID = 1
#RESERVED_TOKENS = [PAD, EOS]

class TransformerEstimator(BaseEstimator):
    """docstring for TransformerEstimator"""

    DEFAULT_DIC =  list(string.ascii_lowercase) + [' '] 

    def __init__(self, params, run_config, **kwargs):
        self.DIC = params.get('dic', self.DEFAULT_DIC)
        self.DIC = RESERVED_TOKENS + self.DIC
        print("DICT: {}".format(self.DIC))
        params.update({'target_vocab_size': len(self.DIC)})
        super(TransformerEstimator, self).__init__(params, run_config, **kwargs)

    def preprocess_labels(self, labels):
        """ preprocess labels to satisfy the need of model.

        Args:
            labels: 1-D Tensor. labels of shape (batch_size,). For example: [ 'ab haha', 'hha fd fd']

        Returns: 2-D int32 Tensor with shape  [batch_size, T]  . labels with numeric shape. EOS is padded to each label.

        """
        pad_tensor = tf.constant([EOS], tf.string)
        # append 'E' to label
        eos = tf.tile(pad_tensor, tf.shape(labels))  
        labels = tf.string_join([labels, eos])   

        label_char_list = label_util.string2char_list(labels)  # B x T   

        numeric_label = label_util.string2indices(
            label_char_list,
            dic=self.DIC)  # SparseTensor, dense_shape: [B x T]       
        numeric_label = tf.sparse_tensor_to_dense(
            numeric_label, default_value=PAD_ID)  #[B x T]

        
        return numeric_label

    def id_to_string(self, predictions):
        """convert predictions to string.

        Args:
            predictions: 3-D int64 Tensor with shape: [batch_size, T, vocab_size]

        Returns: 1-D string SparseTensor with dense shape: [batch_size,]

        """
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=PAD_ID)  # remove PAD_ID
        predictions = tf.sparse_tensor_to_dense(predictions, EOS_ID)
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=EOS_ID)  # remove EOS_ID
        predicted_char_list = label_util.indices2string(predictions, self.DIC)
        predicted_string = label_util.char_list2string(
            predicted_char_list)  # ['ab', 'abc']
        return predicted_string

    def model_fn(self, features, labels, mode, params):
        """ Model function for transformer.

        Args:
            features: float Tensor with shape [batch_size, T, H, W, C]. Input sequence.
            labels: string Tensor with shape [batch_size,]. Target labels.
            mode: Indicate train or eval or predict.
            params: dict. model parameters.

        Returns: tf.estimator.EstimatorSpec.

        """
    #    learning_rate = params.get('learning_rate', 0.001)

        in_training = mode == tf.estimator.ModeKeys.TRAIN

        video_grid_1 = features['video_grid_1']
#        video_grid_2 = features['video_grid_2']

        class_a=tf.squeeze(labels['class_a'])
        inputs_unpadded_length = features['unpadded_length']  #50
#        inputs_unpadded_length_2 = features['unpadded_length_2']  #50

        if params.get('feature_extractor') == 'early_fusion':
            from .cnn_extractor_dis_t_2 import EarlyFusion2D as CnnExtractor
        else:
            from .cnn_extractor_dis_t_2 import LipNet as CnnExtractor

        feature_extractor = CnnExtractor(
            feature_len=params.get('hidden_size'),
            training=in_training,
#            training=False,
            scope='cnn_feature_extractor')
        transformer = Transformer(params, in_training)
        
        inputs_grid_1 ,x_grid_on_1,x_grid_down_1,= feature_extractor.build(video_grid_1)  # [batch_size, input_length, hidden_size]  [B,75,512]
#        inputs_grid_2 ,x_grid_on_2,x_grid_down_2,= feature_extractor.build(video_grid_2)  # [batch_size, input_length, hidden_size]  [B,75,512]
        x_grid1_on_50=tf.slice(x_grid_on_1,[0,0,0,0,0],[32,-1,-1,-1,-1,])
        x_grid1_down_50=tf.slice(x_grid_down_1,[0,0,0,0,0],[-1,-1,-1,-1,-1,])
        x_grid1 = tf.slice(video_grid_1,[0,0,0,0,0],[32,-1,-1,-1,-1,])
        x_grid1_top = tf.slice(video_grid_1,[0,0,0,0,0],[16,-1,-1,-1,-1,])
        x_grid1_last = tf.slice(video_grid_1,[0,0,0,0,0],[16,-1,-1,-1,-1,])
        x_grid1 = tf.concat([x_grid1_last, x_grid1_top] , 0)
#        x_grid1_on_50_32=tf.slice(x_grid_on_1,[0,0,0,0,0],[16,75,-1,-1,-1,])
#        x_grid1_down_50_32=tf.slice(x_grid_down_1,[16,0,0,0,0],[16,75,-1,-1,-1,])
        x_grid1_down_50_1=tf.slice(x_grid_down_1,[0,0,0,0,0],[16,-1,-1,-1,-1,])
        x_grid1_down_50_2=tf.slice(x_grid_down_1,[16,0,0,0,0],[16,-1,-1,-1,-1,])
        x_grid1_down_50_32 = tf.concat( [x_grid1_down_50_2, x_grid1_down_50_1], 0)        




        res_decoder= re_decoder(
        #    training=in_training,
            training=False,
            scope='res_decoder'  
        )
#        out_res_deocder_grid_1=res_decoder.build(x_grid1_on_50,x_grid1_down_50)
        out_res_deocder_grid_1=res_decoder.build(x_grid1_on_50,x_grid1_down_50_32)
        loss_res_grid_1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(out_res_deocder_grid_1 - x_grid1))))

        classfi=classfica(
            training=in_training,
        #    training=False,
            scope='Classfication'  
        )
        fc1,fc1_softmax=classfi.build(x_grid1_on_50)
        fc2,fc2_softmax=classfi.build(x_grid1_on_50)
#        fc2,fc2_softmax=classfi.build(x_grid1_down_50)

#        loss_class_1=tf.constant(0.0)    
        loss_class_1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc1, labels=class_a))
        loss_class_1 = loss_class_1 + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=class_a))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc1_softmax, -1), tf.argmax(class_a, -1)), tf.float32))
#        accuracy = tf.constant(0.0)
        loss_class_eu=tf.constant(0.0)
        con= tf.constant(1/33, shape=[33])
    

        # train and eval
        labels_grid_1 = tf.squeeze(labels['label_grid_1'])  # [batch_size, ]  默认删除所有为1的维度
        char_list_labels_grid_1 = label_util.string2char_list(labels_grid_1)  #[ 'ab'] -> [ 'a', 'b']
        targets_grid_1 = self.preprocess_labels(labels_grid_1)  # [batch_size, target_length] 每个字符的索引？
#        logits_grid_1 = transformer(x_grid_down_1, inputs_unpadded_length, targets_grid_1)
        logits_grid_1 = transformer(inputs_grid_1, inputs_unpadded_length, targets_grid_1)
        # Calculate model loss.
        # xentropy contains the cross entropy loss of every nonpadding token in the
        xentropy_grid_1, weights_grid_1 = metrics.padded_cross_entropy_loss(
            logits_grid_1, targets_grid_1, params["label_smoothing"], params["vocab_size"])
        loss_lip_grid_1 = tf.reduce_sum(xentropy_grid_1) / tf.reduce_sum(weights_grid_1)
        

        
        loss = loss_lip_grid_1  + 0.1*loss_class_1 + 0.1*loss_class_eu   + 1e-5*(loss_res_grid_1 - 100)


        if mode == tf.estimator.ModeKeys.TRAIN:

        #    train_op, metric_dict = get_train_op_and_metrics(loss_lip,loss_res,loss_class ,params)
            train_op, metric_dict = get_train_op_and_metrics(loss, params)
            variables = tf.contrib.framework.get_variables_to_restore() #得到该网络中，所有可以加载的参数
            #print(variables)
            var_conv_restore = [val for val in variables if (
                                            (val.name.split('/')[0]=='res_decoder'))] 
            with tf.Session() as sess:
                saver = tf.train.Saver(var_conv_restore)
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                saved_model='/data/users/qianxilu/lipreading_paper/data/ckpt/model.ckpt-10000'
                saver.restore(sess=sess, save_path= saved_model)
                print ("load model from %s successfully" %saved_model)
   

            # Epochs can be quite long. This gives some intermediate information
            # in TensorBoard.
            metric_dict["minibatch_loss"] = loss
            metric_dict["loss_class"] = loss_class_1
            metric_dict["loss_class_eu"] = loss_class_eu
#            metric_dict["loss_class_eu_on"] = loss_class_eu_on
            metric_dict["loss_lip"] = loss_lip_grid_1
            metric_dict["class_accuracy"] = accuracy
            metric_dict["loss_res"] = loss_res_grid_1
            record_scalars(metric_dict)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

            # Save loss as named tensor that will be logged with the logging hook.
            #tf.identity(loss, "cross_entropy")

        if mode == tf.estimator.ModeKeys.EVAL:
        #    logits = transformer(x_grid_down_1, inputs_unpadded_length, None)
            logits = transformer(inputs_grid_1, inputs_unpadded_length, None)
            predictions = logits['outputs']  # [batch_size, T]
            predicted_string = self.id_to_string(predictions)
            predicted_char_list = label_util.string2char_list(predicted_string)

            # calculate metrics
            cer, wer = self.cal_metrics(char_list_labels_grid_1, predicted_char_list)
            tf.summary.scalar('cer', tf.reduce_mean(cer))
            tf.summary.scalar('wer', tf.reduce_mean(wer))
            tf.summary.scalar('accuracy', accuracy)
            eval_metric_ops = {
                'cer': tf.metrics.mean(cer),
                'wer': tf.metrics.mean(wer),
                'accuracy':tf.metrics.mean(accuracy),
               # 'wer': tf.reduce_mean(wer),
               # 'wer': tf.reduce_mean(wer),
            }

            logging_hook = tf.train.LoggingTensorHook(
                {
                    'loss': loss,
                    'loss_res_grid_1': loss_res_grid_1,
                #    'loss_res_grid_2': loss_res_grid_2,
                #    'loss_cons':loss_cons,
                    #'loss_res_grid': loss_res_b2,
                    'loss_lip' :loss_lip_grid_1,
                    'loss_class_eu':loss_class_eu,
#                    'loss_class_eu_on':loss_class_eu_on,
                    'loss_class_1': loss_class_1,
                #    'loss_class_2': loss_class_2,
                    'accuracy': tf.reduce_mean(accuracy),
                    'cer': tf.reduce_mean(cer),
                    'wer': tf.reduce_mean(wer),
                    'predicted': predicted_string[:5],
                    'labels_grid_1': labels_grid_1[:5]
                },
                every_n_iter=10)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                predictions={"predictions": predicted_string},
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[logging_hook]
                # eval_metric_ops=metrics.get_eval_metrics(
                # logits, targets, params)
            )


def record_scalars(metric_dict):
    for key, value in metric_dict.items():
        tf.summary.scalar(name=key, tensor=value)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size**-0.5)
        # Apply linear warmup
        # step /= 10.0
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # learning_rate *= 0.1
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")

        return learning_rate

def learning_rate_decay():
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        
        step = tf.to_float(tf.train.get_or_create_global_step())
        x=tf.constant(5000.0)
        y=tf.constant(15000.0)
        learning_rate=tf.cond(step<=x,lambda: 0.0001, lambda: 0.0003)
        learning_rate=tf.cond(step > y,lambda: 0.0001, lambda: learning_rate)
        tf.identity(learning_rate, "learning_rate")

        return learning_rate

def get_train_op_and_metrics(loss, params):
#def get_train_op_and_metrics(loss_lip,loss_res,loss_class, params):
    """Generate training op and metrics to save in TensorBoard."""
    with tf.variable_scope("get_train_op"):
    #    learning_rate = 0.0001
    #    learning_rate = learning_rate_decay()
        learning_rate = get_learning_rate(
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
        #learning_rate=params["learning_rate"]
        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster

#        print('a:',a)
#        gradients = gradient_lip + gradient_res + grad_class
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        
        all_variables = tvars
        all_gradients = tf.gradients(loss, all_variables)
        #print('all_gradients',all_gradients)
        #print('len all_gradients',len(all_gradients)) #205

        shared_vars = all_variables[:14]
        shared_subnet_gradients = all_gradients[:14]
        print('shared_subnet_gradients',shared_subnet_gradients)
    #    print('shared_vars',shared_vars)
        res_vars =  all_variables[14:(14+8)]
        res_gradients = all_gradients[14:(14+8)]
        print('res_gradients',res_gradients)
        class_vars = all_variables[(14+8):(14+8+8)]
        class_gradients = all_gradients[(14+8):(14+8+8)]
        print('class_gradients',class_gradients)
        transformer_vars  = all_variables[(14+8+8):]
        transformer_gradients = all_gradients[(14+8+8):]
        print('transformer_gradients',transformer_gradients)
#        gradients=shared_subnet_gradients+res_gradients+class_gradients+transformer_gradients
        for i in res_vars :
            shared_vars.append(i)
        for i in transformer_vars:
            shared_vars.append(i)
        for j in res_gradients:
            shared_subnet_gradients.append(j)
        for j in transformer_gradients:
            shared_subnet_gradients.append(j)
        shared_subnet_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=params["optimizer_adam_beta1"],
            beta2=params["optimizer_adam_beta2"],
            epsilon=params["optimizer_adam_epsilon"])
        class_optimizer = tf.train.AdamOptimizer(0.1*learning_rate,beta1=params["optimizer_adam_beta1"],
            beta2=params["optimizer_adam_beta2"],
            epsilon=params["optimizer_adam_epsilon"])
      
#        train_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=params["optimizer_adam_beta1"],
#            beta2=params["optimizer_adam_beta2"],
#            epsilon=params["optimizer_adam_epsilon"])
#        train_op=train_optimizer.apply_gradients(zip(all_gradients, all_variables),global_step=global_step,name="train_cnn")
        train_shared_op = shared_subnet_optimizer.apply_gradients(zip(shared_subnet_gradients, shared_vars),name="train_cnn")
        train_class_op = class_optimizer.apply_gradients(zip(class_gradients, class_vars),global_step=global_step,name="train_class")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        train_op = tf.group(train_op,update_ops)    
        train_op = tf.group(train_shared_op, train_class_op,update_ops)    
        #train_op = tf.group(train_shared_op, train_res_op, train_class_op,train_transformer_op,update_ops)    
        
        
#        minimize_op = optimizer.apply_gradients(
#            gradients, global_step=global_step, name="train")
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        train_op = tf.group(minimize_op, update_ops)

        train_metrics = {"learning_rate": learning_rate}

        # gradient norm is not included as a summary when running on TPU, as
        # it can cause instability between the TPU and the host controller.
#        gradient_norm = tf.global_norm(list(zip(*gradients))[0])
#        train_metrics["global_norm/gradient_norm"] = gradient_norm

        return train_op, train_metrics
