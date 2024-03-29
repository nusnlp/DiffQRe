# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import random
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import json


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import torch.quantization.quantize as QT
from torch.autograd import Variable



from .activations import ACT2FN
from .configuration_bert import BertConfig
from .file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .utils import logging
from .modeling_wrapper import WRAPPERS, load_wrapper_config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


## read the configuration file
#training_mode = {
#    'prune_top_layer': False,
#    'prune_low_layer': False,
#    'prune_dynamic': False,
#    'prune_adapter_top_layer': False,
#    'prune_adapter_low_layer': False,
#    'prune_adapter_dynamic': False
#}


prune_keep_top_layer = False ## keep top layers
prune_keep_low_layer = False ## keep 
prune_dynamic = False
prune_keep_top_layer_dynamic = False
prune_keep_low_layer_dynamic = False
prune_dynamic_succession = False
prune_adapter_keep_top_layer = False
prune_adapter_keep_low_layer = False
prune_adapter_dynamic = False
prune_adapter_keep_top_layer_dynamic = False
prune_adapter_keep_low_layer_dynamic = False

#filename = "/home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/glue/bin/config.json"
#config_training = json.load(open(filename, 'r'))
#training_mode = config_training['mode']
#print (training_mode)
"""
for mode in training_mode:
    if training_mode[mode] == True:
        if mode == 'prune_keep_top_layer':
            prune_keep_top_layer = True
        elif mode == 'prune_keep_low_layer':
            prune_keep_low_layer = True
        elif mode == 'prune_dynamic':
            prune_dynamic = True
        elif mode == 'prune_keep_top_layer_dynamic':
            prune_keep_top_layer_dynamic = True
        elif mode == 'prune_keep_low_layer_dynamic':
            prune_keep_low_layer_dynamic = True
        elif mode == 'prune_dynamic_succession':
            prune_dynamic_succession = True
        elif mode == 'prune_adapter_keep_top_layer':
            prune_adapter_keep_top_layer = True
        elif mode == 'prune_adapter_keep_low_layer':
            prune_adapter_keep_low_layer = True
        elif mode == 'prune_adapter_dynamic':
            prune_adapter_dynamic = True
        elif mode == 'prune_adapter_keep_top_layer_dynamic':
            prune_adapter_keep_top_layer_dynamic = True
        elif mode == 'prune_adapter_keep_low_layer_dynamic':
            prune_adapter_keep_low_layer_dynamic = True
        break

layer_keep = int(config_training['layer_keep'])
if_agent_token = config_training['if_agent_token']

print('\n************************************')
for mode in training_mode:
    if training_mode[mode] == True:
        print('\ttraining_mode: %s\n' % (mode))
        break
print('\tlayer_keep: %d' %(layer_keep))
if if_agent_token:
    print('\tif_agent_token is *True*')
else:
    print('\tif_agent_token is *False*')
print('\n************************************')
"""



### gumbel_softmax ###
def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, doing_sample = True):
    if doing_sample:
        y = logits + sample_gumbel(logits.size())
    else:
        y = logits
    #print (F.softmax(y / temperature, dim=-1))
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5, doing_sample = True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, doing_sample)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    #print((y_hard - y).detach() + y)
    return (y_hard - y).detach() + y
### gumbel_softmax ###



def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        #mixed_query_layer = self.query(self.adapter_for_LayerNorm(self.tanh(self.adapter_for_query(hidden_states))))
        #PET
        #mixed_query_layer = self.query(self.dropout(self.adapter_PET(hidden_states)))

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            #mixed_key_layer = self.key(self.adapter_for_LayerNorm(self.tanh(self.adapter_for_key(encoder_hidden_states))))
            #mixed_value_layer = self.value(self.adapter_for_LayerNorm(self.tanh(self.adapter_for_value(encoder_hidden_states))))
            #mixed_key_layer = self.key(self.dropout(self.adapter_PET(encoder_hidden_states)))
            #mixed_value_layer = self.value(self.dropout(self.adapter_PET(encoder_hidden_states)))
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            #mixed_key_layer = self.key(self.adapter_for_LayerNorm(self.tanh(self.adapter_for_key(hidden_states))))
            #mixed_value_layer = self.value(self.adapter_for_LayerNorm(self.tanh(self.adapter_for_value(hidden_states))))
            #mixed_key_layer = self.key(self.dropout(self.adapter_PET(hidden_states)))
            #mixed_value_layer = self.value(self.dropout(self.adapter_PET(hidden_states)))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)


        ##PET
        #context_layer = self.dropout(self.adapter_PET(context_layer))


        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


### PET
class Adapter_PET(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_down = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.dense_up = nn.Linear(int(config.hidden_size/2), config.hidden_size)

        self.activation = nn.Tanh()

    def forward(self, hidden_states):
    
        return self.dense_up(self.activation(self.dense_down(hidden_states))) + hidden_states


class Agent(nn.Module):
    def __init__(self, config, num_class = 2):
        super().__init__()
        self.num_class = num_class
        #self.dense = nn.Linear(config.hidden_size, self.num_class)
        self.dense1 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.dense2 = nn.Linear(int(config.hidden_size/2), self.num_class)
        #self.dense3 = nn.Linear(int(config.hidden_size/2), config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        # batch, seq, hiddenm
        # att_mask: batch, 1, 1, seq
        batch, seq, hidden_size = hidden_states.size()
        #attention_mask = attention_mask.view(batch, seq, 1)
        #print(attention_mask)

        #print(hidden_states.size())
        #print(attention_mask.size())
        
        ## option 1
        #hidden_states = torch.mean(hidden_states, 1).view(batch, -1)
        #return self.dense(hidden_states) #batch, 2

        ## option 2
        #hidden_states =  torch.sum(hidden_states * attention_mask.view(batch, seq, 1), dim = 1).view(batch,  -1) / torch.sum(attention_mask, dim = 1).view(batch, 1)
        #hidden_states = hidden_states.view(batch, -1)

        hidden_states = torch.mean(hidden_states, 1).view(batch, -1)
        return self.dense2(self.activation(self.dense1(hidden_states)))
        ## option 3
        #hidden_states = self.dense3(self.activation(self.dense1(hidden_states))) #+ hidden_states
        #return self.dense(self.activation(hidden_states))


class Agent_token(nn.Module):
    def __init__(self, config, num_class = 2):
        super().__init__()
        self.num_class = num_class
        #self.dense = nn.Linear(config.hidden_size, self.num_class)
        self.dense1 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.dense2 = nn.Linear(int(config.hidden_size/2), self.num_class)
        #self.dense3 = nn.Linear(int(config.hidden_size/2), config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        # batch, seq, hidden
        batch, seq, hidden_size = hidden_states.size()
        ## option 2
        #hidden_states = torch.mean(hidden_states, 1).view(batch, -1)
        return self.dense2(self.activation(self.dense1(hidden_states))) # batch, seq, num_class
        ## option 3
        #hidden_states = self.dense3(self.activation(self.dense1(hidden_states))) #+ hidden_states
        #return self.dense(self.activation(hidden_states))


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layer_id = 0

        ##PET
        self.adapter_PET = Adapter_PET(config)

        ##Spot
        #self.agent = Agent(config)

    def forward(self, hidden_states, input_tensor, mask = None):
        if prune_keep_top_layer or prune_keep_low_layer or prune_dynamic or prune_dynamic_succession or\
               prune_keep_top_layer_dynamic or prune_keep_low_layer_dynamic:
            ### Mode 1 ####
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)

            ##PET
            hidden_states = self.adapter_PET(hidden_states)
            hidden_states = self.dropout(hidden_states)

            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        elif prune_adapter_keep_top_layer:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id >= 12 - layer_keep:
                hidden_states = self.dropout(self.adapter_PET(hidden_states))
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states
        
        elif prune_adapter_keep_low_layer:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id <= layer_keep - 1:
                hidden_states = self.dropout(self.adapter_PET(hidden_states))
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        elif prune_adapter_dynamic:
            f1 = self.dropout(self.dense(hidden_states))
            f2 = self.dropout(self.adapter_PET(f1))
            x = f1 * (1 - mask) + f2 * mask
            x = self.LayerNorm(x + input_tensor)
            return x

        elif prune_adapter_keep_top_layer_dynamic:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id >= 12 - layer_keep:
                f2 = self.dropout(self.adapter_PET(hidden_states))
                x = hidden_states * (1 - mask) + f2 * mask
                #-->
                #x = f2 * mask 
            else:
                x = hidden_states
            x = self.LayerNorm(x + input_tensor)
            return x

        elif prune_adapter_keep_low_layer_dynamic:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id <= layer_keep - 1:
                f2 = self.dropout(self.adapter_PET(hidden_states))
                x = hidden_states * (1 - mask) + f2 * mask
            else:
                x = hidden_states
            x = self.LayerNorm(x + input_tensor)
            return x



        '''
        ## spot tune
        batch, seq, hidden_size = hidden_states.size()

        f1 = self.dropout(self.dense(hidden_states))
        f2 = self.dropout(self.adapter_PET(f1))

    
        x = f1*(1-mask) + f2*mask
        x = self.LayerNorm(x + input_tensor)

        return x 
        '''


        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        #if self.layer_id < 3:
        if self.layer_id < 6:
        #if self.layer_id >= 9:
        #if self.layer_id >= 6:
            ##PET
            hidden_states = self.adapter_PET(hidden_states)
            hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        '''


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        mask = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states, mask)
        #print(mask.size())
        #print('---')
        #outputs = (attention_output,) + self_outputs[1:] # add attentions if we output them
        # add mask
        outputs = (attention_output,)  + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        ##PET
        self.adapter_PET = Adapter_PET(config)

        ## Agent
        #self.agent = Agent(config)

        self.layer_id = 0

    def forward(self, hidden_states, input_tensor, mask = None):
        if prune_keep_top_layer or prune_keep_low_layer or prune_dynamic or prune_dynamic_succession or\
                prune_keep_top_layer_dynamic or prune_keep_low_layer_dynamic:
            ### Mode 1 ####
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)

            ##PET
            hidden_states = self.adapter_PET(hidden_states)
            hidden_states = self.dropout(hidden_states)

            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        elif prune_adapter_keep_top_layer:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id >= 12 - layer_keep:
                hidden_states = self.dropout(self.adapter_PET(hidden_states))
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        elif prune_adapter_keep_low_layer:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id <= layer_keep - 1:
                hidden_states = self.dropout(self.adapter_PET(hidden_states))
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        elif prune_adapter_dynamic:
            f1 = self.dropout(self.dense(hidden_states))
            f2 = self.dropout(self.adapter_PET(f1))
            x = f1 * (1 - mask) + f2 * mask
            x = self.LayerNorm(x + input_tensor)
            return x

        elif prune_adapter_keep_top_layer_dynamic:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id >= 12 - layer_keep:
                f2 = self.dropout(self.adapter_PET(hidden_states))
                x = hidden_states * (1 - mask) + f2 * mask
                # -->
                #x = f2 * mask
            else:
                x = hidden_states
            x = self.LayerNorm(x + input_tensor)
            return x

        elif prune_adapter_keep_low_layer_dynamic:
            hidden_states = self.dropout(self.dense(hidden_states))
            if self.layer_id <= layer_keep - 1:
                f2 = self.dropout(self.adapter_PET(hidden_states))
                x = hidden_states * (1 - mask) + f2 * mask
            else:
                x = hidden_states
            x = self.LayerNorm(x + input_tensor)
            return x


        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        ##PET
        hidden_states = self.adapter_PET(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        '''


        '''
        ## spot tune
        batch, seq, hidden_size = hidden_states.size()

        f1 = self.dropout(self.dense(hidden_states))
        f2 = self.dropout(self.adapter_PET(f1))

        x = f1*(1-mask) + f2*mask
        x = self.LayerNorm(x + input_tensor)
        return x
        '''


        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)


        #if self.layer_id < 3:
        if self.layer_id < 6:
        #if self.layer_id >= 9:
        #if self.layer_id >= 6:
            #print (self.layer_id)
            ##PET
            hidden_states = self.adapter_PET(hidden_states)
            hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        '''




        

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config) ## 888
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        #self.agent = Agent(config)

    def forward(
        self,
        mask,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

       
        #layer_output = apply_chunking_to_forward(
        #    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output[0], attention_output[1].expand_as(attention_output[0])
        #)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, mask.expand_as(attention_output)
        )

        if prune_adapter_dynamic or prune_adapter_keep_top_layer or prune_adapter_keep_low_layer or prune_adapter_keep_top_layer_dynamic\
            or prune_adapter_keep_low_layer_dynamic:
            pass
        elif prune_keep_top_layer:
            if self.output.layer_id < 12 - layer_keep:
                layer_output = hidden_states
        elif prune_keep_low_layer:
            if self.output.layer_id > layer_keep - 1:
                layer_output = hidden_states

        elif prune_keep_top_layer_dynamic:
            if self.output.layer_id >= 12 - layer_keep:
                layer_output = hidden_states * (1 - mask) + layer_output * mask
            else:
                layer_output = hidden_states


        elif prune_keep_low_layer_dynamic:
            if self.output.layer_id <= layer_keep - 1:
                layer_output = hidden_states * (1 - mask) + layer_output * mask
            else:
                layer_output = hidden_states


        elif prune_dynamic or prune_dynamic_succession:
            layer_output = hidden_states * (1 - mask) + layer_output * mask



        ### add mask
        #layer_output = hidden_states*(1-mask) + layer_output*mask

        ## drop top layer
        #if self.output.layer_id >= 6:
        #    layer_output = hidden_states



        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output, mask):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, mask)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return  BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class BertEncoderWithAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        for i, layer_module in enumerate(self.layer):
            layer_module.attention.output.layer_id = i #999
            layer_module.output.layer_id = i

        #self.agent = nn.ModuleList([Agent(config) for _ in range(config.num_hidden_layers)])
        self.agent = Agent(config, len(self.layer) * 2)
        self.agent_token = Agent_token(config, len(self.layer) * 2)
        self.agent_layer = Agent(config, len(self.layer))

        self.agent_state = True
        self.time_change = 0

        #self.masks = Variable(torch.tril(torch.ones(len(self.layer), len(self.layer))), dtype = torch.float, requires_grad = False, device = 'cuda'))


        self.masks = Variable(torch.tensor(torch.tril(torch.ones(len(self.layer), len(self.layer))), dtype = torch.float, requires_grad = False, device = 'cuda'))
        
        self.masks_adapter = Variable(torch.tensor(torch.tril(torch.ones(len(self.layer), len(self.layer))).transpose(0, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))
        

        #self.masks = Variable(torch.tril(torch.ones(len(self.layer), len(self.layer)).clone().detach(), dtype = torch.float, requires_grad = False, device = 'cuda'))


        #self.M = Variable(torch.tensor((np.random.rand(len(self.layer), 2)-0.5), dtype = torch.float, requires_grad = True, device = 'cuda'))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None



        '''
        ## get the actions
        masks = []
        output_mask = 0
        batch, seq, hidden_size = hidden_states.size()
        for i, agent_module in enumerate(self.agent):
            probs = agent_module(hidden_states) # batch, 2
            action = gumbel_softmax(probs.view(batch, 1, -1)).expand(batch, seq, 2)
            policy = action[:, :, 1].contiguous() ### 77777
            mask = policy.float().view(batch, seq, 1)
            masks.append(mask)
            #output_mask += mask[0, 0][0]
        #print(output_mask)
        '''

        

        if prune_keep_low_layer_dynamic:

            if if_agent_token:
                ### Token level Agent:
                batch, seq, hidden_size = hidden_states.size()
                #if self.agent.training is True or True:
                probs = self.agent_token(hidden_states) # batch, seq, 12 * 2
                probs = probs.view(probs.size(0), probs.size(1), -1, 2)
                action = gumbel_softmax(probs, temperature = 5) # batch, seq, 12, 2 
                policy = action[:, :, :, 1].contiguous() # batch, seq, 12, 1
                masks = policy.float().view(policy.size(0), policy.size(1), policy.size(2), 1).permute(0, 2, 1, 3) #.expand(policy.size(0), -1, seq, 1) # batch, 12, seq, 1

                #### make masks to [0, 0.9]
                #masks = ((masks - 0.1) > 0) * 1. * (masks - 0.1)


                    #print(masks.size())
                    #print(torch.sum(masks[0, :, 0]))
                #else:
                #    probs = self.agent_token(hidden_states).view(batch, seq, -1, 2) # batch, 12 * 2
                #    _, action = torch.max(probs, dim = 3) # batch, seq, 12, 1
                #    policy = action
                #    masks = policy.float().view(policy.size(0), policy.size(1), policy.size(2), 1).permute(0, 2, 1, 3) #.view(policy.size(0), -1, 1, 1).expand(policy.size(0), -1, seq, 1)


            else:
                '''
                ### Another way to get the actions
                batch, seq, hidden_size = hidden_states.size()
                #if self.agent.training is True:
                probs = self.agent(hidden_states) # batch, 12 * 2
                probs = probs.view(probs.size(0), -1, 2)
                action = gumbel_softmax(probs, temperature = 5) # batch, 12, 2 
                #print(action.data)
                policy = action[:, :, 1].contiguous() # batch, 12, 1
                masks = policy.float().view(policy.size(0), -1, 1, 1).expand(policy.size(0), -1, seq, 1) # batch, 12, seq, 1
                '''
                '''
                batch, seq, hidden_size = hidden_states.size()
                layer_head_mask = head_mask[0] if head_mask is not None else None
                masks = Variable(torch.tensor(torch.zeros(batch, seq, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))

                layer_outputs = self.layer[0](
                    masks, #masks[i],
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    )
                hidden_states_policy = layer_outputs[0]

                layer_head_mask = head_mask[1] if head_mask is not None else None
                layer_outputs = self.layer[1](
                    masks, #masks[i],
                    hidden_states_policy,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
                hidden_states_policy = layer_outputs[0]
                '''
                ## another way
                batch, seq, hidden_size = hidden_states.size()
                if self.time_change <= 30:
                    action = Variable(torch.tensor(torch.zeros(batch, 12, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))
                    action[:, 11] = 1
                else:
                    #action = Variable(torch.tensor(torch.zeros(batch, 12, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))
                    #action[:, 5] = 1
                    probs = self.agent_layer(hidden_states)# batch, 12
                    action = gumbel_softmax(probs[:, :layer_keep], temperature = 1).view(batch, -1, 1) # batch, layer_keep, 1 
                    #if self.time_change <=20:
                    #    action = F.gumbel_softmax(probs[:, layer_keep-2: layer_keep], dim = 1, hard = False, tau = 5).view(batch, -1, 1)
                    #else:
                    #    action = F.gumbel_softmax(probs[:, layer_keep-2: layer_keep], dim = 1, hard = True, tau = 1).view(batch, -1, 1)
                    #action = torch.cat([Variable(torch.tensor(torch.zeros(batch, layer_keep-2, 1), dtype = float, requires_grad = False, device = 'cuda')), action,\
                    #        Variable(torch.tensor(torch.zeros(batch, 12-layer_keep, 1), dtype = float, requires_grad = False, device = 'cuda'))], dim = 1)


                '''
                if self.time_change <= 40:
                    action = Variable(torch.tensor(torch.zeros(batch, 12, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))
                    action[:, 4] = 1
                elif self.time_change > 40 and self.time_change <=80:
                    action = Variable(torch.tensor(torch.zeros(batch, 12, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))
                    action[:, 5] = 1
                elif self.time_change > 80 and self.time_change <= 120:
                    action = Variable(torch.tensor(torch.zeros(batch, 12, 1), dtype = torch.float, requires_grad = False, device = 'cuda'))
                    action[:, 6] = 1
                '''



                masks = self.masks.view(1, len(self.layer), -1).expand(batch, len(self.layer), len(self.layer)) # batch, 12, 1230
                policy = action.contiguous()
                policy = policy.float()
                masks = torch.sum(masks * policy, dim = 1).view(batch, -1, 1, 1).expand(batch, -1, seq, 1) # batch, 1, 12

                if self.agent_state != self.agent.training:
                    self.agent_state = self.agent.training
                    self.time_change += 1


        else:
            batch, seq, hidden_size = hidden_states.size()
            probs = self.agent_layer(hidden_states) # batch, 12
            if self.time_change >= 10 and False:
                #masks = masks.detach()
                if prune_adapter_keep_top_layer_dynamic:
                    action = gumbel_softmax(probs[:, 12-layer_keep:], temperature = 1, doing_sample = False).view(batch, -1, 1) # batch, 12, 1
                    action = torch.cat([Variable(torch.tensor(torch.zeros(batch, 12-layer_keep, 1), dtype = float, requires_grad = False, device = 'cuda')), action], dim = 1)

                else:
                    action = gumbel_softmax(probs, temperature = 1, doing_sample = False).view(batch, -1, 1) # batch, 12, 1
                #action = action.detach()

                #action = gumbel_softmax(probs, temperature = 1, doing_sample = True).view(batch, -1, 1) # batch, 12, 1
                #action = action.detach()

            else:
                #action = gumbel_softmax(probs, temperature = 1, doing_sample = True).view(batch, -1, 1)
                ### keep_layer
                if prune_adapter_keep_top_layer_dynamic:
                    action = gumbel_softmax(probs[:, 12-layer_keep:], temperature = 1, doing_sample = True).view(batch, -1, 1) ### batch, 12-layer_keep
                    action = torch.cat([Variable(torch.tensor(torch.zeros(batch, 12-layer_keep, 1), dtype = float, requires_grad = False, device = 'cuda')), action], dim = 1)
                else:
                    action = gumbel_softmax(probs, temperature = 1, doing_sample = True).view(batch, -1, 1)


            masks = self.masks_adapter.view(1, len(self.layer), -1).expand(batch, len(self.layer), len(self.layer)) # batch, 12, 1230
            policy = action.contiguous()
            policy = policy.float()
            masks = torch.sum(masks * policy, dim = 1).view(batch, -1, 1, 1).expand(batch, -1, seq, 1) # batch, 1, 12

        


            

            if self.agent_state != self.agent.training:
                self.agent_state = self.agent.training
                self.time_change += 1

            '''
            #probs = self.agent_layer(hidden_states) # batch, 12
            probs = self.agent_layer(hidden_states)
            action = gumbel_softmax(probs, temperature = 1).view(batch, -1, 1) # batch, 12, 1
            masks = self.masks.view(1, len(self.layer), -1).expand(batch, len(self.layer), len(self.layer)) # batch, 12, 12
            policy = action.contiguous()
            policy = policy.float() # batch, 12, 1
            masks = torch.sum(masks * policy, dim = 1).view(batch, -1, 1, 1).expand(batch, -1, seq, 1) # batch, 1, 12
            '''





            '''
            if self.time_change <= 20:
                masks = ((masks - 0.1) > 0)  * (masks - 0.1) + ((masks + 0.1) < 1)  * (masks + 0.1)

            if self.agent_state != self.agent.training:
                self.agent_state = self.agent.training
                self.time_change += 1
            '''




            #### make masks to [0.1, 0.9]
            #masks = ((masks - 0.1) > 0) * 1. * (masks - 0.1) + ((masks + 0.1) < 1) * 1. * (masks + 0.1)
            #print(masks)

            #print(torch.sum(masks[0, :, 0]))
            #else:
            #    probs = self.agent(hidden_states).view(batch, -1, 2) # batch, 12 * 2
            #    _, action = torch.max(probs, dim = 2) # batch, 12, 1
            #    policy = action
            #    masks = policy.float().view(policy.size(0), -1, 1, 1).expand(policy.size(0), -1, seq, 1)



        '''
        ### use self.M to generate the mask
        batch, seq, hidden_size = hidden_states.size()
        if self.agent.training is True:
            action = gumbel_softmax(self.M).view(1, self.M.size(0), -1).expand(batch, self.M.size(0), self.M.size(1)) # batch, 12, 2
            policy = action[:, :, 1].contiguous() # batch, 12, 1
            masks = policy.float().view(policy.size(0), -1, 1, 1).expand(policy.size(0), -1, seq, 1) # batch, 12, seq, 1
            #print(torch.sum(masks[0, :, 0]))
        else:
            _, action = torch.max(self.M, dim = 1)
            policy = action.view(-1).view(1, action.size(0), 1).expand(batch, -1, 1)
            masks = policy.float().view(policy.size(0), -1, 1, 1).expand(policy.size(0), -1, seq, 1) # batch, 12, seq, 1
            #print(torch.sum(masks[0, :, 0]))
        '''


        for i, layer_module in enumerate(self.layer):
            #layer_module.attention.output.layer_id = i #999
            #layer_module.output.layer_id = i
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                #print(masks[:, i].view(masks.size(0), masks.size(2), 1))
                layer_outputs = layer_module(
                    masks[:, i].view(masks.size(0), masks.size(2), 1), #masks[i],
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
         
            hidden_states = layer_outputs[0] # batch, seq_len, hidden

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        #print ('output is here ...')
        #print(len(all_hidden_states))
        if not return_dict:
            if prune_keep_low_layer_dynamic:
                return [tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None), masks, policy]
            else:
                return [tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None), masks]



        ## 222
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )








class BertEncoderDetached(nn.Module):
    """Added by Linlin

       BERT encoder with freezed parameters.
    """
    def __init__(self, config, wrapper_config=None):
        super().__init__()
        self.config = config
        self.wrapper_config = wrapper_config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if self.wrapper_config is not None and 'encoder_finetune_layers' in self.wrapper_config:
            self.finetune_layers = list(map(int, self.wrapper_config['encoder_finetune_layers'].split(',')))
        else:
            self.finetune_layers = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if (self.finetune_layers is None) or (i not in self.finetune_layers):
                hidden_states = hidden_states.detach()

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1].detach(),)

        if (self.finetune_layers is None) or (len(self.layer) not in self.finetune_layers):
            hidden_states = hidden_states.detach()

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""




@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModelOrig(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        print ('\n')

        for n, p in self.embeddings.named_parameters():
            #print (n + ' is freezed ...')
            p.requires_grad = False

        #print ('embedding layer is freezed ...')

        self.encoder = BertEncoder(config)


        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """




        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class BertWithWrapper(BertModelOrig):
    """Added by Linlin

       BERT model wrapper.
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # select which wrapper to use
        bert_wrapper_opt = WRAPPERS
        try:
            select_wrapper = os.environ['USE_WRAPPER']
        except:
            select_wrapper = None

        try:
            wrapper_config_path = os.environ['WRAPPER_CONFIG_PATH']
        except:
            wrapper_config_path = None

        self.wrapper_config = None
        if select_wrapper is None:
            self.wrapper = None
            print('*' * 20)
            print('\n\nnot using wrapper\n\n')
            print('*' * 20)
        elif select_wrapper in bert_wrapper_opt:
            print('*' * 20)
            print('\n\nusing BertWithWrapper: %s\n\n' % select_wrapper)
            print('*' * 20)
            if wrapper_config_path is not None:
                if not os.path.isfile(wrapper_config_path):
                    print('WRAPPER_CONFIG_PATH is not found: %s' % wrapper_config_path)
                    raise FileNotFoundError
                wrapper_config = load_wrapper_config(wrapper_config_path)
                if select_wrapper in wrapper_config:
                    self.wrapper_config = wrapper_config[select_wrapper]
                    print('*' * 20)
                    print('\nloaded wrapper config from %s\n' % wrapper_config_path)
                    print('*' * 20)

            ## implement 1
            #self.encoder = BertEncoderDetached(config, wrapper_config=self.wrapper_config)

            ## implement 2
            '''
            self.encoder = BertEncoder(config)
            for name, param in self.encoder.named_parameters():
                #print (name)
                if 'layer' in name:
                    param.requires_grad = False
                    print (name + ' is freezed')
            '''
            ## implement 3: adaper 
            self.encoder = BertEncoderWithAdapter(config)
            #self.encoder = BertEncoder(config)
            print ("\n")
            print ("\t\t Using Adapter")
            print ("\n")
            for name, param in self.encoder.named_parameters():
                #print (name)
                
                #if 'layer' in name: #and 'adapter_for_' not in name:
                #if 'adapter_PET' in name:
                #    print ('[adapter_PET] is in %s' % (name) )

                if 'layer' in name:
                    if ('adapter_PET' not in name) and ('agent' not in name):
                    #if ('adapter_PET' not in name) and ('agent' not in name) and ('LayerNorm' not in name):
                        param.requires_grad = False
                        #print (name + ' is freezed')


                '''
                if 'layer' in name and 'adapter_PET' not in name: # or 'layer' in name and 'agent' not in name:
                    param.requires_grad = False
                    print (name + ' is freezed')
                if 'layer' in name and 'agent' not in name:
                    params
                '''


            self.wrapper = bert_wrapper_opt[select_wrapper](config, wrapper_config=self.wrapper_config)
        else:
            print('*' * 20)
            print('\n\ninvalid choice of wrapper: %s\n\n' % select_wrapper)
            print('*' * 20)
            raise NotImplementedError


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        '''
        for n, p in self.embeddings.named_parameters():
            if 'LayerNorm.bias' in n:
                print (p)
        '''


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        ##333


        if prune_keep_low_layer_dynamic:

            encoder_outputs, masks, policy = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            output_hidden_states=True,
            return_dict=return_dict,
            )
        else:
            encoder_outputs, masks = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            output_hidden_states=True,
            return_dict=return_dict,
            )

        #print(len(encoder_outputs[1]))
        all_hidden_states = encoder_outputs[1]
        #print(all_hidden_states[1])

        #print(masks)

        # add wrapper here - BEGIN
        if self.wrapper is not None:
            sequence_output = self.wrapper(encoder_outputs[1], attention_mask=attention_mask)
        else:
            sequence_output = encoder_outputs[0]

        if not output_hidden_states:
            encoder_outputs = tuple(encoder_outputs[0])

        # add wrapper here - END

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None


        ##333

        if not return_dict:
            if prune_keep_low_layer_dynamic:
                return [(sequence_output, pooled_output) + encoder_outputs[1:], masks, policy, all_hidden_states]
            else:
                return [(sequence_output, pooled_output) + encoder_outputs[1:], masks, all_hidden_states]                

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )




BertModel = BertWithWrapper




@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning. """, BERT_START_DOCSTRING
)
class BertLMHeadModel(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            n ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> config.is_decoder = True
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config, return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """,
    BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring).  Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', return_dict=True)

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, next_sentence_label=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), next_sentence_label.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_layers = nn.Linear(config.hidden_size, 12 * config.num_labels)


        self.agent_state = True
        self.time_change = 0

        
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if prune_keep_low_layer_dynamic:
            outputs, masks, policy, all_hidden_states= self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        else:
            outputs, masks, all_hidden_states= self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        #print(outputs)


        pooled_output = outputs[1]
        #all_hidden_states = outputs[2] # list(batch, seq, hidden) --> batch, 12, seq, hidden ---> batch, 12, hidden --> batch, 12, 12 * n_class ---> batch, n_class
        # batch, 12, n_class --> 
        pooled_output = self.dropout(pooled_output)

        

        if prune_keep_low_layer_dynamic: 
            ## 555
            if self.time_change <= 10:
                batch, seq, hidden_size = all_hidden_states[0].size()
                pooled_output_layers = self.dropout(self.bert.pooler(all_hidden_states[-1])) # batch, seq, hidden

                logits = self.classifier_layers(pooled_output_layers).view(batch, 12, -1)[:, 11].view(batch, -1)

            elif self.time_change > 10 and self.time_change <= 30:
                batch, seq, hidden_size = all_hidden_states[0].size()
                pooled_output_layers = [self.dropout(self.bert.pooler(h)) for h in all_hidden_states[1:] ] # batch, seq, hidden
                logits = []
                logits_label = []
                for i in range(layer_keep):
                    logits_ = self.classifier_layers(pooled_output_layers[i]).view(batch, 12, -1)[:, i].view(batch, -1) # batch, nclass
                    if i == layer_keep - 1:
                        logits_label_ = F.softmax(logits_/5., dim = 1)  #torch.exp(logits_/5.) / torch.sum(logits_/5., dim = 1).view(-1, 1)
                        logits_.detach()
                    else:
                        logits_label_ = F.softmax(logits_/5., dim = 1)
                    logits.append(logits_)
                    logits_label.append(logits_label_)
                ## do distillation
            else:
                ## freeze adapter and classifier
                for n, p in self.classifier_layers.named_parameters():
                    p.requires_grad = False
                for name, param in self.bert.encoder.named_parameters():
                    if 'adapter_PET' in name:
                        param.requires_grad = False


                logits = self.classifier_layers(pooled_output).view(pooled_output.size(0), 12, -1) # batch, 12, n_class
                #logits = torch.sum(logits * policy.view(pooled_output.size(0), -1, 1), dim = 1).view(pooled_output.size(0), -1).detach() + self.classifier(pooled_output)
                logits = torch.sum(logits * policy.view(pooled_output.size(0), -1, 1), dim = 1).view(pooled_output.size(0), -1)


            '''
            if self.time_change > 120:
                logits = self.classifier_layers(pooled_output).view(pooled_output.size(0), 12, -1) # batch, 12, n_class
                logits = torch.sum(logits * policy.view(pooled_output.size(0), -1, 1), dim = 1).view(pooled_output.size(0), -1).detach() + self.classifier(pooled_output)
            else:
                logits = self.classifier_layers(pooled_output).view(pooled_output.size(0), 12, -1) # batch, 12, n_class
                logits = torch.sum(logits * policy.view(pooled_output.size(0), -1, 1), dim = 1).view(pooled_output.size(0), -1)# batch, 12, n_class
            '''
            if self.agent_state != self.bert.training:
                self.agent_state = self.bert.training
                self.time_change += 1

        else:   
            logits = self.classifier(pooled_output) # batch, 12 * n_class --> batch, 12, n_class   mask: batch, 12 1


        #print('loss is calculated here ...') ##111
        #print(masks)

        ## add norm for the masks

        '''
        # list(batch, seq, 1
        batch, seq, _ = masks[0].size()
        new_masks = [mask[:, 0].view(-1, 1) for mask in masks]
        new_masks = torch.cat(new_masks, dim = 1) # batch, num_layers
        #print(new_masks.size())
        layer_keep = 3 #6 #12
        loss_masks = torch.mean((torch.sum(new_masks, dim = 1) - layer_keep) ** 2)
        #loss_masks = torch.mean(torch.sum(new_masks-, dim = 1))
        #print(loss_masks)
        '''



        masks_for_return = 0

        # masks: batch, 12, seq, 1
        #layer_keep = 6 #9 #12 #3
        if prune_adapter_keep_low_layer_dynamic or prune_adapter_keep_top_layer_dynamic:
            if prune_adapter_keep_low_layer_dynamic:
                if if_agent_token:
                    #loss_masks = torch.mean(torch.sum(masks[:, :layer_keep, :], dim = 2)) * 0 #1e-4
                    #loss_masks = torch.mean(torch.sum(masks[:, :layer_keep, :], dim = 2))
                    loss_masks = torch.mean(torch.mean((torch.sum(masks[:, :layer_keep, :], dim = 2).view(masks.size(0), -1) - 50) ** 2, dim = 1)) * 1e-5# batch, L
                else:
                    loss_masks = torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1)) * 1e-3
            elif prune_adapter_keep_top_layer_dynamic:
                if if_agent_token:
                    #loss_masks = torch.mean(torch.sum(masks[:, 12-layer_keep:, :], dim = 2).view(-1)) * 1e-4
                    loss_masks = torch.mean(torch.mean(torch.sum(masks[:, 12-layer_keep:, :], dim = 2).view(masks.size(0), -1), dim = 1)) * 1e-4 # batch, L 
                    #loss_masks = torch.mean(torch.mean((torch.sum(masks[:, 12-layer_keep:, :], dim = 2).view(masks.size(0), -1) - 50) ** 2, dim = 1)) * 1e-5# batch, L
                else:
                    '''
                    ### loss 1
                    loss_masks = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(masks.size(0), -1) - 7
                    loss_masks = torch.mean((loss_masks > 0.) * loss_masks) 
                    loss_masks *= 0.1 #1e-2 #5e-3
                    '''


                    '''
                    ### loss 2
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    loss_masks = loss_mean - 5
                    loss_masks = (loss_masks > 0) * 1 #- (loss_div < 2) * loss_div * 0.01
                    '''

                    '''
                    ### loss 3
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    loss_masks = torch.mean((loss_sum - 6 > 0) * loss_sum) * 0.1 - (loss_div < 2) * loss_div * 0.1
                    '''

                    '''
                    ### loss 4
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    loss_masks = torch.mean((loss_sum - 6 > 0) * loss_sum) * 0.1 - (loss_div < 2) * loss_div * 0.1 + (loss_div > 4) * loss_div * 0.1
                    '''


                    '''
                    ### loss 5
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    loss_masks = torch.mean((loss_sum - 6 > 0) * loss_sum) * 0.1 - (loss_div < 4) * loss_div * 0.1
                    '''

                    '''
                    ### loss 6
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    loss_masks = torch.mean((loss_sum - 6 > 0) * loss_sum) * 0.1 - (loss_div < 7) * loss_div * 0.1
                    '''

                    '''
                    ### loss 7
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    loss_masks =  (loss_mean - 6 ) ** 2 * 0.1   - (loss_div < 2) * loss_div * 0.1
                    '''


                    '''
                    ### loss 8
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    #loss_masks =  (loss_mean - 6 ) ** 2 * 0.1  - (loss_div < 2) * loss_div * 0.1
                    loss_masks =  (loss_mean - 5.5 ) ** 2 * 0.1  - (loss_div < 2) * loss_div * 0.1
                    #loss_masks =  (loss_mean - 6 ) ** 2 * 1  - (loss_div < 2) * loss_div * 0.1
                    '''

                    ### loss 9
                    loss_sum = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(-1) 
                    loss_mean = torch.mean(loss_sum)
                    loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                    #loss_masks = loss_sum - 5     #loss_mean - 5
                    #loss_masks =  (loss_mean - 6 ) ** 2 * 0.1  - (loss_div < 2) * loss_div * 0.1
                    loss_masks = -((loss_div < 2) * loss_div * 0.1)  + (loss_div > 6) * loss_div * 0.1  #(loss_mean - 5.5 ) ** 2 * 0.1  - (loss_div < 2) * loss_div * 0.1





                    masks_for_return = torch.mean(torch.sum(masks[:, 12-layer_keep:, 0], dim = 1))

                    ## constrain the top layers
                    #loss_masks += torch.mean(torch.sum(masks[:, 6:, 0], dim = 1)) * 10

                    #loss_masks = torch.mean(torch.sum(masks[:, 6:, 0], dim = 1)) * 10

                    



                    if not self.bert.training:
                        print(torch.sum(masks[0, 12-layer_keep:, 0]))
                        print(masks[0, 12-layer_keep:, 0].view(-1))
                        print(masks[0, :, 0].view(-1))
                        print('*******************************')
                        #print(masks.size())



                '''
                if not self.bert.training or True:
                    print(torch.sum(attention_mask[1]))
                    print(attention_mask[1])
                    print(torch.sum(masks[1, 12-layer_keep:13- layer_keep, :].view(-1) * attention_mask[1].view(-1)))
                    print(masks[1, 12-layer_keep:13- layer_keep, :].view(-1))
                    print(torch.sum(masks[1, 10:11, :].view(-1) * attention_mask[1].view(-1)))
                    print (masks[1, 10:11, :].view(-1))
                    print('*******************************************')
                '''
                #print (masks[1, 12-layer_keep:, 50:])
            #loss_masks = torch.mean(torch.sum(masks[:, :, 0], dim = 1)) * 0.01

        elif prune_keep_top_layer_dynamic or prune_keep_low_layer_dynamic:
            if prune_keep_top_layer_dynamic:
                loss_masks = torch.sum(masks[:, 12-layer_keep:, 0], dim = 1).view(masks.size(0), -1) - (layer_keep - 1)
                loss_masks = torch.mean((loss_masks > 0.) * loss_masks) 
                loss_masks *= 5e-3
            elif prune_keep_low_layer_dynamic:

                # loss 1
                #loss_masks = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(masks.size(0), -1) - (layer_keep - 3)
                #loss_masks = torch.mean((loss_masks > 0.) * loss_masks) 
                #loss_masks *= 0.1

                '''
                # loss 2
                loss_masks = torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1).view(masks.size(0), -1)) 
                loss_masks = loss_masks - (layer_keep - 6) #torch.mean((loss_masks > 0.) * loss_masks) 
                loss_masks = (loss_masks > 0) * loss_masks

                loss_div = torch.mean((torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1) - torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1))) ** 2)  # batch

                loss_masks = loss_masks * 0.01 - loss_div * 0.01
                '''

                '''
                ## loss 4
                loss_masks = torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1).view(masks.size(0), -1)) 
                loss_masks = loss_masks - (layer_keep - 6) #torch.mean((loss_masks > 0.) * loss_masks) 
                loss_masks = (loss_masks > 0) * loss_masks
                loss_div = torch.mean((torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1) - torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1))) ** 2)  # batch
                loss_masks = loss_masks * 0.01 - (loss_div < 3) * loss_div * 0.01
                '''

                '''
                ### loss 5
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean( (loss_sum - loss_mean) ** 2 )
                loss_masks = (loss_mean - (layer_keep-6) ) ** 2 * 0.1 - (loss_div < 3) * loss_div ** 2 * 0.01 + (loss_div > 9) * loss_div ** 2 * 0.01  #(loss_div < 3) * loss_div * 0.01
                '''

                '''
                ### loss 6
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean( (loss_sum - loss_mean) ** 2 )
                loss_masks = (loss_mean - (layer_keep-6) ) ** 2 * 0.1  + (loss_div > 1) * loss_div * 0.1  #(loss_div < 3) * loss_div * 0.01
                '''


                '''
                ### loss 7
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean( (loss_sum - loss_mean) ** 2 )
                loss_masks = (loss_mean - (layer_keep-6) ) ** 2 * 0.1  - (loss_div < 0.5) * loss_div  * 0.1  + (loss_div > 1) * loss_div * 0.1  #(loss_div < 3) * loss_div * 0.01
                '''


                '''
                ### loss 8
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean( (loss_sum - loss_mean) ** 2 )
                loss_masks = (loss_mean - (layer_keep-6) ) ** 2 * 0.1  - (loss_div < 3) * loss_div  * 0.1  + (loss_div > 9) * loss_div * 0.1  #(loss_div < 3) * loss_div * 0.01
                '''

                '''
                ### loss 9
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean( (loss_sum - loss_mean) ** 2 )
                #patience = (self.time_change / 6.)
                #loss_masks = (loss_mean - 7 > 0) * loss_mean * 0.1 - (loss_div < 3) * loss_div  * 0.1  #+ (loss_div > 9) * loss_div * 0.1 # patience small to large

                #loss_masks = (loss_mean - 7 > 0) * loss_mean * 0.1 - (loss_mean - 3.5 < 0) * loss_mean * 0.1 - (loss_div < 3) * loss_div  * 0.1  #+ (loss_div > 9) * loss_div * 0.1 # patience small to large

                loss_masks = (loss_mean - 9 > 0) * loss_mean * 0.1 - (loss_div < 3) * loss_div  * 0.1  #+ (loss_div > 9) * loss_div * 0.1 # patience small to large
                '''

                '''
                ### loss 10
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean( (loss_sum - loss_mean) ** 2 )
                #patience = (self.time_change / 6.)
                #loss_masks = (loss_mean - 7 > 0) * loss_mean * 0.1 - (loss_div < 3) * loss_div  * 0.1  #+ (loss_div > 9) * loss_div * 0.1 # patience small to large

                #loss_masks = (loss_mean - 7 > 0) * loss_mean * 0.1 - (loss_mean - 3.5 < 0) * loss_mean * 0.1 - (loss_div < 3) * loss_div  * 0.1  #+ (loss_div > 9) * loss_div * 0.1 # patience small to large

                loss_masks = (loss_mean - 9 > 0) * loss_mean * 0.1 - (loss_div < 5) * loss_div  * 0.1  #+ (loss_div > 9) * loss_div * 0.1 # patience small to large
                '''
            
                '''
                ### loss 11
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                loss_masks = -((loss_div < 2) * loss_div * 0.1) + (loss_div > 6) * loss_div * 0.1
                '''


                ### loss 12
                loss_sum = torch.sum(masks[:, :layer_keep, 0], dim = 1).view(-1)
                loss_mean = torch.mean(loss_sum)
                loss_div = torch.mean((loss_sum - loss_mean) ** 2)
                loss_masks = -((loss_div < 1) * loss_div * 0.1) #+ (loss_div > 6) * loss_div * 0.1



                #print(loss_masks.size())



                #loss_masks *= 0.01

                # loss 3
                #loss_masks = torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1).view(masks.size(0), -1)) 
                #loss_masks *= 1e-3


                #masks_for_return = loss_masks.items()
                masks_for_return = torch.mean(torch.sum(masks[:, :layer_keep, 0], dim = 1))#.items()
                #print(masks_for_return.cpu().numpy())
                #loss_masks *= 0.1#1e-2 #5e-3

                '''
                if not self.bert.training :
                    print(torch.sum(masks[0, :layer_keep, 0]))
                    print(masks[0, :layer_keep, 0].view(-1))
                    print(masks[0, :, 0].view(-1))
                    print('*******************************')
                    #print(masks.size())
                '''


        else:
            if if_agent_token:
                loss_masks = torch.mean(torch.mean(torch.sum(masks[:, :, :], dim = 2).view(masks.size(0), -1), dim = 1)) * 1e-4

            else:
                #loss_masks = torch.mean((torch.sum(masks[:, :, 0], dim = 1) - layer_keep) ** 2)
                loss_masks = torch.sum(masks[:, :, 0], dim = 1).view(masks.size(0), -1) - layer_keep
                loss_masks = torch.mean((loss_masks > 0.) * loss_masks)

                masks_for_return = torch.mean(torch.sum(masks[:, :, 0], dim = 1).view(masks.size(0), -1))

                ## constrain the top layers
                #loss_masks += torch.mean(torch.sum(masks[:, 6:, 0], dim = 1))

                loss_masks *= 5e-2 #1e-2 

                '''
                if not self.bert.training:
                    print(torch.sum(masks[0, 12-layer_keep:, 0]))
                    print(masks[0, 12-layer_keep:, 0].view(-1))
                    print('*******************************')
                    #print(masks.size())
                '''


        if prune_dynamic_succession:
            loss_masks += torch.mean(  torch.sum((masks[:, 1:, 0] - masks[:, :-1, 0]) ** 2, dim = 1))


        

        ### 555
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                if isinstance(logits, list):
                    loss = 0
                    _logits = 0
                    '''
                    for logits_ in logits:
                        loss += loss_fct(logits_.view(-1), labels.view(-1))
                        _logits += logits_ / layer_keep
                    '''
                    ## add the distillation loss
                    for i in range(len(logits)):
                        if i < len(logits) - 1:
                            loss += loss_fct(logits[i].view(-1), labels.view(-1)) + 1 * nn.KLDivLoss(reduction = 'batchmean')(torch.log(logits_label[i].view(-1)),  logits_label[-1].view(-1))  #0.01 * loss_fct(logits[i].view(-1), logits[-1].view(-1))
                            _logits += logits[i] / (len(logits)-1)
                        #else:
                        #    loss += loss_fct(logits[i].view(-1), labels.view(-1))
                        #_logits += logits[i] / len(logits)

                    loss += loss_masks
                    logits = _logits

                else:
                    loss = loss_fct(logits.view(-1), labels.view(-1)) + loss_masks
            else:
                loss_fct = CrossEntropyLoss()
                if isinstance(logits, list):
                    loss = 0
                    _logits = 0
                    '''
                    for logits_ in logits:
                        loss += loss_fct(logits_.view(-1, self.num_labels), labels.view(-1))
                        _logits += logits_ / layer_keep
                    '''
                    for i in range(len(logits)):
                        if i < len(logits) - 1:
                            loss += loss_fct(logits[i].view(-1, self.num_labels), labels.view(-1)) + 100 * nn.KLDivLoss(reduction = 'batchmean')(torch.log(logits_label[i].view(-1, self.num_labels)),  logits_label[-1].view(-1, self.num_labels))  #torch.mean((logits[i] - logits[-1]) ** 2) * 0.1  #100 * nn.KLDivLoss(reduction = 'batchmean')(torch.log(logits_label[i].view(-1, self.num_labels)),  logits_label[-1].view(-1, self.num_labels)) 
                            _logits += logits[i] / (len(logits)-1)
                            #if self.time_change > 10:
                            #    print(nn.KLDivLoss(size_average = True)(torch.log(logits[i].view(-1, self.num_labels)),  logits[-1].view(-1, self.num_labels)).item())
                        #else:
                        #    loss += loss_fct(logits[i].view(-1, self.num_labels), labels.view(-1))
                        

                    loss += loss_masks
                    logits = logits[6] #_logits
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + loss_masks


        ## haiye: return here
        if not return_dict:
            output = (logits,) + outputs[2:]
            ### haiye: 
            #if not self.bert.training:
            #    print ('return dict ... ')

            return ((loss,) + output) if loss is not None else output, masks_for_return

        ### haiye:
        #if not self.bert.training:
        #    print ('return sequence ... ')
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
