import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import configparser
import math
from .activations import ACT2FN
from .modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
#from .modeling_bert import BertLayer






def load_wrapper_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


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

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            print(attention_scores.size())
            print(attention_mask.size())
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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


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
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
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

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
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

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class HorizontalPooler(nn.Module):
    """Added by Linlin
 
       mean horizontal pooler.
    """
    def __init__(self, config, wrapper_config):
        super().__init__()
 
    def forward(self, hidden_states, attention_mask):
        hidden_states_wo_mask = ()
        for layer_hidden_states in hidden_states:
            hidden_states_wo_mask += (layer_hidden_states * attention_mask.unsqueeze(2),)
        hidden_states_wo_mask = torch.stack(hidden_states_wo_mask)
        hidden_states_sum = torch.sum(hidden_states_wo_mask, dim=2)
        attention_sum = torch.sum(attention_mask, dim=1)
        pooled_mean = hidden_states_sum / torch.unsqueeze(attention_sum, 1)
        pooled_output = torch.cat((torch.unsqueeze(pooled_mean, 2), hidden_states[:,:,1:]), 2)
        return pooled_output


class HorizontalAttentionPooler(nn.Module):
    """
    Added by Ruidan

    Attentioin horizontal pooling  
    """
    def __init__(self, config, wrapper_config):
        super().__init__()
        num_layers = config.num_hidden_layers
        n_dim = config.hidden_size
        stdv = 1./math.sqrt(n_dim)
        self.activation = nn.Tanh()
        #use different attention vectors for different layers
        self.att_vectors = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_dim, 1).uniform_(-stdv, stdv), 
                requires_grad=True) for _ in range(num_layers)])

        self.h_pooler = HorizontalPooler(config, wrapper_config)


    def forward(self, hidden_states, attention_mask):
        hidden_states_att = ()
        for i, features in enumerate(hidden_states):
            weights = torch.bmm(features, # (batch_size, seq_len, hidden_size)
                                self.att_vectors[i]  # (hidden_size, 1)
                                .unsqueeze(0)  # (1, hidden_size, 1)
                                .repeat(features.shape[0], 1, 1) # (batch_size, hidden_size, 1)
                                )

            weights = self.activation(weights.squeeze())
            weights = torch.exp(weights) * attention_mask
            weights = weights / torch.sum(weights, dim=1, keepdim=True)
            features = features * weights.unsqueeze(2)
            hidden_states_att += (features,)

        hidden_states_att = torch.stack(hidden_states_att)        
        pooled_output = self.h_pooler(hidden_states_att, attention_mask)
        return pooled_output




class HorizontalCNNPooler(nn.Module):
    """
    Added by Ruidan

    1dCNN with horizontal mean pooler 
    """
    def __init__(self, config, wrapper_config):
        super().__init__()
        num_layers = config.num_hidden_layers
        in_dim = config.hidden_size
        out_dim = config.hidden_size
        # default kernel size is 3
        kernel_size = 3
        if wrapper_config is not None:
            if "cnn_kernel_size" in wrapper_config:
                kernel_size = int(wrapper_config['cnn_kernel_size'])

        #use different CNNs for different layers
        self.cnns = nn.ModuleList([nn.Conv1d(in_dim, out_dim, kernel_size, 
                        padding=int(kernel_size/2)) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.h_pooler = HorizontalPooler(config, wrapper_config)


    def forward(self, hidden_states, attention_mask):
        hidden_states_cnn = ()
        for i, features in enumerate(hidden_states):
            # convert shape (batch_size, seq_len, hidden_size) to (batch_size, hidden_size, seq_len)
            features = torch.transpose(features, 1, 2)
            features = self.cnns[i](features)
            features = self.activation(features)
            # convert shape back to (batch_size, seq_len, hidden_size)
            features = torch.transpose(features, 1, 2)
            hidden_states_cnn += (features,)

        hidden_states_cnn = torch.stack(hidden_states_cnn)        
        pooled_output = self.h_pooler(hidden_states_cnn, attention_mask)
        return pooled_output




class BertWrapper(nn.Module):
    """Added by Linlin

       wrapper API
    """
    def __init__(self, config, wrapper_config=None):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.h_pooler = HorizontalPooler(config, wrapper_config)
        if wrapper_config is not None:
            if "num_wrapper_layers" in wrapper_config:
                self.num_layers = int(wrapper_config['num_wrapper_layers'])

    def forward(self, hidden_states, attention_mask):
        raise NotImplementedError


class BertWrapperLastLayer(BertWrapper):
    """Added by Linlin

       do nothing, just return last layer output. for ablation study.
    """
    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)

    def forward(self, hidden_states, attention_mask):
        return hidden_states[-1]


class BertWrapperMean(BertWrapper):
    """Added by Linlin

       mean vertical pooling of cls vectors
    """
    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[-self.num_layers:])
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        return output


class BertWrapperMeanHori(BertWrapper):
    """
    Added by Ruidan

    mean vertical pooling after layer-wise horizontal pooling
    """

    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[-self.num_layers:])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        return output


class BertWrapperMeanHoriNoActivation(BertWrapper):
    """
    Added by Linlin

    mean vertical pooling after layer-wise horizontal pooling
    """

    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[-self.num_layers:])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        return output

class ProductWeight(nn.Module):
    """Added by Linlin

       product weight.
    """
    def __init__(self, init_val):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([float(init_val)]), requires_grad=True)

    def forward(self, hidden_states):
        return self.weight * hidden_states


class BertWrapperWeight(BertWrapper):
    """Added by Linlin

       weighted vertical pooling.
    """
    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)
        self.layer_weight = nn.ModuleList([ProductWeight(1.0 / self.num_layers) for _ in range(self.num_layers)])
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[-self.num_layers:])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.zeros_like(hidden_states_top_layers[0])
        for i in range(len(hidden_states_top_layers)):
            output += self.layer_weight[i](hidden_states_top_layers[i])
        #output = self.activation(output)
        return output



class BertWrapperCnnMEAN(BertWrapper):
    """
    Added by Ruidan

    use CNN for horizontal pooling and mean for vertical pooling
    """

    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)
        self.activation = nn.Tanh()
        self.h_pooler = HorizontalCNNPooler(config, wrapper_config)

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[-self.num_layers:])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        #print(output.size())
        return output



class BertWrapperAttMean(BertWrapper):
    """
    Added by Ruidan

    use simple attention for horizontal pooling and mean for vertical pooling
    """

    def __init__(self, config, wrapper_config=None):
        super().__init__(config, wrapper_config)
        self.activation = nn.Tanh()
        self.h_pooler = HorizontalAttentionPooler(config, wrapper_config)

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[-self.num_layers:])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        #print (output.size())
        return output

class BertWrapperOcean(BertWrapper):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__(config, wrapper_config)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.act = torch.tanh

    def init_weights(self):
        initrange = 0.01
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        for feature in hidden_states:
            #print(feature.size()) # 32, 128, 768
            #print(mask.size())
            layer_features.append(torch.mean(feature * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, hidden_size))

        operated_layer_features = [layer_features[0]]

        for i in range(1, len(layer_features)):
            f1 = self.act(self.fc1(operated_layer_features[i-1]))
            f2 = torch.cat((layer_features[i], f1), dim = 1)
            f_out = self.act(self.fc2(f2))
            operated_layer_features.append(f_out)

        # dimension problem
        return operated_layer_features[-1].view(batch_size, 1, hidden_size).expand(batch_size, seq_len, hidden_size)




class BertWrapperRNNOcean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()
        self.RNN = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False)

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        for feature in hidden_states:
            layer_features.append(torch.mean(feature * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(1, batch_size, hidden_size))

        layer_features = torch.cat(layer_features, dim = 0) #torch.stack(layer_features, dim = 0) # layer_num, batch, hidden_size
        #print(layer_features.size())

        output, hn = self.RNN(layer_features) # hn: 1, batch, hidden_size
        hn = hn.view(batch_size, -1)
        #hn = torch.mean(output, dim = 0).view(batch_size, -1)
        #print(output.size())
        #print(hn.size()) # 1, 64, 768

        return hn.view(batch_size, 1, hidden_size).expand(batch_size, seq_len, hidden_size)


        #return hn #operated_layer_features[-1]#.view(batch_size, 1, hidden_size).expand(batch_size, seq_len, hidden_size



class BertWrapperHireRNNOcean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        pre_state = [hidden_states[-6].permute(1, 0, 2)]
        for i, feature in enumerate(hidden_states[-5:]):
            feature = feature.permute(1, 0, 2)
            if i == 0:
                fi, _ = self.RNN1(feature + pre_state[i])
            elif i == 1:
                fi, _ = self.RNN2(feature + pre_state[i])
            elif i == 2:
                fi, _ = self.RNN3(feature + pre_state[i])
            elif i == 3:
                fi, _ = self.RNN4(feature + pre_state[i])
            elif i == 4:
                fi, _ = self.RNN5(feature + pre_state[i])
            fi = fi.permute(1, 0, 2)
            fi = self.activation(fi)
            fi = self.LayerNorm(fi)
            pre_state.append(fi.permute(1, 0, 2))

        return torch.mean(pre_state[-1], dim = 0).view(batch_size, 1, -1)



class BertWrapperResiduOcean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.dense_down = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.dense_up = nn.Linear(int(config.hidden_size/2), config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        self.LayerNorm = nn.LayerNorm([768], eps=config.layer_norm_eps)


    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        for feature in hidden_states:
            layer_features.append(torch.mean(feature * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, hidden_size))


        pre_state = [layer_features[0]]

        for i, feature in enumerate(layer_features):
            if i != 0:
                pre_feature = pre_state[i-1]
                pre_feature = self.activation(self.dense_down(pre_feature))
                pre_feature = self.activation(self.dense_up(pre_feature))

                cur_feature = self.activation(pre_feature + feature)
                cur_feature = self.LayerNorm(cur_feature)

                pre_state.append(cur_feature)

        return pre_state[-1].view(batch_size, 1, hidden_size)


        #layer_features = torch.cat(layer_features, dim = 0) #torch.stack(layer_features, dim = 0) # layer_num, batch, hidden_size
        #print(layer_features.size())

        #output, hn = self.RNN(layer_features) # hn: 1, batch, hidden_size
        #hn = hn.view(batch_size, -1)
        #hn = torch.mean(output, dim = 0).view(batch_size, -1)
        #print(output.size())
        #print(hn.size()) # 1, 64, 768

        #return hn.view(batch_size, 1, hidden_size).expand(batch_size, seq_len, hidden_size)


        #return hn #operated_layer_features[-1]#.view(batch_size, 1, hidden_size).expand(batch_size, seq_len, hidden_size



class BertWrapperRNN_F_TN_LNOcean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        pre_state = [hidden_states[-6].permute(1, 0, 2)]
        for i, feature in enumerate(hidden_states[-5:]):
            feature = feature.permute(1, 0, 2)
            if i == 0:
                RNN_fi, _ = self.RNN1(feature + pre_state[i])
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature + pre_state[i])
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature + pre_state[i])
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature + pre_state[i])
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature + pre_state[i])
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = self.activation(RNN_fi) # 

            fi = self.activation(feature.permute(1, 0, 2)) # batch, seq_len, hidden

            TN_fi = self.activation(self.dense(fi))


            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi)

            #

            pre_state.append(LN_fi.permute(1, 0, 2))

        return torch.mean(pre_state[-1], dim = 0).view(batch_size, 1, -1)



class BertWrapperRNN_F_TN_LNOcean_for_CoLA(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 2
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = pre_state[i] #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)



class BertWrapperRNN_F_TN_LN_V2Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        pre_state = [hidden_states[-6].permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-5:]):
            feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature + pre_state[i])
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature + pre_state[i])
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature + pre_state[i])
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature + pre_state[i])
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature + pre_state[i])
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = self.activation(RNN_fi) # 

            fi = self.activation(feature.permute(1, 0, 2)) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i].permute(1, 0, 2))

            TN_fi = self.activation(self.dense(fi + pre_state[i].permute(1, 0, 2)))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi.permute(1, 0, 2))

        return torch.mean(pre_state[-1], dim = 0).view(batch_size, 1, -1)



class BertWrapperRNN_F_TN_LN_V3Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = pre_state[i] #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)



class BertWrapperRNN_F_TN_LN_V4Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        return pre_state[-1]
        #return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)


class BertWrapperRNN_F_TN_LN_V5Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))


            #LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)
            LN_fi = self.LayerNorm( TN_fi + RNN_fi )
            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        return pre_state[-1]
        #return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)



class BertWrapperRNN_F_TN_LN_V6Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)





class BertWrapperRNN_F_TN_LN_V7Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            #RNN_fi, _ = self.RNN1(self.LayerNorm(feature + pre_state[i]).permute(1, 0, 2))

            if i == 0:
                RNN_fi, _ = self.RNN1(self.LayerNorm(feature + pre_state[i]).permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(self.LayerNorm(feature + pre_state[i]).permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(self.LayerNorm(feature + pre_state[i]).permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(self.LayerNorm(feature + pre_state[i]).permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(self.LayerNorm(feature + pre_state[i]).permute(1, 0, 2))

            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(self.LayerNorm(fi + pre_state[i])))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)


class BertWrapperRNN_F_TN_LN_V8Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1((feature * attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2((feature* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3((feature* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4((feature* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5((feature* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi* attention_mask.view(batch_size, seq_len, 1) + pre_state[i]* attention_mask.view(batch_size, seq_len, 1)))



            LN_fi = self.LayerNorm((fi + TN_fi + RNN_fi + hi_1)* attention_mask.view(batch_size, seq_len, 1))

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1]* attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)




class BertWrapperRNN_F_TN_LN_V9Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []
        for feature in hidden_states:
            layer_features.append(feature * attention_mask.view(batch_size, seq_len, 1))


        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [layer_features[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(layer_features[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + (pre_state[i]* attention_mask.view(batch_size, seq_len, 1)).permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense((fi + pre_state[i])* attention_mask.view(batch_size, seq_len, 1)))



            LN_fi = self.LayerNorm((fi + TN_fi + RNN_fi + hi_1)* attention_mask.view(batch_size, seq_len, 1))

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        out_feature = 0
        for feature in pre_state:
            out_feature += torch.mean(feature* attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1) / len(pre_state)
        return out_feature
        #return pre_state[-1]
        #return torch.mean(pre_state[-1], dim = 1).view(batch_size, 1, -1)


class BertWrapperRNN_F_TN_LN_V10Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)

class BertWrapperRNN_F_TN_LN_V11Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)


class BertWrapperRNN_F_TN_LN_V12Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 5
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        out_feature = 0
        for feature in pre_state:
            out_feature += torch.mean(feature* attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1) / len(pre_state)

        return out_feature

        #return pre_state[-1]
        #return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)


class BertWrapperRNN_F_TN_LN_V13Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        
        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 2
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)




class BertWrapperRNN_F_TN_LN_V14Ocean(nn.Module):
    """
    Added by Hai Ye
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN6 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN7 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN8 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)


        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []

        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 8
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 5:
                RNN_fi, _ = self.RNN6(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 6:
                RNN_fi, _ = self.RNN7(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 7:
                RNN_fi, _ = self.RNN8(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)



class BertWrapperRNN_F_TN_LN_V15Ocean(nn.Module):
    """
    Added by Hai Ye
    add transformer layers in the last N layers
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.transformer = BertLayer(config)
        self.last_N = 5
        self.transformers = nn.ModuleList([BertLayer(config) for _ in range(self.last_N)])



        '''
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN6 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN7 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN8 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        '''

        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []
        last_N_hidden_states = []

        pre_state = [hidden_states[-self.last_N-1]]

        for i, layer_module in enumerate(self.transformers):
            layer_outputs = layer_module(
                    pre_state[i],
                    None, #attention_mask.view(batch_size, seq_len, 1),
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    output_attentions=False)
            pre_state.append(layer_outputs[0])



        #for feature, layer_module in zip(hidden_states[-self.last_N:], self.transformers):
        #    layer_outputs = layer_module(
        #            feature,
        #            None, #attention_mask.view(batch_size, seq_len, 1),
        #            head_mask=None,
        #            encoder_hidden_states=None,
        #            encoder_attention_mask=None,
        #            output_attentions=False)
        #    last_N_hidden_states.append(layer_outputs[0])

        return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)


        '''
        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = 8
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, feature in enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            if i == 0:
                RNN_fi, _ = self.RNN1(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 1:
                RNN_fi, _ = self.RNN2(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 2:
                RNN_fi, _ = self.RNN3(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 3:
                RNN_fi, _ = self.RNN4(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 4:
                RNN_fi, _ = self.RNN5(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 5:
                RNN_fi, _ = self.RNN6(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 6:
                RNN_fi, _ = self.RNN7(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            elif i == 7:
                RNN_fi, _ = self.RNN8(feature.permute(1, 0, 2) + pre_state[i].permute(1, 0, 2))
            RNN_fi = RNN_fi.permute(1, 0, 2)
            RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))
        '''
        #return pre_state[-1]
        #return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)


class BertWrapperRNN_F_TN_LN_V16Ocean(nn.Module):
    """
    Added by Hai Ye
    add transformer layers in the last N layers
    """
    def __init__(self, config, wrapper_config = None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm([128, 768], eps=config.layer_norm_eps)
        self.activation = nn.Tanh()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.transformer = BertLayer(config)
        self.last_N = 5 #2
        self.transformers = nn.ModuleList([BertLayer(config) for _ in range(self.last_N)])

        '''
        self.RNN = []
        self.RNN1 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN2 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN3 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN4 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN5 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN6 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN7 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        self.RNN8 = nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = True)
        '''

        #self.RNN = [self.RNN1, self.RNN2, self.RNN3, self.RNN4, self.RNN5]


        #for i in range(5):
        #    self.RNN.append(nn.GRU(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = 1, batch_first = False, bidirectional = False))

    def init_weights(self):
        initrange = 0.01


    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: batch_size, seq_len
        batch_size, seq_len, hidden_size = hidden_states[0].size()
        layer_features = []
        last_N_hidden_states = []


        # input: seq_len, batch_size, hidden_size 
        # output: seq_len, batch, num_directions * hidden_size 
        kk = self.last_N
        pre_state = [hidden_states[-(kk+1)]]#.permute(1, 0, 2)] # seq first
        for i, (feature, layer_module) in enumerate(zip(hidden_states[-kk:], self.transformers)): #enumerate(hidden_states[-kk:]):
            # feature: batch, seq, hidden_size
            #feature = feature.permute(1, 0, 2) # seq first
            layer_outputs = layer_module(
                    feature + pre_state[i],
                    None, #attention_mask.view(batch_size, seq_len, 1),
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    output_attentions=False)
            RNN_fi = layer_outputs[0]
            #RNN_fi = RNN_fi.permute(1, 0, 2)
            #RNN_fi = RNN_fi[:, :, :hidden_size] + RNN_fi[:, :, hidden_size:]
            RNN_fi = self.activation(RNN_fi) # 

            fi = feature #feature.permute(1, 0, 2) # batch, seq_len, hidden
            hi_1 = self.activation(pre_state[i]) #.permute(1, 0, 2)

            TN_fi = self.activation(self.dense(fi + pre_state[i]))



            LN_fi = self.LayerNorm(fi + TN_fi + RNN_fi + hi_1)

            #

            pre_state.append(LN_fi)#.permute(1, 0, 2))

        #return pre_state[-1]
        return torch.mean(pre_state[-1] * attention_mask.view(batch_size, seq_len, 1), dim = 1).view(batch_size, 1, -1)


WRAPPERS = {'mean': BertWrapperMean,
            'mean_hori': BertWrapperMeanHori,
            'mean_hori_na': BertWrapperMeanHoriNoActivation,
            'weight': BertWrapperWeight,
            'last_layer': BertWrapperLastLayer,
            'cnn_mean': BertWrapperCnnMEAN,
            'att_mean': BertWrapperAttMean,
            'hai_ye': BertWrapperOcean,
            'RNNOcean': BertWrapperRNNOcean,
            'HireRNNOcean': BertWrapperHireRNNOcean,
            'ResiduOcean': BertWrapperResiduOcean,
            'RNN_F_TN_LNOcean': BertWrapperRNN_F_TN_LNOcean,
            'RNN_F_TN_LN_V2Ocean': BertWrapperRNN_F_TN_LN_V2Ocean,
            'RNN_F_TN_LNOcean_for_CoLA': BertWrapperRNN_F_TN_LNOcean_for_CoLA,
            'RNN_F_TN_LN_V3Ocean': BertWrapperRNN_F_TN_LN_V3Ocean,
            'RNN_F_TN_LN_V4Ocean': BertWrapperRNN_F_TN_LN_V4Ocean,
            'RNN_F_TN_LN_V5Ocean': BertWrapperRNN_F_TN_LN_V5Ocean,
            'RNN_F_TN_LN_V6Ocean': BertWrapperRNN_F_TN_LN_V6Ocean,
            'RNN_F_TN_LN_V7Ocean': BertWrapperRNN_F_TN_LN_V7Ocean,
            'RNN_F_TN_LN_V8Ocean': BertWrapperRNN_F_TN_LN_V8Ocean,
            'RNN_F_TN_LN_V9Ocean': BertWrapperRNN_F_TN_LN_V9Ocean,
            'RNN_F_TN_LN_V10Ocean': BertWrapperRNN_F_TN_LN_V10Ocean,
            'RNN_F_TN_LN_V11Ocean': BertWrapperRNN_F_TN_LN_V11Ocean,
            'RNN_F_TN_LN_V12Ocean': BertWrapperRNN_F_TN_LN_V12Ocean,
            'RNN_F_TN_LN_V13Ocean': BertWrapperRNN_F_TN_LN_V13Ocean,
            'RNN_F_TN_LN_V14Ocean': BertWrapperRNN_F_TN_LN_V14Ocean,
            'RNN_F_TN_LN_V15Ocean': BertWrapperRNN_F_TN_LN_V15Ocean,
            'RNN_F_TN_LN_V16Ocean': BertWrapperRNN_F_TN_LN_V16Ocean
           }
