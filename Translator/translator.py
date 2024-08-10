import torch
from torch import nn
import math

class MLP(nn.Module):
    def __init__(self, d_input, d_ff, d_output):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_output, bias=True),
        )

    def forward(self, input):
        output = self.mlp(input)
        return output

class Multi_Head_CrossAttention(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_heads, dropout):
        super(Multi_Head_CrossAttention, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.d_q = self.d_k = d_qk
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.d_model, self.d_q * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)
        self.ln = nn.LayerNorm(self.d_model)

    # 计算q*k*v
    def qkv(self, Q, K, V):
        att_scores = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d_q)
        m_r = torch.ones_like(att_scores) * self.dropout
        att_scores = att_scores + torch.bernoulli(m_r) * -1e12
        att_scores = nn.Softmax(dim=-1)(att_scores)
        att_output = torch.matmul(att_scores, V)
        return att_output

    def forward(self, input_Q, input_K, input_V):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # Q: [batch_size, n_heads, L, d_q]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, L, d_k]
        V = self.W_Q(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, L, d_v]
        att_output = self.qkv(Q, K, V)
        att_output = att_output.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        att_output= self.fc(att_output)  # output: [batch_size, L, d_model]
        return att_output

class TLencoder(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device):
        super(TLencoder, self).__init__()

        self.d_model = d_model
        self.device = device
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.context_attn = Multi_Head_CrossAttention(
            d_model=d_model, 
            d_qk=d_qk, 
            d_v=d_v, 
            n_heads=n_heads, 
            dropout=dropout
        )
        self.global_attn = Multi_Head_CrossAttention(
            d_model=d_model, 
            d_qk=d_qk, 
            d_v=d_v, 
            n_heads=n_heads, 
            dropout=dropout
        )
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.mlp = MLP(d_input=d_model, d_ff=d_ff, d_output=d_model)

    def block_forward(self, x, hn, input):    # (bs, d_model)
        x = x.unsqueeze(dim=1)

        res = x
        x = self.conv1(x)
        x_context = self.context_attn(hn, x, x)
        x = self.ln1(x_context + res)
     
        res = x
        x = self.conv2(x)
        h = self.global_attn(x, input, input)
        x = self.ln2(h + res)

        res = x
        y = self.mlp(x)
        y = self.ln3(y + res)

        return y, h
    
    def forward(self, input, h0=None):
        bs = input.size(0)

        hn = h0 if h0 is not None else input[:, 0, :].unsqueeze(dim=1)
        x = torch.cat((h0, input), dim=1) if h0 is not None else input
        output = torch.ones(bs, 1, self.d_model).to(self.device)

        for i in range(x.size(1)):
            y, hn = self.block_forward(x[:, i, :], hn, input)
            output = torch.cat((output, y), dim=1)

        output = output[:, 2:, :] if h0 is not None else output[:, 1:, :]
        
        return output, hn
    
class Translator_Encoder_Block(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device):
        super(Translator_Encoder_Block, self).__init__()

        self.TLencoder = TLencoder(d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device)

    def forward(self, inputs):

        x, hn = inputs

        x, hn = self.TLencoder(x, hn)

        output = x

        return (output, hn)
    
class TLdecoder(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device):
        super(TLdecoder, self).__init__()

        self.d_model = d_model
        self.device = device
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)
        self.context_attn = Multi_Head_CrossAttention(
            d_model=d_model, 
            d_qk=d_qk, 
            d_v=d_v, 
            n_heads=n_heads, 
            dropout=dropout
        )
        self.global_attn = Multi_Head_CrossAttention(
            d_model=d_model, 
            d_qk=d_qk, 
            d_v=d_v, 
            n_heads=n_heads, 
            dropout=dropout
        )
        self.cross_attn = Multi_Head_CrossAttention(
            d_model=d_model, 
            d_qk=d_qk, 
            d_v=d_v, 
            n_heads=n_heads, 
            dropout=dropout
        )
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.mlp = MLP(d_input=d_model, d_ff=d_ff, d_output=d_model)

    def block_forward(self, x, encoder_x, hn, input):    # (bs, d_model)
        x = x.unsqueeze(dim=1)
        encoder_x = encoder_x.unsqueeze(dim=1)

        res = x
        x = self.conv1(x)
        x_context = self.context_attn(hn, x, x)
        x = self.ln1(x_context + res)
     
        res = x
        x = self.conv2(x)
        x = self.global_attn(x, input, input)
        x = self.ln2(x + res)

        res = x
        h = self.cross_attn(encoder_x, x, x)
        x = self.ln3(h + res)

        res = x
        y = self.mlp(x)
        y = self.ln4(y + res)

        return y, h
    
    def forward(self, input, encoder_output, h0=None):
        bs = input.size(0) 

        hn = h0
        x = torch.cat((h0, input), dim=1)
        encoder_x = torch.cat((h0, encoder_output), dim=1)
        output = torch.ones(bs, 1, self.d_model).to(self.device)

        for i in range(x.size(1)):
            y, hn = self.block_forward(x[:, i, :], encoder_x[:, i, :], hn, input)
            output = torch.cat((output, y), dim=1)

        output = output[:, 2:, :]
        
        return output, hn
    
class Translator_Decoder_Block(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device):
        super(Translator_Decoder_Block, self).__init__()

        self.TLdecoder = TLdecoder(d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device)

    def forward(self, inputs):

        x, encoder_output, hn = inputs

        x, hn = self.TLdecoder(x, encoder_output, hn)

        output = x

        return (output, encoder_output, hn)

class Translator(nn.Module):
    def __init__(self, num_layers, d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device):
        super(Translator, self).__init__()
        
        self.ln0 = nn.LayerNorm(d_model)

        translator_Encoder_layer_list = []
        for layer in range(num_layers):
            translator_Encoder_layer_list.append(Translator_Encoder_Block(d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device))
        self.translator_Encoder_layer = nn.Sequential(*translator_Encoder_layer_list)

        translator_Decoder_layer_list = []
        for layer in range(2):
            translator_Decoder_layer_list.append(Translator_Decoder_Block(d_model, d_qk, d_v, n_heads, d_ff, kernel_size, dropout, device))
        self.translator_Decoder_layer = nn.Sequential(*translator_Decoder_layer_list)

    def forward(self, input):

        x = self.ln0(input)
        
        encoder_output, hn = self.translator_Encoder_layer((x, None))
        decoder_output, _, last_hidden_state = self.translator_Decoder_layer((x, encoder_output, hn))
        
        output = decoder_output  
    
        return decoder_output, last_hidden_state