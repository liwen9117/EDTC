import torch
import torch.nn as nn

import torch
from torch import nn
import numpy as np
import math

# 添加位置信息
class Positional_Encoding(nn.Module):
    def __init__(self, d_model, len, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.dorpout = nn.Dropout(p=dropout)
        pos_table = np.zeros((len, d_model))
        for i in range(len):
            for pos in range(d_model):
                if i % 2 == 0:
                    pos_table[i][pos] = np.sin(pos / np.power(10000, 2 * i / d_model))
                else:
                    pos_table[i][pos] = np.cos(pos / np.power(10000, 2 * i / d_model))
        self.pos_table = torch.FloatTensor(pos_table)

    def forward(self, input):
        inputs = torch.unsqueeze(self.pos_table, 0).to(self.device) + input.to(self.device)
        inputs = self.dorpout(input)
        return inputs

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
        return att_output, att_scores

    def forward(self, input_Q, input_K, input_V):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # Q: [batch_size, n_heads, L, d_q]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, L, d_k]
        V = self.W_Q(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, L, d_v]
        att_output, att_scores = self.qkv(Q, K, V)
        att_output = att_output.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        att_output= self.fc(att_output)  # output: [batch_size, L, d_model]
        return att_output

class Feed_Forward_Net(nn.Module):
    def __init__(self, d_model, d_ff):
        super(Feed_Forward_Net, self).__init__()
        self.d_model = d_model
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ln3 = nn.LayerNorm(self.d_model)
        self.ln4 = nn.LayerNorm(self.d_model)

    def forward(self, x, y, z, sum):    # inputs: [batch_size, seq_len, d_model]
        residual_x = x
        residual_y = y
        residual_z = z
        residual_sum = sum

        x = self.shared_fc(x)
        y = self.shared_fc(y)
        z = self.shared_fc(z)
        sum = self.shared_fc(sum)

        output_x = self.ln1(x + residual_x)
        output_y = self.ln2(y + residual_y)
        output_z = self.ln3(z + residual_z)
        output_sum = self.ln4(sum + residual_sum)

        return output_x, output_y, output_z, output_sum  # [batch_size, seq_len, d_model]

class Conv_Block(nn.Module):
    def __init__(self, num_input):
        super(Conv_Block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_input, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GELU()
        )
    
    def forward(self, *inputs):
        conv_input = torch.cat([input.unsqueeze(1) for input in inputs], dim=1)
        x = self.conv_block(conv_input)
        x = x.squeeze(1)

        return x

class Detail_Conv_Layer(nn.Module):
    def __init__(self):
        super(Detail_Conv_Layer, self).__init__()
        self.detail_conv_block = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.detail_conv_block(x)
        x = x.squeeze(1)

        return x

class Input_Layer(nn.Module):
    def __init__(self, sq_len, d_model, d_qk, d_v, n_heads, d_ff, dropout, device):
        super(Input_Layer, self).__init__()
        self.d_model = d_model
        self.pos_emb = Positional_Encoding(d_model, sq_len, dropout, device)
        self.atten1 = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.atten2 = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.atten3 = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.atten4 = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.atten5 = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.atten6 = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.fead_forward = Feed_Forward_Net(d_model, d_ff) 
        self.conv_block = Conv_Block(num_input=3)  

    def forward(self, input1, input2, input3):
        x1 = self.pos_emb(input1)
        x2 = self.pos_emb(input2)
        x3 = self.pos_emb(input3)
        
        y12 = self.atten1(x2, x1, x1)
        y13 = self.atten2(x3, x1, x1)

        y21 = self.atten3(x1, x2, x2)
        y23 = self.atten4(x3, x2, x2)

        y31 = self.atten5(x1, x3, x3)
        y32 = self.atten6(x2, x3, x3)
        
        y1 = (y12 + y13)/2
        y2 = (y21 + y23)/2
        y3 = (y31 + y32)/2

        y_sum = self.conv_block(y1, y2, y3)

        y1, y2, y3, y_sum = self.fead_forward(y1, y2, y3, y_sum)
        
        return y1, y2, y3, y_sum

class Fusion_Layer(nn.Module):
    def __init__(self, sq_len, d_model, d_qk, d_v, n_heads, d_ff, dropout):
        super(Fusion_Layer, self).__init__()
        self.d_model = d_model
        self.sl_atten = Multi_Head_CrossAttention(d_model, d_qk, d_v, n_heads, dropout)
        self.d_model_atten = Multi_Head_CrossAttention(sq_len, d_qk, d_v, n_heads, dropout)
        self.detail_conv_block1 = Detail_Conv_Layer()
        self.detail_conv_block2 = Detail_Conv_Layer()
        self.detail_conv_block3 = Detail_Conv_Layer()
        self.fead_forward = Feed_Forward_Net(d_model, d_ff) 
        self.conv_block = Conv_Block(num_input=3)
        self.fc1 = nn.Linear(self.d_model, 128, bias=False)
        self.fc2 = nn.Linear(128, self.d_model, bias=False)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ln3 = nn.LayerNorm(self.d_model)

    def global_forward(self, x1, x2, x3):

        x1_ = self.fc1(x1).transpose(1, 2)
        x2_ = self.fc1(x2).transpose(1, 2)
        x3_ = self.fc1(x3).transpose(1, 2)

        y1_ = self.d_model_atten(x1_, x1_, x1_)
        y2_ = self.d_model_atten(x2_, x2_, x2_)
        y3_ = self.d_model_atten(x3_, x3_, x3_)

        y1_ = (self.fc2(y1_.transpose(1, 2)) + x1)/2
        y2_ = (self.fc2(y2_.transpose(1, 2)) + x2)/2
        y3_ = (self.fc2(y3_.transpose(1, 2)) + x3)/2

        y1_global = self.sl_atten(y1_, y1_, y1_)
        y2_global = self.sl_atten(y2_, y2_, y2_)
        y3_global = self.sl_atten(y3_, y3_, y3_)

        return (y1_global+y1_)/2, (y2_global+y2_)/2, (y3_global+y3_)/2
    
    def detail_forward(self, x1, x2, x3):
        
        y1_detail = self.detail_conv_block1(x1)
        y2_detail = self.detail_conv_block2(x2)
        y3_detail = self.detail_conv_block3(x3)

        return y1_detail, y2_detail, y3_detail

    def forward(self, x):    
        x1, x2, x3, sum = x

        x1 = self.ln1(x1)
        x2 = self.ln2(x2)
        x3 = self.ln3(x3)

        y1_global, y2_global, y3_global = self.global_forward(x1, x2, x3)
        y1_detail, y2_detail, y3_detail = self.detail_forward(x1, x2, x3)

        y1 = (y1_detail + y1_global)/2
        y2 = (y2_detail + y2_global)/2
        y3 = (y3_detail + y3_global)/2

        y_sum = self.conv_block(y1, y2, y3) 

        output1, output2, output3, y_sum = self.fead_forward(y1, y2, y3, y_sum)

        return output1, output2, output3, y_sum

class Fuser(nn.Module):
    def __init__(self, sq_len, d_model, d_qk, d_v, n_heads, d_ff, n_encoder, dropout, device):
        super(Fuser, self).__init__()
        self.n_encoder = n_encoder
        self.input_layer = Input_Layer(sq_len, d_model, d_qk, d_v, n_heads, d_ff, dropout, device)
        fusion_layer_list = []
        for layer in range(self.n_encoder):
            fusion_layer_list.append(Fusion_Layer(sq_len, d_model, d_qk, d_v, n_heads, d_ff, dropout))
        self.fusion_layer = nn.Sequential(*fusion_layer_list)    
        
    def forward(self, audio_feature1, audio_feature2, audio_feature3):
        x1, x2, x3, sum = self.input_layer(audio_feature1, audio_feature2, audio_feature3)
        x1, x2, x3, sum = self.fusion_layer((x1, x2, x3, sum))

        return sum

if __name__ == "__main__":
    # input: [batch_size, seq_len, d_model]
    # output: [batch_size, seq_len, d_model]
    audio_feature1 = torch.rand(10, 32, 768)
    audio_feature2 = torch.rand(10, 32, 768)
    audio_feature3 = torch.rand(10, 32, 768)
    model = Fuser(
        sq_len=32,
        d_model=768, 
        d_qk=768, 
        d_v=768, 
        n_heads=4, 
        d_ff=1024, 
        n_encoder=3, 
        dropout=0.1
    )
    
    y = model(audio_feature1, audio_feature2, audio_feature2)
    print(y.shape)
