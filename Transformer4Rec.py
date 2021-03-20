import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

d_model = 32  # Embedding Size
d_ff = 64  # FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
sub_embed = 32

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


def get_attn_pad_mask(seq_q, seq_k, pad_idx):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class LSTM(nn.Module):
    def __init__(self, num_courses):
        super(LSTM, self).__init__()
        self.num_course = num_courses
        self.ipt_course = nn.Embedding(num_courses, d_model)
        # self.tgt_course = nn.Embedding(num_courses, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True, num_layers=2, dropout=0.1)

        for param in self.parameters():
            nn.init.normal_(param, 0, 0.01)

    def forward(self, ipt_seq, tgt_seq, seq_subject, tgt_subject):
        enc_ipt = self.ipt_course(ipt_seq)
        dec_out = self.ipt_course(tgt_seq)

        out, (h_n, c_n) = self.lstm(enc_ipt)
        enc_outputs = h_n[-1]  # [batch, d_model]
        # print(enc_outputs.shape)

        out = torch.sum(enc_outputs.unsqueeze(1) * dec_out, dim=2, keepdim=False)  # [batch, tgt_len]
        return out



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1
                                                                           ,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, num_courses):
        super(Encoder, self).__init__()
        self.num_courses = num_courses
        self.src_emb = nn.Embedding(num_courses, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        for param in self.parameters():
            nn.init.normal_(param, 0, 0.01)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs,
                                               self.num_courses)  # [batch_size, src_len, src_len]
        enc_self_attn_subsequence_mask = get_attn_subsequence_mask(enc_inputs).cuda()  # [batch_size, src_len, src_len]
        enc_self_attn_mask = torch.gt((enc_self_attn_mask + enc_self_attn_subsequence_mask),
                                      0).cuda()  # [batch_size, src_len, src_len]

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, num_courses, course_emb):
        super(Decoder, self).__init__()
        self.num_courses = num_courses
        # self.tgt_emb = nn.Embedding(num_courses, d_model)
        self.tgt_emb = course_emb
        self.pos_emb = PositionalEncoding(d_model)
        # self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        for param in self.parameters():
            nn.init.normal_(param, 0, 0.01)

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        # dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda() # [batch_size, tgt_len, d_model]

        return dec_outputs


class Transformer4Rec(nn.Module):
    def __init__(self, num_courses):
        super(Transformer4Rec, self).__init__()
        self.encoder = Encoder(num_courses).cuda()
        self.course_emb = self.encoder.src_emb
        self.decoder = Decoder(num_courses, self.course_emb).cuda()
        # self.projection = nn.Linear(d_model, num_courses, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs, seq_subject, tgt_subject):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs = self.decoder(dec_inputs)  # [batch * tgt_len * d_model]

        enc_rps = torch.mean(enc_outputs, dim=1, keepdim=True)  # [batch, 1, d_model]
        out = torch.sum(enc_rps * dec_outputs, dim=2, keepdim=False)  # [batch, tgt_len]

        return out


class FM(nn.Module):
    def __init__(self, num_course, course_emb=None, num_subjects=21):
        super(FM, self).__init__()
        self.sub_emb = nn.Embedding(num_subjects, d_model)
        if not course_emb:
            self.course_emb = nn.Embedding(num_course, d_model)
        else:
            self.course_emb = course_emb
        self.ffn = nn.Sequential(nn.Linear(sub_embed * 3, d_ff, bias=False),
                                 nn.LeakyReLU(),
                                 nn.Linear(d_ff, d_model, bias=False)
                                 )
        self.ffn2 = nn.Sequential(nn.Linear(d_model * 2, d_ff, bias=False),
                                  nn.LeakyReLU(),
                                  nn.Linear(d_ff, d_model, bias=False),
                                  )
        for param in self.parameters():
            nn.init.normal_(param, 0, 1e-2)


    def forward(self, input_seq, tgt_seq, seq_subjects, tgt_subjects):
        # seq_subjects [batch, L, 2]

        seq_subjcet1, seq_subject2 = torch.split(seq_subjects, [1, 1], dim=2,)  # get each subject [batch L]
        seq_subjcet1, seq_subject2 = seq_subjcet1.squeeze(2), seq_subject2.squeeze(2)
        seq_subjcet1_emb, seq_subject2_emb = self.sub_emb(seq_subjcet1), self.sub_emb(seq_subject2)     # [batch, L, d]
        seq_subject_feature = torch.cat([seq_subjcet1_emb * seq_subject2_emb, seq_subjcet1_emb, seq_subject2_emb], dim=2)  # [batch, L, 3d]
        seq_subject_feature = self.ffn(seq_subject_feature)  # [batch, L, d]

        tgt_subjcet1, tgt_subject2 = torch.split(tgt_subjects, [1, 1], dim=2,)  # get each subject [batch K]
        tgt_subjcet1, tgt_subject2 = tgt_subjcet1.squeeze(2), tgt_subject2.squeeze(2)
        tgt_subjcet1_emb, tgt_subject2_emb = self.sub_emb(tgt_subjcet1), self.sub_emb(tgt_subject2)     # [batch, K, d]
        tgt_subject_feature = torch.cat([tgt_subjcet1_emb * tgt_subjcet1_emb, tgt_subjcet1_emb, tgt_subjcet1_emb], dim=2)  # [batch, K, 3d]
        # tgt_subject_feature = self.ffn(tgt_subject_feature)  # [batch, L, d]

        seq_course_emb, tgt_course_emb = self.course_emb(input_seq), self.course_emb(tgt_seq)
        seq_course_subject = torch.cat([seq_course_emb, seq_subject_feature], dim=2)
        tgt_course_subject = tgt_course_emb + tgt_subjcet1_emb + tgt_subject2_emb

        seq_course_subject = self.ffn2(seq_course_subject)  # [batch L d]

        # seq_subjects_out = torch.mean(seq_course_subject, dim=1, keepdim=True)
        seq_subjects_out = torch.max(seq_course_subject, dim=1, keepdim=True)[0]
        # tgt_subject_feature = tgt_subjcet1_emb + tgt_subject2_emb

        FM_out = torch.sum(seq_subjects_out * tgt_course_subject, dim=2)

        return FM_out

class FM_Transoformer(nn.Module):
    def __init__(self, num_course):
        super(FM_Transoformer, self).__init__()
        self.Transformer = Transformer4Rec(num_course)
        self.FM = FM(num_course, self.Transformer.course_emb)
        self.alpha = 0.7

    def forward(self, enc_input, dec_input, seq_subjects, tgt_subjects):
        transformer_out = self.Transformer(enc_input, dec_input, seq_subjects, tgt_subjects)
        FM_out = self.FM(enc_input, dec_input, seq_subjects, tgt_subjects)
        out = self.alpha * transformer_out + (1 - self.alpha) * FM_out
        return out




