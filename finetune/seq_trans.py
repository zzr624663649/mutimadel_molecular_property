import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch import nn
from grover.data import mol2graph
from coca_pytorch import CoCa
import os


def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to('cuda')
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm.cuda()(output + residual)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        super(MultiHeadAttention, self).__init__()
        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layernorm(output + residual)

class K_BERT_WCL(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim, atom_label_dim,
                 use_atom=True):
        super(K_BERT_WCL, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.use_atom = use_atom
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])

        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.2),
            nn.Tanh(),
        )
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_global = output[:, 0]

        # h_atom = output[:, 1:]

        h_atom = self.fc(output)
        # h_atom_emb = h_atom.reshape([len(output) * (self.maxlen - 1), self.d_model])
        # logits_atom = self.classifier_atom(h_atom_emb)
        h_global = self.fc(h_global)
        # logits_global = self.classifier_global(h_global)
        return h_global, h_atom

def collate_pretrain_data(data):
    tokens_idx, global_label_list, atom_labels_list, atom_mask_list, smile_list = map(list, zip(*data))
    tokens_idx = torch.tensor(tokens_idx)
    global_label = torch.tensor(global_label_list)
    atom_labels = torch.tensor(atom_labels_list)
    atom_mask = torch.tensor(atom_mask_list)
    return tokens_idx, global_label, atom_labels, atom_mask, smile_list

class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth',
                 pretrain_layer=6, mode='higher', patience=10, task_name="None"):
        assert mode in ['higher', 'lower']
        self.pretrain_layer = pretrain_layer
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = '../model/{}_early_stop.pth'.format(task_name)
        self.pretrain_save_filename = '../model/pretrain_{}_epoch_'.format(task_name)
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = '../model/{}'.format(pretrained_model)

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def pretrain_step(self, epoch, model):
        print('Pretrain epoch {} is finished and the model is saved'.format(epoch))
        self.pretrain_save_checkpoint(epoch, model)

    def pretrain_save_checkpoint(self, epoch, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.pretrain_save_filename + str(epoch) + '.pth')
        # print(self.filename)

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])





def run_a_pretrain_epoch(args, args_g, epoch, model_bert, model_grover, model_coca, data_loader,
                         optimizer_bert, optimizer_coca, optimizer_grover):
    model_bert.train()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        token_idx, global_labels, atom_labels, atom_mask, smiles = batch_data
        optimizer_coca.zero_grad()
        optimizer_bert.zero_grad()
        optimizer_grover.zero_grad()
        # model_grover.requires_grad_(False)
        # model_bert.requires_grad_(False)
        #GROVER
        batchgraph = mol2graph(smiles, shared_dict=[], args=args_g).get_components()
        node_rep = model_grover(batchgraph)
        graph_atom = node_rep
        _, _, _, _, _, a_scope, _, _ = batchgraph
        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_vecs = []
        for _, (a_start, a_size) in enumerate(a_scope):
            cur_hiddens = node_rep.narrow(0, a_start, a_size)
            cur_hiddens = cur_hiddens.sum(dim=0) / a_size
            mol_vecs.append(cur_hiddens)
        graph_global = torch.stack(mol_vecs, dim=0)

        #grover
        token_idx = token_idx.long().to(args['device'])
        global_labels = global_labels.float().to(args['device'])
        atom_labels = atom_labels[:, 1:].float().to(args['device'])
        atom_mask = atom_mask[:, 1:].float().to(args['device'])
        # atom_labels = atom_labels.reshape([len(token_idx)*(args['maxlen']-1), args['atom_labels_dim']])
        # atom_mask = atom_mask.reshape(len(token_idx)*(args['maxlen']-1), 1)
        logits_global, logits_atom, text_global, text_atom = model_bert(token_idx)
        # loss = (loss_criterion_global(logits_global, global_labels).float()).mean() \
        #        + (loss_criterion_atom(logits_atom, atom_labels)*(atom_mask != 0).float()).mean()
        token_idx = token_idx[:, 1:]
        loss = model_coca(graph_atom, graph_global, text_global, text_atom, token_idx, return_loss=True)
        loss.backward()
        optimizer_coca.step()
        optimizer_bert.step()
        optimizer_grover.step()

        total_loss = total_loss + loss*len(token_idx)
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss))
        del token_idx, global_labels, atom_labels, atom_mask, loss, logits_global, logits_atom
        torch.cuda.empty_cache()
    print('epoch {:d}/{:d}, pre-train loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], total_loss))
    torch.save(model_coca.state_dict(), os.path.join('/home/ubuntu/zzr/smiles_bertZ/pretrain_model', 'model_coca{}.pth'.format(str(epoch))))
    torch.save(model_bert.state_dict(), os.path.join('/home/ubuntu/zzr/smiles_bertZ/pretrain_model', 'model_bert{}.pth'.format(str(epoch))))
    return total_loss

def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)