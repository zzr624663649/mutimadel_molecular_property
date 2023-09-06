import build_data
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from bert_model import collate_pretrain_data, EarlyStopping, run_a_pretrain_epoch, set_random_seed, K_BERT_WCL
from grover.util.utils import build_optimizer, build_lr_scheduler, makedirs, load_checkpoint, get_loss_func, \
    save_checkpoint, build_model
from grover.model.models import GROVEREmbedding
from logging import Logger
import argparse
import os
import time
from coca_pytorch import CoCa
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_random_seed()


apex_support = False
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=50,help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.00001,help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0,help='weight decay (default: 0)')
parser.add_argument('--num_layer', type=int, default=5,help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300,help='embedding dimensions (default: 300)')
parser.add_argument('--dropout', type=float, default=0,help='dropout ratio (default: 0.2)')
parser.add_argument('--graph_pooling', type=str, default="mean",help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
parser.add_argument("--hidden_size", type=int, default=768, help='hidden size')
parser.add_argument('--backbone', type=str, default="dualtrans",help='backbone')
parser.add_argument('--dist_coff', type=float, default=0.1, help='distcoff')
parser.add_argument('--embedding_output_type', type=str, default="both",help='embedding_output_type')
parser.add_argument('--activation', type=str, default="PReLU",help='activation')
parser.add_argument('--input_layer', type=str, default="fc",help='backbone')
parser.add_argument('--num_mt_block', type=int, default=1,help='num_mt_block')
parser.add_argument('--num_attn_head', type=int, default=4,help='num_attn_head')
parser.add_argument('--bias', type=bool, default=False,help='bias')
parser.add_argument('--cuda', type=bool, default=True,help='cuda')
parser.add_argument('--depth', type=int, default=6,help='depth')
parser.add_argument('--dense', type=bool, default=False,help='dense')
parser.add_argument('--undirected', type=bool, default=False, help='undirected')
parser.add_argument('--bond_drop_rate', type=int, default=0, help='undirected')
parser.add_argument('--features_only', action='store_true', default=False,
                    help='Use only the additional features in an FFN, no graph network')
parser.add_argument('--no_cache', type=bool, default=True, help='undirected')


args_g = parser.parse_args()

# define parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['batch_size'] = 32
args['num_epochs'] = 50
args['d_model'] = 768
args['n_layers'] = 6
args['vocab_size'] = 47
args['maxlen'] = 201
args['d_k'] = 64
args['d_v'] = 64
args['d_ff'] = 768*4
args['n_heads'] = 12
args['global_labels_dim'] = 154
args['atom_labels_dim'] = 15
args['lr'] = 0.00001
args['task_name'] = 'k_bert_wcl'
args['pretrain_data_path'] = '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/CHEMBL_maccs'
pretrain_set_1 = build_data.load_data_for_pretrain_1(
    pretrain_data_path=args['pretrain_data_path'])
print("Pretrain data generation is complete !")

pretrain_loader = DataLoader(dataset=pretrain_set_1,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             collate_fn=collate_pretrain_data)

global_pos_weight = torch.tensor([884.17, 70.71, 43.32, 118.73, 428.67, 829.0, 192.84, 67.89, 533.86, 18.46, 707.55, 160.14, 23.19, 26.33, 13.38, 12.45, 44.91, 173.58, 40.14, 67.25, 171.12, 8.84, 8.36, 43.63, 5.87, 10.2, 3.06, 161.72, 101.75, 20.01, 4.35, 12.62, 331.79, 31.17, 23.19, 5.91, 53.58, 15.73, 10.75, 6.84, 3.92, 6.52, 6.33, 6.74, 24.7, 2.67, 6.64, 5.4, 6.71, 6.51, 1.35, 24.07, 5.2, 0.74, 4.78, 6.1, 62.43, 6.1, 12.57, 9.44, 3.33, 5.71, 4.67, 0.98, 8.2, 1.28, 9.13, 1.1, 1.03, 2.46, 2.95, 0.74, 6.24, 0.96, 1.72, 2.25, 2.16, 2.87, 1.8, 1.62, 0.76, 1.78, 1.74, 1.08, 0.65, 0.97, 0.71, 5.08, 0.75, 0.85, 3.3, 4.79, 1.72, 0.78, 1.46, 1.8, 2.97, 2.18, 0.61, 0.61, 1.83, 1.19, 4.68, 3.08, 2.83, 0.51, 0.77, 6.31, 0.47, 0.29, 0.58, 2.76, 1.48, 0.25, 1.33, 0.69, 1.03, 0.97, 3.27, 1.31, 1.22, 0.85, 1.75, 1.02, 1.13, 0.16, 1.02, 2.2, 1.72, 2.9, 0.26, 0.69, 0.6, 0.23, 0.76, 0.73, 0.47, 1.13, 0.48, 0.53, 0.72, 0.38, 0.35, 0.48, 0.12, 0.52, 0.15, 0.28, 0.36, 0.08, 0.06, 0.03, 0.07, 0.01])
atom_pos_weight = torch.tensor([4.81, 1.0, 2.23, 53.49, 211.94, 0.49, 2.1, 1.13, 1.22, 1.93, 5.74, 15.42, 70.09, 61.47, 23.2])
loss_criterion_global = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=global_pos_weight.to('cuda'))
loss_criterion_atom = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=atom_pos_weight.to('cuda'))
model_bert = K_BERT_WCL(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
                   maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
                   global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'])

# model_grover=GROVEREmbedding(args_g).to('cuda')
model_grover = load_checkpoint('/home/ubuntu/zzr/smiles_bertZ/grover_large.pt', current_args=args_g, logger=None)
model_coca = CoCa(
    dim = 768,                     # model dimension
    img_encoder = None,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
    num_tokens = 15,              # number of text tokens
    sub_graph = 85,
    unimodal_depth = 6,            # depth of the unimodal transformer
    multimodal_depth = 6,          # depth of the multimodal transformer
    dim_head = 64,                 # dimension per attention head
    heads = 8,                     # number of attention heads
    caption_loss_weight = 1.,      # weight on the autoregressive caption loss
    contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
).to('cuda')
# model_bert.load_state_dict(torch.load('/home/ubuntu/zzr/smiles_bertZ/pretrain_model/model_bert_nocon0.pth'), strict=False)
# model_coca.load_state_dict(torch.load('/home/ubuntu/zzr/smiles_bertZ/pretrain_model/model_coca_nocon0.pth'), strict=False)
for name, parameter in model_grover.named_parameters():
    parameter.requires_grad = False
# for name, parameter in model_bert.named_parameters():
#     parameter.requires_grad = False
# for name, parameter in model_coca.named_parameters():
#     parameter.requires_grad = False
optimizer_grover = torch.optim.Adam(
    model_grover.parameters(), args['lr'],
    weight_decay=eval('1e-5')
)
optimizer_grover = Adam(model_grover.parameters(), lr=args['lr'])
optimizer_bert = Adam(model_bert.parameters(), lr=args['lr'])
optimizer_coca = Adam(model_coca.parameters(), lr=args['lr'])
stopper = EarlyStopping(task_name=args['task_name'])
model_bert.to(args['device'])
model_grover.to(args['device'])
model_coca.to(args['device'])

for epoch in range(args['num_epochs']):
    start = time.time()
    # Train
    run_a_pretrain_epoch(args,args_g, epoch, model_bert, model_grover, model_coca, pretrain_loader, optimizer_bert=optimizer_bert, optimizer_coca = optimizer_coca, optimizer_grover = optimizer_grover, loss_criterion_atom = loss_criterion_atom)
    # Validation and early stop

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("An epoch time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))












