import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .gcn import GCN, GCNII
from .word_embedding import load_word_embeddings
import scipy.sparse as sp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Classifier(nn.Module):
    def __init__(self, dset, args):
        super(Classifier, self).__init__()
        self.args = args
        self.dset = dset

        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.pairs = dset.pairs

        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        if args.nlayers and args.target=='attr':
            self.image_embedder = MLP(dset.feat_dim, 115, num_layers= args.nlayers, dropout = self.args.dropout, # attr class nums
                norm = self.args.norm, layers = layers, relu = True)
        elif args.nlayers and args.target=='obj':
            self.image_embedder = MLP(dset.feat_dim, 245, num_layers= args.nlayers, dropout = self.args.dropout, # obj class nums
                norm = self.args.norm, layers = layers, relu = True)
        elif args.nlayers :
            self.image_embedder = MLP(dset.feat_dim, self.args.emb_dim, num_layers= args.nlayers, dropout = self.args.dropout, # attr class nums
                norm = self.args.norm, layers = layers, relu = True)
        

        all_words = list(self.dset.attrs) + list(self.dset.objs)
        self.displacement = len(all_words)

        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}

        if args.graph_init is not None:
            path = args.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings'].to(device)
            adj = graph['adj']
            self.embeddings = embeddings
        else:
            embeddings = self.init_embeddings(all_words).to(device)
            adj = self.adj_from_pairs()
            self.embeddings = embeddings

        hidden_layers = self.args.gr_emb
        if args.gcn_type == 'gcn':
            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda = 0.5, alpha = 0.1, variant = False)



    def init_embeddings(self, all_words):

        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj]+self.num_attrs]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds

        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)

        return full_embeddings


    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def adj_from_pairs(self):

        def edges_from_pairs(pairs):
            weight_dict = {'data':[],'row':[],'col':[]}


            for i in range(self.displacement):
                self.update_dict(weight_dict,i,i,1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs

                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                node_id = idx + self.displacement
                self.update_dict(weight_dict,node_id,node_id,1.)

                self.update_dict(weight_dict, node_id, attr_idx, 1.)
                self.update_dict(weight_dict, node_id, obj_idx, 1.)


                self.update_dict(weight_dict, attr_idx, node_id, 1.)
                self.update_dict(weight_dict, obj_idx, node_id, 1.)

            return weight_dict

        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs)+self.displacement, len(self.pairs)+self.displacement))

        return adj



    def train_forward_normal(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        pair_pred = self.image_embedder(img)
        if self.args.target == 'attr':
            loss = F.cross_entropy(pair_pred, attrs)
        elif self.args.target == 'obj':
            loss = F.cross_entropy(pair_pred, objs)
        else:
            loss = 0
        return  loss, None

    def val_forward_dotpr(self, x):
        img = x[0]

        score = self.image_embedder(img)

        scores = {}
        
        if self.args.target == 'attr':
            for itr, attr in enumerate(self.dset.attrs):
                scores[attr] = score[:,self.dset.attr2idx[attr]]
        elif self.args.target == 'obj':
            for itr, obj in enumerate(self.dset.objs):
                scores[obj] = score[:,self.dset.obj2idx[obj]]
        else:
            loss = 0

        return None, scores

    def val_forward_distance_fast(self, x):
        img = x[0]

        img_feats = (self.image_embedder(img))
        current_embeddings = self.gcn(self.embeddings)
        pair_embeds = current_embeddings[self.num_attrs+self.num_objs:,:]

        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:,None,:].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None,:,:].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds)**2
        score = diff.sum(2) * -1

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def forward(self, x, _):
        if self.training:
            loss, pred = self.train_forward(x)
            margin_loss=torch.tensor(0)
            loss_con_pos=torch.tensor(0)
            loss_con_neg=torch.tensor(0)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
                margin_loss=torch.tensor(0)
                loss_con_pos=torch.tensor(0)
                loss_con_neg=torch.tensor(0)
        return loss,margin_loss, pred, loss_con_pos, loss_con_neg
