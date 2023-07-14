import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .gcn import GCN, GCNII
from .word_embedding import load_word_embeddings
import scipy.sparse as sp
import random
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Annoemb(nn.Module):
    def __init__(self, dset, args):
        super(Annoemb, self).__init__()
        self.args = args
        self.dset = dset

        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.pairs = dset.pairs

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)

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
        
        self.img_projection = MLP(dset.feat_dim, 512, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)    # args.emb_dim(non-codebook) 512(codebook)
        self.img2attr = MLP(512, 512, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)     # args.emb_dim(non-codebook) 512(codebook)
        self.img2obj = MLP(512, 512, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)     # args.emb_dim(non-codebook) 512(codebook)
        self.anno_classifier = MLP(dset.feat_dim, 512, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)

        ############## word to shared semantic space #################
        # word embedding
        input_dim = args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

        # init with word embeddings
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        all_words = list(self.dset.attrs) + list(self.dset.objs)
        self.displacement = len(all_words)

        self.proj_word = MLP(input_dim * 2, 512, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)
        self.proj_attr = nn.Linear(input_dim, 512) # args.emb_dim(non-codebook) 512(codebook)
        self.proj_obj = nn.Linear(input_dim, 512)
        self.softmax = nn.Softmax(dim=1)

    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def compose_ones(self, objs, attrs, all=False, is_sota=False):
        objs, attrs = self.obj_embedder(objs), self.attr_embedder(attrs)
        if all:
            concat_emb = []
            for i in range(objs.size(0)):
                for j in range(attrs.size(0)):
                    concat_emb.append(torch.cat((objs[i], attrs[j])))
            concat_emb = torch.stack(concat_emb)
        else:
            concat_emb = torch.cat((objs,attrs), 1)
        if is_sota:
            return concat_emb
        output = self.proj_word(concat_emb)
        return output
    
    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        output_attrs = self.proj_attr(attrs)
        output_objs = self.proj_obj(objs)

        return output_attrs, output_objs
    
    def codebook_train(self, x):
        img = x[0]
        pos_obj, img_neg = x[-2], x[-1]

        cls_fea = self.anno_classifier(img)  

        loss_con_pos = 0
        loss_con_neg = 0
        for i in range(len(img_neg[0])):
            neg_sample = img_neg[:, i, :]
            obj_neg_feats = self.anno_classifier(neg_sample)

            index_1 = random.choice(range(pos_obj.shape[1]))
            pos_sample = pos_obj[:, index_1, :]
            pos_feats = self.anno_classifier(pos_sample)

            loss_con_pos += F.triplet_margin_loss(cls_fea, pos_feats, cls_fea, margin=0)
            loss_con_neg += F.triplet_margin_loss(cls_fea, cls_fea, obj_neg_feats, margin=10)
            if math.isnan(loss_con_neg):
                print(torch.min(neg_sample))

        loss_con_pos /= pos_obj.shape[1] 
        loss_con_neg /= pos_obj.shape[1] 
        loss_con_obj = loss_con_pos + loss_con_neg

        cls_loss = torch.tensor(0)

        return cls_loss, loss_con_obj, None, loss_con_pos, loss_con_neg

    def eval_classifier_only(self, x):
        img = x[0]

        attr_embs, objs_embs = self.compose(self.uniq_attrs, self.uniq_objs)

        if self.args.target == 'attr':
            attr_cls_fea = self.anno_classifier(img)   #attr_classifier

            score = torch.matmul(attr_cls_fea, torch.transpose(attr_embs,0,1))

            scores = {}
            for itr, attr in enumerate(self.dset.attrs):
                    scores[attr] = score[:,self.dset.attr2idx[attr]]

        elif self.args.target == 'obj':
            obj_cls_fea = self.anno_classifier(img)   #obj_classifier

            score = torch.matmul(obj_cls_fea, torch.transpose(objs_embs,0,1))
            
            scores = {}
            for itr, obj in enumerate(self.dset.objs):
                    scores[obj] = score[:,self.dset.obj2idx[obj]]

        return None, scores

    def make_code_book(self, store_path):
        with torch.no_grad():
            self.dset.code_book = torch.zeros(len(self.dset.objs), len(self.dset.attrs), 512)

            for i_key, key in enumerate(self.dset.fea_dict.keys()):
                tmp_attr, tmp_obj = key.split('_')
                for i_fea, feature in enumerate(self.dset.fea_dict[key]):
                    self.dset.fea_dict[key][i_fea] = self.anno_classifier(feature.to(device))
                avg_fea = torch.mean(self.dset.fea_dict[key], dim=0)
                self.dset.code_book[self.dset.obj2idx[tmp_obj],self.dset.attr2idx[tmp_attr]] = avg_fea
        
        code_book = {'code_book': self.dset.code_book}
        self.dset.train_triplet_loss = False
        torch.save(code_book, store_path)

    def train_forward(self, x):
        img, attrs, objs = x[0], x[1], x[2]

        attr_embs, obj_embs = self.compose(self.uniq_attrs, self.uniq_objs)

        img = self.img_projection(img)

        obj_fea = self.img2obj(img)
        attr_fea = self.img2attr(img)

        obj_pred = torch.matmul(obj_fea, torch.transpose(obj_embs, 0, 1))
        attr_pred = torch.matmul(attr_fea, torch.transpose(attr_embs, 0, 1))

        obj_weights = self.softmax(obj_pred).detach()
        attr_weights = self.softmax(attr_pred).detach()

        if self.args.use_code_book:

            temp_obj_code_book = torch.permute(self.dset.code_book, (2, 1, 0)).to(device)
            object_adapted_code_book = torch.matmul(temp_obj_code_book, obj_weights.permute(1, 0))

            temp_attr_code_book = torch.permute(self.dset.code_book, (2, 0, 1)).to(device)
            attr_adapted_code_book = torch.matmul(temp_attr_code_book, attr_weights.permute(1, 0))

            attr_embs = object_adapted_code_book + attr_embs.permute(1, 0).unsqueeze(2)
            attr_embs = torch.permute(attr_embs, (2, 1, 0))   # object 정보가 첨가된 attribute embs (이미지 정보 -> word 정보 주입)

            obj_embs = attr_adapted_code_book + obj_embs.permute(1, 0).unsqueeze(2)
            obj_embs = torch.permute(obj_embs, (2, 1, 0))   # attribute 정보가 첨가된 object embs   (이미지 정보 -> word 정보 주입)

        # if self.args.use_code_book:
        attr_pred = torch.zeros(attr_fea.size(0), attr_embs.size(1)).to(device)
        for i, (feature, code_book_feature) in enumerate(zip(attr_fea, attr_embs)):
            attr_pred[i] = torch.matmul(feature.unsqueeze(0), torch.transpose(code_book_feature,0,1))

        obj_pred = torch.zeros(obj_fea.size(0), obj_embs.size(1)).to(device)
        for i, (feature, code_book_feature) in enumerate(zip(obj_fea, obj_embs)):
            obj_pred[i] = torch.matmul(feature.unsqueeze(0), torch.transpose(code_book_feature,0,1))

        loss = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            
        return  loss, None

    def val_forward(self, x):
        img = x[0]

        attr_embs, obj_embs = self.compose(self.uniq_attrs, self.uniq_objs)

        img = self.img_projection(img)
        
        obj_fea = self.img2obj(img)
        attr_fea = self.img2attr(img)

        attr_pred = torch.matmul(attr_fea, torch.transpose(attr_embs, 0, 1))
        obj_pred = torch.matmul(obj_fea, torch.transpose(obj_embs, 0, 1))

        scores = {}

        for itr, pair in enumerate(self.dset.full_pairs):
            scores[pair] = (attr_pred[:,self.dset.attr2idx[pair[0]]] + obj_pred[:,self.dset.obj2idx[pair[1]]]).cpu()

        return None, scores

    def forward(self, x, train_cls_only):
        if train_cls_only:
            if self.training :
                CE_loss, margin_loss, pred, loss_con_pos, loss_con_neg = self.codebook_train(x)
            else:
                with torch.no_grad():
                    CE_loss, pred = self.eval_classifier_only(x)
                    margin_loss=torch.tensor(0)
                    loss_con_pos=torch.tensor(0)
                    loss_con_neg=torch.tensor(0)
        else:
            if self.training:
                CE_loss, pred = self.train_forward(x)
                margin_loss=torch.tensor(0)
                loss_con_pos=torch.tensor(0)
                loss_con_neg=torch.tensor(0)
            else:
                with torch.no_grad():
                    CE_loss, pred = self.val_forward(x)
                    margin_loss=torch.tensor(0)
                    loss_con_pos=torch.tensor(0)
                    loss_con_neg=torch.tensor(0)

        return CE_loss, margin_loss, pred, loss_con_pos, loss_con_neg
