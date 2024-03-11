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
from .compcos import compute_cosine_similarity


def adj_to_edges(adj):
    # Adj sparse matrix to list of edges
    rows, cols = np.nonzero(adj)
    edges = list(zip(rows.tolist(), cols.tolist()))
    return edges


def edges_to_adj(edges, n):
    # List of edges to Adj sparse matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype='float32')
    return adj


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphFull(nn.Module):
    def __init__(self, dset, args):
        super(GraphFull, self).__init__()
        self.args = args
        self.dset = dset

        self.attrs = dset.attrs
        self.objs = dset.objs

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.pairs = dset.pairs
        self.pair2idx = dset.all_pair2idx

        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal

        self.feasibility_adjacency = args.feasibility_adjacency
        self.cosloss = args.cosine_classifier

        self.known_pairs = dset.train_pairs
        seen_pair_set = set(self.known_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in self.pairs]
        self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

        self.feasibility_scores = {}
        self.feasibility_margins = -(1 - self.seen_mask).float()
        self.init_feasibility_scores()

        self.epoch_max_margin = self.args.epoch_max_margin
        self.scale = self.args.cosine_scale
        self.cosine_margin_factor = -args.margin

        # Intsantiate attribute-object relations, needed just to evaluate mined pairs
        self.obj_by_attrs_train = {k: [] for k in self.attrs}
        for (a, o) in self.known_pairs:
            self.obj_by_attrs_train[a].append(o)

        # Intanstiate attribute-object relations, needed just to evaluate mined pairs
        self.attrs_by_obj_train = {k: [] for k in self.objs}
        for (a, o) in self.known_pairs:
            self.attrs_by_obj_train[o].append(a)

        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(self.pairs)


        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current] + self.num_attrs + self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        # feat_dim=512, emb_dim=512
        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)
            self.img2attr = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)
            self.img2obj = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)
            self.anno_classifier = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)

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
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda=0.5,
                             alpha=0.1, variant=False)

        self.softmax = nn.Softmax(dim=1)


    def init_feasibility_scores(self):
        if self.feasibility_adjacency and self.dset.open_world:
            for idx, p in enumerate(self.pairs):
                self.feasibility_scores[p] = 1. if p in self.dset.train_pairs else 0.
                self.feasibility_margins[idx] =  0. if p in self.dset.train_pairs else -10.
        else:
            for idx, p in enumerate(self.pairs):
                self.feasibility_scores[p] = 1.
                self.feasibility_margins[idx] =  0.


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
            return composition_embeds

        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)

        return full_embeddings


    def update_adj(self, epochs=0.):
        self.compute_feasibility(epochs)
        adj = self.adj_from_pairs()
        self.gcn.update_adj(adj)

    def update_dict(self, wdict, row, col, data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)


    def adj_from_pairs(self):

        def edges_from_pairs(pairs):
            weight_dict = {'data': [], 'row': [], 'col': []}

            for i in range(self.displacement):
                self.update_dict(weight_dict, i, i, 1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs

                weight = self.feasibility_scores[(attr, obj)] if self.feasibility_adjacency else 1.

                self.update_dict(weight_dict, attr_idx, obj_idx, weight)
                self.update_dict(weight_dict, obj_idx, attr_idx, weight)

                node_id = idx + self.displacement
                self.update_dict(weight_dict, node_id, node_id, 1.)

                self.update_dict(weight_dict, node_id, attr_idx, weight)
                self.update_dict(weight_dict, node_id, obj_idx, weight)

                self.update_dict(weight_dict, attr_idx, node_id, 1.)
                self.update_dict(weight_dict, obj_idx, node_id, 1.)

            return weight_dict

        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs) + self.displacement, len(self.pairs) + self.displacement))

        return adj

    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o != obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj, o)]
                if temp_score > score:
                    score = temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score


    def compute_feasibility(self,epoch):
        self.gcn.eval()
        embeddings = self.gcn(self.embeddings).detach()

        if self.training:
            self.gcn.train()
        obj_embeddings = embeddings[len(self.attrs):self.displacement]
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                      return_dict=True)
        attr_embeddings = embeddings[:len(self.attrs)]
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                       return_dict=True)

        for (a,o) in self.pairs:
                idx = self.pair2idx[(a, o)]
                if (a, o) not in self.known_pairs:
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    self.feasibility_scores[(a,o)] = max(0.,score)
                    self.feasibility_margins[idx] = score
                else:
                    self.feasibility_scores[(a,o)] = 1.
                    self.feasibility_margins[idx] = 0.
        self.feasibility_margins *= min(1., epoch / self.epoch_max_margin)*self.cosine_margin_factor

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


    def train_forward_normal(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
            obj_fea = self.img2obj(img_feats)
            attr_fea = self.img2attr(img_feats)
        else:
            img_feats = (img)

        if self.cosloss:
            img_feats = F.normalize(img_feats, dim=1)

        current_embeddings = self.gcn(self.embeddings)

        if self.args.train_only:
            pair_embed = current_embeddings[self.train_idx]
        else:
            pair_embed = current_embeddings[
                         self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :]

        pair_embed = pair_embed.permute(1, 0)
        pair_pred = torch.matmul(img_feats, pair_embed)

        
        attr_embs = current_embeddings[:self.num_attrs]
        obj_embs = current_embeddings[self.num_attrs:self.num_attrs+self.num_objs]

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

        if self.cosloss:
            if self.dset.open_world:
                pair_pred = (pair_pred + self.feasibility_margins) * self.scale
            else:
                pair_pred = pair_pred * self.scale

        loss = loss + F.cross_entropy(pair_pred, pairs)

        return loss, None

    def val_forward_dotpr(self, x):
        img = x[0]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
            obj_fea = self.img2obj(img_feats)
            attr_fea = self.img2attr(img_feats)
        else:
            img_feats = (img)

        if self.cosloss:
            img_feats = F.normalize(img_feats, dim=1)

        current_embeddings = self.gcn(self.embeddings)

        pair_embeds = current_embeddings[
                      self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :].permute(1, 0)

        attr_embs = current_embeddings[:self.num_attrs]
        obj_embs = current_embeddings[self.num_attrs:self.num_attrs+self.num_objs]

        attr_pred = torch.matmul(attr_fea, torch.transpose(attr_embs, 0, 1))
        obj_pred = torch.matmul(obj_fea, torch.transpose(obj_embs, 0, 1))
        
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

        score = torch.matmul(img_feats, pair_embeds) + (1/2 * (attr_pred.unsqueeze(2) + obj_pred.unsqueeze(1)).view(attr_pred.size(0),-1))

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]] + (attr_pred[:,self.dset.attr2idx[pair[0]]] + obj_pred[:,self.dset.obj2idx[pair[1]]])

        # for itr, pair in enumerate(self.dset.full_pairs):
        #     scores[pair] = (attr_pred[:,self.dset.attr2idx[pair[0]]] + obj_pred[:,self.dset.obj2idx[pair[1]]]).cpu()

        return score, scores


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
