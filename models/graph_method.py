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
        self.code_book_unseen_mask = (1. - self.seen_mask).view(self.num_attrs,-1)

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
                                      norm=self.args.norm, layers=layers, relu=False)
            self.img2obj = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=False)

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


    def softmax_with_temperature(self, z, T=1) : 
        '''
        T = 1 -> common softmax
        T = 2,3,... -> scaled softmax 
        '''
        z = z / T 
        max_z, _ = torch.max(z, dim=1)
        exp_z = torch.exp(z-max_z.unsqueeze(1)) 
        sum_exp_z = torch.sum(exp_z, dim=1)
        y = exp_z / sum_exp_z.unsqueeze(1)
        return y
    
    def ce_with_temperature(self, logits, target, temperature=1):

        logits = self.softmax_with_temperature(logits, temperature)
        return torch.mean((-1.0)*torch.sum(torch.log(logits)*target, dim=1))

    def train_forward_normal(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        attr_target = torch.zeros(img.size(0), self.num_attrs).to(device)
        obj_target = torch.zeros(img.size(0), self.num_objs).to(device)

        attr_target[range(img.size(0)), attrs] = 1
        obj_target[range(img.size(0)), objs] = 1

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
            attr_fea = self.img2attr(img)
            obj_fea = self.img2obj(img)
        else:
            img_feats = (img)

        if self.cosloss:
            img_feats = F.normalize(img_feats, dim=1)
            attr_fea = F.normalize(attr_fea, dim=1)
            obj_fea = F.normalize(obj_fea, dim=1)

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

        # if self.args.use_code_book:

        #     temp_attr_code_book = self.dset.attr_code_book.to(device)   # [attr_nums, obj_nums, 512]
        #     attr_sim_code_book = torch.matmul(temp_attr_code_book, attr_fea.permute(1, 0))   # [attr_nums, obj_nums, batch]
        #     attr_pred = torch.sum(torch.permute(attr_sim_code_book, (2, 0, 1)), dim=-1) / torch.sum(self.code_book_unseen_mask, dim=1)
        #     # attr_pred = attr_pred + torch.matmul(attr_fea, attr_embs)

        #     temp_obj_code_book = self.dset.obj_code_book.to(device)   # [attr_nums, obj_nums, 512]
        #     obj_sim_code_book = torch.matmul(temp_obj_code_book, obj_fea.permute(1, 0))    # [attr_nums, obj_nums, batch]
        #     obj_pred = torch.sum(torch.permute(obj_sim_code_book, (2, 1, 0)), dim=-1) / torch.sum(self.code_book_unseen_mask, dim=0)
            # obj_pred = obj_pred + torch.matmul(obj_fea, obj_embs)
        
        # loss = self.ce_with_temperature(attr_pred, attr_target,5) + self.ce_with_temperature(obj_pred, obj_target,5)

        if self.cosloss:
            if self.dset.open_world:
                pair_pred = (pair_pred + self.feasibility_margins) * self.scale
            else:
                pair_pred = pair_pred * self.scale

        loss = F.cross_entropy(pair_pred, pairs) #+ loss

        # pos_score = pair_pred[torch.arange(len(pairs)), pairs[torch.arange(len(pairs))]]
        
        # a = pair_pred.view(-1, self.num_attrs, self.num_objs)
        # f = a[torch.arange(len(attrs)), attrs[torch.arange(len(attrs))]]
        # g = a.permute(0,2,1)[torch.arange(len(objs)), objs[torch.arange(len(objs))]]

        # neg_score = torch.cat((f,g), dim=1)

        # numbers = list(range(neg_score.size(1)))
        # sampled_numbers = random.sample(numbers, 5)
        # neg_score = neg_score[:, sampled_numbers]

        # contrastive_loss = F.triplet_margin_loss(img_feats, pos_embed, neg_embed[:,i,:], margin=1)


        attr_pred = torch.matmul(attr_fea, attr_embs.T) * 0.5
        obj_pred = torch.matmul(obj_fea, obj_embs.T) * 0.5

        # attr_loss = F.cross_entropy(attr_pred, attrs)
        # obj_loss = F.cross_entropy(obj_pred, objs)
        # loss = loss + attr_loss + obj_loss

        attr_pos_score = attr_pred[torch.arange(len(attrs)), attrs[torch.arange(len(attrs))]]
        obj_pos_score = obj_pred[torch.arange(len(objs)), objs[torch.arange(len(objs))]]

        attrs_list = [[x] for x in attrs.tolist()]
        objs_list = [[x] for x in objs.tolist()]
        attr_negs = torch.stack([attr_pred[idx][torch.tensor(list(set(range(attr_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(attrs_list)])
        obj_negs = torch.stack([obj_pred[idx][torch.tensor(list(set(range(obj_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(objs_list)])

        attr_negs_threshold = torch.mean(attr_negs, dim=1).detach()
        obj_negs_threshold = torch.mean(obj_negs, dim=1).detach()

        attr_negs_score = torch.where(attr_negs > attr_negs_threshold.view(-1, 1), attr_negs_threshold.view(-1, 1), attr_negs)
        obj_negs_score = torch.where(obj_negs > obj_negs_threshold.view(-1, 1), obj_negs_threshold.view(-1, 1), obj_negs)

        attr_ctrt_loss = -torch.max(torch.mean(attr_pos_score) - torch.mean(attr_negs_score) + 0., torch.tensor(0))
        obj_ctrt_loss = -torch.max(torch.mean(obj_pos_score) - torch.mean(obj_negs_score) + 0., torch.tensor(0))

        contrastive_loss = (attr_ctrt_loss + obj_ctrt_loss) * 0.1
        # contrastive_loss = torch.tensor(0)

        return loss, contrastive_loss, None

    def val_forward_dotpr(self, x):
        img = x[0]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
            obj_fea = self.img2obj(img)
            attr_fea = self.img2attr(img)
        else:
            img_feats = (img)  

        if self.cosloss:
            img_feats = F.normalize(img_feats, dim=1)
            attr_fea = F.normalize(attr_fea, dim=1)
            obj_fea = F.normalize(obj_fea, dim=1)

        current_embeddings = self.gcn(self.embeddings)

        pair_embeds = current_embeddings[
                      self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :].permute(1, 0)

        attr_embs = current_embeddings[:self.num_attrs]
        obj_embs = current_embeddings[self.num_attrs:self.num_attrs+self.num_objs]

        # if self.args.use_code_book:

        #     temp_attr_code_book = self.dset.attr_code_book.to(device)   # [attr_nums, obj_nums, 512]
        #     attr_sim_code_book = torch.matmul(temp_attr_code_book, attr_fea.permute(1, 0))   # [attr_nums, obj_nums, batch]
        #     attr_pred = self.softmax_with_temperature(torch.sum(torch.permute(attr_sim_code_book, (2, 0, 1)), dim=-1) / torch.sum(self.code_book_unseen_mask, dim=1),5)

        #     temp_obj_code_book = self.dset.obj_code_book.to(device)   # [attr_nums, obj_nums, 512]
        #     obj_sim_code_book = torch.matmul(temp_obj_code_book, obj_fea.permute(1, 0))    # [attr_nums, obj_nums, batch]
        #     obj_pred = self.softmax_with_temperature(torch.sum(torch.permute(obj_sim_code_book, (2, 1, 0)), dim=-1) / torch.sum(self.code_book_unseen_mask, dim=0),5)

        
        attr_pred = torch.matmul(attr_fea, attr_embs.T) * 0.5
        obj_pred = torch.matmul(obj_fea, obj_embs.T) * 0.5

        score = torch.matmul(img_feats, pair_embeds) + (attr_pred.unsqueeze(2) + obj_pred.unsqueeze(1)).view(attr_pred.size(0),-1) 

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
                CE_loss, margin_loss, pred = self.train_forward(x)
                # margin_loss=torch.tensor(0)
                loss_con_pos=torch.tensor(0)
                loss_con_neg=torch.tensor(0)
            else:
                with torch.no_grad():
                    CE_loss, pred = self.val_forward(x)
                    margin_loss=torch.tensor(0)
                    loss_con_pos=torch.tensor(0)
                    loss_con_neg=torch.tensor(0)

        return CE_loss, margin_loss, pred, loss_con_pos, loss_con_neg
