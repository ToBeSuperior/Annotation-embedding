import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_embedding import load_word_embeddings
from .common import MLP

from itertools import product

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to('cpu')

class CompCos(nn.Module):

    def __init__(self, dset, args):
        super(CompCos, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.attrs = dset.attrs
        self.objs = dset.objs
        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.factor = 2

        self.scale = self.args.cosine_scale

        if dset.open_world:
            self.train_forward = self.train_forward_open
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

            self.activated = False

            # Init feasibility-related variables
            self.possible_pairs = dset.pairs

            self.validation_pairs = dset.val_pairs

            self.feasibility_margin = (1-self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)

        else:
            self.train_forward = self.train_forward_closed

        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)


        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                  dropout=self.args.dropout,
                                  norm=self.args.norm, layers=layers)
        self.img2attr = MLP(dset.feat_dim, args.emb_dim, relu=args.relu, num_layers=args.nlayers, dropout=self.args.dropout,
                                  norm=self.args.norm, layers=layers)
        self.img2obj = MLP(dset.feat_dim, args.emb_dim, relu=args.relu, num_layers=args.nlayers, dropout=self.args.dropout,
                                  norm=self.args.norm, layers=layers)

        # Fixed
        self.composition = args.composition

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

        # Composition MLP
        self.projection = nn.Linear(input_dim * 2, args.emb_dim)
        self.attr_projection = nn.Linear(input_dim, args.emb_dim)
        self.obj_projection = nn.Linear(input_dim, args.emb_dim)


    def freeze_representations(self):
        print('Freezing representations')
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False


    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output = F.normalize(output, dim=1)
        return output


    def compute_feasibility(self):
        obj_embeddings = self.obj_embedder(torch.arange(len(self.objs)).long().to('cuda'))
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                           return_dict=True)
        attr_embeddings = self.attr_embedder(torch.arange(len(self.attrs)).long().to('cuda'))
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                            return_dict=True)

        feasibility_scores = self.seen_mask.clone().float()
        for a in self.attrs:
            for o in self.objs:
                if (a, o) not in self.known_pairs:
                    idx = self.dset.all_pair2idx[(a, o)]
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    feasibility_scores[idx] = score

        self.feasibility_scores = feasibility_scores

        return feasibility_scores * (1 - self.seen_mask.float())


    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o!=obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj,o)]
                if temp_score>score:
                    score=temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score

    def update_feasibility(self,epoch):
        self.activated = True
        feasibility_scores = self.compute_feasibility()
        self.feasibility_margin = min(1.,epoch/self.epoch_max_margin) * \
                                  (self.cosine_margin_factor*feasibility_scores.float().to(device))


    def val_forward(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)

        pair_preds = torch.matmul(img_feats_normed, pair_embeds)
        score = pair_preds
        
        if self.args.lema:
            attr_embed, obj_embed = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)
            attr_embed, obj_embed = self.attr_projection(attr_embed), self.obj_projection(obj_embed)
            attr_embed_normed, obj_embed_normed = F.normalize(attr_embed, dim=1), F.normalize(obj_embed, dim=1)
            
            attr_feats = self.img2attr(img)
            obj_feats = self.img2obj(img)
            
            attr_feats_normed = F.normalize(attr_feats, dim=1)
            obj_feats_normed = F.normalize(obj_feats, dim=1)
            
            attr_pred = torch.matmul(attr_feats_normed, attr_embed_normed.permute(1, 0))
            obj_pred = torch.matmul(obj_feats_normed, obj_embed_normed.permute(1, 0))
            
            attr_pos_score = attr_pred[torch.arange(len(attrs)), attrs[torch.arange(len(attrs))]]
            obj_pos_score = obj_pred[torch.arange(len(objs)), objs[torch.arange(len(objs))]]

            attrs_list = [[x] for x in attrs.tolist()]
            objs_list = [[x] for x in objs.tolist()]
            attr_negs_score = torch.stack([attr_pred[idx][torch.tensor(list(set(range(attr_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(attrs_list)])
            obj_negs_score = torch.stack([obj_pred[idx][torch.tensor(list(set(range(obj_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(objs_list)])
            
            lema_preds = (attr_pred.unsqueeze(2) + obj_pred.unsqueeze(1)).view(attr_pred.size(0),-1)
            score = pair_preds + torch.min(lema_preds, torch.tensor(0.))
            
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return score, scores


    def val_forward_with_threshold(self, x, th=0.):
        img, attrs, objs = x[0], x[1], x[2]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)

        pair_preds = torch.matmul(img_feats_normed, pair_embeds)
        score = pair_preds
        
        if self.args.lema:
            
            attr_embed, obj_embed = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)
            attr_embed, obj_embed = self.attr_projection(attr_embed), self.obj_projection(obj_embed)
            attr_embed_normed, obj_embed_normed = F.normalize(attrs, dim=1), F.normalize(objs, dim=1)
            
            attr_feats = self.img2attr(img)
            obj_feats = self.img2obj(img)
            
            attr_feats_normed = F.normalize(attr_feats, dim=1)
            obj_feats_normed = F.normalize(obj_feats, dim=1)
            
            attr_pred = torch.matmul(attr_feats_normed, attr_embed_normed.permute(1, 0))
            obj_pred = torch.matmul(obj_feats_normed, obj_embed_normed.permute(1, 0))
            
            lema_preds = (attr_pred.unsqueeze(2) + obj_pred.unsqueeze(1)).view(attr_pred.size(0),-1)
            score = pair_preds + torch.min(lema_preds, torch.tensor(0.))

        # Note: Pairs are already aligned here
        mask = (self.feasibility_scores>=th).float()
        score = score*mask + (1.-mask)*(-1.)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return score, scores


    def train_forward_open(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.train_attrs, self.train_objs).permute(1, 0)

        pair_preds = torch.matmul(img_feats_normed, pair_embeds)

        if self.activated:
            pair_preds += (1 - self.seen_mask) * self.feasibility_margin
            loss_cos = F.cross_entropy(self.scale * pair_preds, pairs)
        else:
            pair_preds = pair_preds * self.seen_mask + (1 - self.seen_mask) * (-10)
            loss_cos = F.cross_entropy(self.scale * pair_preds, pairs)
            
        if self.args.lema:
            
            attr_embed, obj_embed = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)
            attr_embed, obj_embed = self.attr_projection(attr_embed), self.obj_projection(obj_embed)
            attr_embed_normed, obj_embed_normed = F.normalize(attr_embed, dim=1), F.normalize(obj_embed, dim=1)
            
            attr_feats = self.img2attr(img)
            obj_feats = self.img2obj(img)
            
            attr_feats_normed = F.normalize(attr_feats, dim=1)
            obj_feats_normed = F.normalize(obj_feats, dim=1)
            
            attr_pred = torch.matmul(attr_feats_normed, attr_embed_normed.permute(1, 0))
            obj_pred = torch.matmul(obj_feats_normed, obj_embed_normed.permute(1, 0))
            
            attr_pos_score = attr_pred[torch.arange(len(attrs)), attrs[torch.arange(len(attrs))]]
            obj_pos_score = obj_pred[torch.arange(len(objs)), objs[torch.arange(len(objs))]]

            attrs_list = [[x] for x in attrs.tolist()]
            objs_list = [[x] for x in objs.tolist()]
            attr_negs_score = torch.stack([attr_pred[idx][torch.tensor(list(set(range(attr_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(attrs_list)])
            obj_negs_score = torch.stack([obj_pred[idx][torch.tensor(list(set(range(obj_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(objs_list)])
            
            attr_lema_loss = torch.mean(attr_negs_score) - torch.mean(attr_pos_score)
            obj_lema_loss = torch.mean(obj_negs_score) - torch.mean(obj_pos_score)
            
            lema_loss = self.args.alpha * (attr_lema_loss + obj_lema_loss)
            
            loss_cos = loss_cos.mean() + lema_loss

        return loss_cos.mean(), None


    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.train_attrs, self.train_objs).permute(1, 0)

        pair_preds = torch.matmul(img_feats_normed, pair_embeds)
        
        if self.args.lema:
            
            attr_embed, obj_embed = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)
            attr_embed, obj_embed = self.attr_projection(attr_embed), self.obj_projection(obj_embed)
            attr_embed_normed, obj_embed_normed = F.normalize(attrs, dim=1), F.normalize(objs, dim=1)
            
            attr_feats = self.img2attr(img)
            obj_feats = self.img2obj(img)
            
            attr_feats_normed = F.normalize(attr_feats, dim=1)
            obj_feats_normed = F.normalize(obj_feats, dim=1)
            
            attr_pred = torch.matmul(attr_feats_normed, attr_embed_normed.permute(1, 0))
            obj_pred = torch.matmul(obj_feats_normed, obj_embed_normed.permute(1, 0))
            
            attr_pos_score = attr_pred[torch.arange(len(attrs)), attrs[torch.arange(len(attrs))]]
            obj_pos_score = obj_pred[torch.arange(len(objs)), objs[torch.arange(len(objs))]]

            attrs_list = [[x] for x in attrs.tolist()]
            objs_list = [[x] for x in objs.tolist()]
            attr_negs_score = torch.stack([attr_pred[idx][torch.tensor(list(set(range(attr_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(attrs_list)])
            obj_negs_score = torch.stack([obj_pred[idx][torch.tensor(list(set(range(obj_pred.size(1))) - set(exclude)))] for idx, exclude in enumerate(objs_list)])
            
            attr_lema_loss = torch.mean(attr_negs_score) - torch.mean(attr_pos_score)
            obj_lema_loss = torch.mean(obj_negs_score) - torch.mean(obj_pos_score)
            
            lema_loss = self.args.alpha * (attr_lema_loss + obj_lema_loss)
            
            loss_cos = loss_cos.mean() + lema_loss
            
        return loss_cos.mean(), None


    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
            return loss, pred
        else:
            with torch.no_grad():
                fast_pred, pred = self.val_forward(x)
            return fast_pred, pred
