#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import math

#Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER

best_auc = 0
best_hm = 0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
 
def main():
    # Get arguments and start logging
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
    print(args)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_triplet_loss=args.train_triplet_loss,
        train_only= args.train_only,
        open_world=args.open_world
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)
    
    seen_testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='seen_val',
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    seen_testloader = torch.utils.data.DataLoader(
        seen_testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)
    
    unseen_testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='unseen_val',
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    unseen_testloader = torch.utils.data.DataLoader(
        unseen_testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    train = train_normal

    evaluator_val =  Evaluator(testset, model)

    print(model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)
        model.dset.train_triplet_loss = False

    if args.use_code_book : 
        path = 'DATA_ROOT/' + args.data_dir 
        if not args.train_triplet_loss:
            print('stored code book loading...')
            # db = torch.load(path + '/code_book.pt')
            # model.dset.code_book = db['code_book']
            model.dset.code_book = model.make_code_book(trainset)
            model.dset.train_triplet_loss = False
            model.load_state_dict(torch.load('DATA_ROOT/' + args.data_dir + '/pretrained.pt'))
        else:
            print('code book starting...')
            for epoch in tqdm(range(start_epoch, 1), desc = 'semantic difference epoch'):
                train(epoch, image_extractor, model, trainloader, optimizer, writer, train_cls_only=True)
                # break
                # if epoch % args.eval_val_every == 0:
                #     with torch.no_grad(): # todo: might not be needed
                #         test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath, train_cls_only=True)

                # if epoch % args.eval_val_every == 0:
                #     with torch.no_grad(): # todo: might not be needed
                #         test(epoch, image_extractor, model, seen_testloader, evaluator_val, writer, args, logpath, train_cls_only=True)

                # if epoch % args.eval_val_every == 0:
                #     with torch.no_grad(): # todo: might not be needed
                #         test(epoch, image_extractor, model, unseen_testloader, evaluator_val, writer, args, logpath, train_cls_only=True)
            model.eval()
            # model.make_code_book(store_path = path + '/code_book.pt')
            model.dset.code_book = model.make_code_book(trainset)
            torch.save(model.state_dict(), 'DATA_ROOT/' + args.data_dir + '/pretrained.pt')

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train(epoch, image_extractor, model, trainloader, optimizer, writer)

        if epoch % args.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                model.dset.code_book = model.make_code_book(testset)
                test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath)

    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer, train_cls_only=False):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train() # Let's switch to training

    train_loss = 0.0 
    train_CE_loss = 0.0
    train_margin_loss = 0.0
    train_con_pos = 0.0
    train_con_neg = 0.0
    
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        optimizer.zero_grad()
        data  = [d.to(device) for d in data]

        if image_extractor and not train_cls_only:
            data[0] = image_extractor(data[0])
        
        CE_loss, margin_loss, _, loss_con_pos, loss_con_neg = model(data, train_cls_only)
        loss = CE_loss + margin_loss

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_CE_loss += CE_loss.item()
        train_margin_loss += margin_loss.item()
        train_con_pos += loss_con_pos.item()
        train_con_neg += loss_con_neg.item()

    optimizer.zero_grad()
    train_loss = train_loss/len(trainloader)
    train_CE_loss = train_CE_loss/len(trainloader)
    train_margin_loss = train_margin_loss/len(trainloader)
    train_con_pos = train_con_pos/len(trainloader)
    train_con_neg = train_con_neg/len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {}| CE_loss: {}| margin_loss: {}| con_pos: {}| con_neg: {}'.format(
        epoch, round(train_loss, 2), round(train_CE_loss, 2), round(train_margin_loss, 2), round(train_con_pos, 2), round(train_con_neg, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, train_cls_only=False):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()
 
    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _,_, predictions, _, _ = model(data, train_cls_only)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)
    if train_cls_only:
        print(torch.mean(torch.tensor(all_pred)))
        return

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    ############## test 0709 : pair 따로 accuracy만 보기 ##################

    # print(f'Test Epoch: {epoch}')

    # scores = {k: v.to('cpu') for k, v in all_pred_dict.items()}
    # scores = torch.stack(
    #     [scores[obj] for obj in evaluator.dset.pairs], 1
    # )
    # scores = torch.argmax(scores,1)
    # predict_acc = torch.sum(scores==all_pair_gt) / len(scores)

    # print("classifier pair prediction accuracy  : {}".format(predict_acc))


    # return

    ############## test 0709 : pair 따로 accuracy만 보기 ##################


    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)
    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)