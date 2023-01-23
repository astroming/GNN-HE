import argparse, os, sys, time, random
import os.path as osp
from shutil import copy
from tqdm import tqdm
from horology import Timing

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torch_sparse

from torch.utils.data import Dataset, DataLoader
# from tensorboardX import SummaryWriter
from torch_geometric.utils import (negative_sampling, to_undirected, add_self_loops)

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


##################################################################################################
##################################################################################################
# evaluation setting #############################################################################
##################################################################################################
##################################################################################################

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if len(result) > 0:
                argmax = result[:, 0].argmax().item()
                print(f'Run {run:02d}:', file=f)
                print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
                print(f'Highest Eval Epoch: {argmax}', file=f)
                print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                if len(r) > 0:
                    valid = r[:, 0].max().item()
                    if valid == 0:
                        continue
                    test = r[r[:, 0].argmax(), 1].item()
                    best_results.append((valid, test))
            if len(best_results) > 0:
                best_result = torch.tensor(best_results)
                print(f'All non-Nan runs: {len(best_result)}', file=f)
                r = best_result[:, 0]
                print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
                r = best_result[:, 1]
                print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)


def get_loggers(args):
    loggers = {
        # 'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        # 'Hits@30': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
        'MRR': Logger(args.runs, args),
        'AUC': Logger(args.runs, args),
        'ACC': Logger(args.runs, args),
    }

    return loggers


def get_eval_result(args, val_pred, val_true, test_pred, test_true):
    results = {}
    if args.dataset.startswith('ogbl'):
        evaluator = Evaluator(name=args.dataset)
        pos_val_pred = val_pred[val_true == 1]
        neg_val_pred = val_pred[val_true == 0]
        pos_test_pred = test_pred[test_true == 1]
        neg_test_pred = test_pred[test_true == 0]

        if 'hits@' in evaluator.eval_metric:
            for K in args.hitK:
                evaluator.K = K
                valid_hits = evaluator.eval({
                    'y_pred_pos': pos_val_pred,
                    'y_pred_neg': neg_val_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']

                results[f'Hits@{K}'] = (valid_hits, test_hits)

        elif 'mrr' == evaluator.eval_metric:
            neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
            neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
            valid_mrr = evaluator.eval({
                'y_pred_pos': pos_val_pred,
                'y_pred_neg': neg_val_pred,
            })['mrr_list'].mean().item()

            test_mrr = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })['mrr_list'].mean().item()

            results['MRR'] = (valid_mrr, test_mrr)
    # AUC
    # valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    # results['AUC'] = (valid_auc, test_auc)
    # acc
    # total_true += labels.size(0)
    # correct += (abs(outputs - labels) < 0.5).sum().item()
    # acc=correct / total_true

    return results


##################################################################################################
##################################################################################################
# evaluation setting end #########################################################################
##################################################################################################
##################################################################################################

##################################################################################################
##################################################################################################
# dataset preparing ##############################################################################
##################################################################################################
##################################################################################################

def maybe_num_nodes(edge_index, num_nodes=None):
    # copied from torch_geometric
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))

def sample(high: int, size: int, device=None):
    # copied from torch_geometric
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None,
                      method="sparse", force_undirected=False):
    # modify the code form torch_geometric: we use np.random.default_rng() to speed up the sampling.
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return. If set to :obj:`None`, will try to return a
            negative edge for every positive edge. (default: :obj:`None`)
        method (string, optional): The method to use for negative sampling,
            *i.e.*, :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '|V|^2 - |E| < |E|'.
    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

    row, col = edge_index

    if force_undirected:
        num_neg_samples = num_neg_samples // 2

        # Upper triangle indices: N + ... + 1 = N (N + 1) / 2
        size = (num_nodes * (num_nodes + 1)) // 2

        # Remove edges in the lower triangle matrix.
        mask = row <= col
        row, col = row[mask], col[mask]

        # idx = N * i + j - i * (i+1) / 2
        idx = row * num_nodes + col - row * (row + 1) // 2
    else:
        idx = row * num_nodes + col

    # Percentage of edges to oversample so that we are save to only sample once
    # (in most cases).
    alpha = abs(1 / (1 - 1.1 * (edge_index.size(1) / size)))

    if method == 'dense':
        mask = edge_index.new_ones(size, dtype=torch.bool)
        mask[idx] = False
        mask = mask.view(-1)

        perm = sample(size, int(alpha * num_neg_samples),
                      device=edge_index.device)
        perm = perm[mask[perm]][:num_neg_samples]

    else:
        rng = np.random.default_rng()
        perm = rng.choice(size, int(alpha * num_neg_samples), replace=False, shuffle=False)
        perm = np.setdiff1d(perm, idx.to('cpu').numpy())
        perm = perm[:num_neg_samples]
        rng.shuffle(perm)
        perm = torch.from_numpy(perm).to(edge_index.device)
        # perm = sample(size, int(alpha * num_neg_samples))
        # mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
        # perms = perm[~mask][:num_neg_samples].to(edge_index.device)

    if force_undirected:
        # (-sqrt((2 * N + 1)^2 - 8 * perm) + 2 * N + 1) / 2
        row = torch.floor((-torch.sqrt((2. * num_nodes + 1.) ** 2 - 8. * perm) +
                           2 * num_nodes + 1) / 2)
        col = perm - row * (2 * num_nodes - row - 1) // 2
        neg_edge_index = torch.stack([row, col], dim=0).long()
        neg_edge_index = to_undirected(neg_edge_index)
    else:
        row = perm // num_nodes
        col = perm % num_nodes
        neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index



def get_x_scale(x, method='gaussian'):
    # method: gaussian, maxmin
    if method == 'gaussian':
        ds_mean = torch.mean(x, dim=0)
        ds_std = torch.std(x, dim=0)
        x = (x - ds_mean) / ds_std
    elif method == 'maxmin':
        ds_min = torch.min(x, dim=0)[0]
        ds_max = torch.max(x, dim=0)[0]
        x = (x - ds_min) / (ds_max - ds_min)
    return x

def get_path(args, name, posneg_split):
    # initialize the dirs of high_adjs
    posneg, split = posneg_split.split('_')
    full_name = f'{name}'
    if args.use_val and split == 'test':
        full_name += '_useval'
    full_name += f'.pt'
    path_name = osp.join(args.dir_root, 'processed', full_name)
    return path_name

def remove_self_connection(adj):
    adj[np.arange(adj.shape[0]), np.arange(adj.shape[0])] = 0
    adj.eliminate_zeros()
    return adj


def get_negative_sampling(args, data, split_edge, num_neg_samples=None):
    print('negative sampling ...')
    edges = split_edge['train']['edge']
    train_size = edges.size(0)
    if num_neg_samples == None:
        num_neg_samples = int(min(train_size, args.batch_size * (args.batch_num + 1)))
    edge_index, _ = add_self_loops(data.edge_index)
    edges = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=num_neg_samples, method=args.dense_sparse)
    print('negative sampling finished')
    return edges


def get_dataset(args, posneg_split):
    ## base data ################################################################################################################
    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    if args.dataset == 'ogbl-citation2':
        split_edge['train']['edge'] = torch.vstack((split_edge['train']['source_node'], split_edge['train']['target_node'])).t()
        split_edge['valid']['edge'] = torch.vstack((split_edge['valid']['source_node'], split_edge['valid']['target_node'])).t()
        split_edge['valid']['edge_neg'] = torch.vstack((split_edge['valid']['source_node'].repeat_interleave(
            split_edge['valid']['target_node_neg'].size(1)), split_edge['valid']['target_node_neg'].view(-1))).t()
        split_edge['test']['edge'] = torch.vstack((split_edge['test']['source_node'], split_edge['test']['target_node'])).t()
        split_edge['test']['edge_neg'] = torch.vstack((split_edge['test']['source_node'].repeat_interleave(
            split_edge['test']['target_node_neg'].size(1)), split_edge['test']['target_node_neg'].view(-1))).t()

    ## basic pre-processing #####################################################################################################
    args.dir_root = dataset.root
    data.num_nodes = args.num_nodes = int(data.edge_index.max()) + 1
    ## x
    if data.x is not None:
        data.x = data.x.to(torch.float32)
    if args.dataset == 'ogbl-ppa':
        data.x = (data.x == 1).nonzero()[:, [1]]

    ## edge index and edge weight
    if args.dataset == 'ogbl-collab':
        idx = (data.edge_year > args.collab_year).squeeze(1)
        edge_index = data.edge_index.transpose(0, 1)
        data.edge_index = edge_index[idx].transpose(0, 1)
        data.edge_weight = data.edge_weight[idx]
        data.edge_year = data.edge_year[idx]

        pos_train = split_edge['train']
        idx = pos_train['year'] > args.collab_year
        split_edge['train']['edge'] = pos_train['edge'][idx]
        split_edge['train']['weight'] = pos_train['weight'][idx]
        split_edge['train']['year'] = pos_train['year'][idx]

    if 'edge_weight' not in data:
        data.edge_weight = torch.ones([data.edge_index.size(1)])
    else:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    if args.use_val and posneg_split.split('_')[1] == 'test':
        # add valid edge into adj in testing stage
        val_edge_index = split_edge['valid']['edge'].t()
        data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        data.edge_weight = torch.cat([data.edge_weight, torch.ones([val_edge_index.size(1)])], 0)

    if args.coalesce and args.directed:
        # compress mutli-edge into single edge with weight
        data.edge_index, data.edge_weight = torch_sparse.coalesce(data.edge_index, data.edge_weight, data.num_nodes, data.num_nodes)
    if not args.directed:
        data.edge_index = to_undirected(data.edge_index)
        data.edge_weight = torch.ones([data.edge_index.size(1)])
    if not args.use_weight:
        data.edge_weight = torch.ones([data.edge_index.size(1)])

    return data, split_edge

class graph_prepare():
    

##################################################################################################
##################################################################################################
# dataset preparing end ##########################################################################
##################################################################################################
##################################################################################################
