# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, sys, time, random
import os.path as osp
from shutil import copy
from tqdm import tqdm
# from horology import Timing

import numpy as np
# import pandas as pd
import scipy.sparse as ssp
import torch
import torch_sparse

from torch.utils.data import Dataset, DataLoader
# from tensorboardX import SummaryWriter
from torch_geometric.utils import (negative_sampling, to_undirected, add_self_loops)

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator
# from sklearn.metrics import roc_auc_score
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


# speed up negative sampling #####################################################################
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
    ### We modify the code form torch_geometric: we use np.random.default_rng() to speed up the sampling.
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
# end speed up negative sampling  #################################################################

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


class Distance_edges(Dataset):
    def __init__(self, args, data, split_edge, use_val=False):
        self.args = args
        self.data = data
        self.split_edge = split_edge
        self.use_val = use_val
        ##################################################################
        self.num_loop = self.args.max_dist - 2 # we will use high-hop adjs to get distance
        self.posneg_splits, self.edges_dict = self.get_splits_edges()
        self.adj = self.get_adj()
        self.adj_t = self.adj.transpose()
        # neis_sum for Jaccard
        self.neis_sum = np.array(self.adj.sum(axis=1)).flatten()
        # adj_log for Adamic/Adar
        adj_log = 1 / np.log(self.neis_sum)
        adj_log = np.nan_to_num(adj_log, nan=0, posinf=1.5, neginf=0)
        adj_log = self.adj.multiply(adj_log).tocsr()
        self.adj_log = adj_log.transpose()
        # adj_ra for ra
        adj_ra = 1 / self.neis_sum
        adj_ra = np.nan_to_num(adj_ra, nan=0, posinf=0, neginf=0)
        adj_ra = self.adj.multiply(adj_ra).tocsr()
        self.adj_ra = adj_ra.transpose()
        # perms for parallel computing
        step, end = self.args.heurisctic_batch_size, self.adj.shape[0]
        self.perms = [[i, i+step] if i + step < end else [i,end] for i in range(0, end, step)]

    def __len__(self):
        return len(self.perms)

    def __getitem__(self, index):
        perm = self.perms[index]
        high_adjs = {}
        #######################################
        A = self.adj[perm[0]:perm[1]]
        # Adamic/Adar
        AAdar = A.dot(self.adj_log)
        AAdar = AAdar.toarray()
        # RA
        ResA = A.dot(self.adj_ra)
        ResA = ResA.toarray()
        for i in range(self.num_loop):
            # with Timing(name='Times of eli: '):
            if i == 0:
                # cn and Jaccard
                A_sum = np.array(A.sum(axis=1)).repeat(A.shape[1], axis=1)
                A = A.dot(self.adj_t) # csr_matrix.dot do the matrix like the matrix operation in math
                Cneis = (A.toarray()).astype(int)
                A_union = A_sum + self.neis_sum - Cneis
                Jaccard = Cneis / A_union
            else:
                A = A.dot(self.adj_t)
            A.data = np.ones((A.data.shape[0]), dtype=self.args.float) + 1 + i
            A = remove_self_connection(A)
            high_adjs[i] = A.toarray()
        ## prepare SPD #####################################
        A = high_adjs[0]
        for i in range(1, self.num_loop):
            high_adjs[i][A > 0] = 0
            A += high_adjs[i]
        del high_adjs
        A[A == 0] = self.args.max_dist
        ## add distace to edges #######################################################
        dist_edges = {}
        for posneg_split in self.posneg_splits:
            edges = self.edges_dict[posneg_split]
            idx = (edges[0] >= perm[0]) & (edges[0] < perm[1])
            source, target = edges[0][idx] - perm[0], edges[1][idx]
            heuristics = [source + perm[0], target]
            ## impantort: do not change the order of the "sentences" below
            heuristics.append(A[source, target]) # SPD
            heuristics.append(Cneis[source, target]) # CN
            heuristics.append(Jaccard[source, target])
            heuristics.append(AAdar[source, target])
            heuristics.append(ResA[source, target])
            dist_edges[posneg_split] = np.vstack(heuristics)#.astype(source.dtype)

        return dist_edges

    def get_splits_edges(self):
        if self.use_val:
            posneg_splits = ['pos_test', 'neg_test']
        else:
            posneg_splits = ['pos_train', 'neg_train', 'pos_valid', 'neg_valid', 'pos_test', 'neg_test']
        edges_dict = {}
        for posneg_split in posneg_splits:
            posneg, split = posneg_split.split('_')
            if posneg_split == 'neg_train':
                train_size = self.split_edge['train']['edge'].size(0)
                num_neg = train_size * self.args.neg_size
                edges = get_negative_sampling(self.args, self.data, self.split_edge, num_neg_samples=num_neg)
            else:
                edges = self.split_edge[split]['edge'] if posneg == 'pos' else self.split_edge[split]['edge_neg']
                edges = edges.t()
            edges_dict[posneg_split] = edges.numpy()
        return posneg_splits, edges_dict

    def get_adj(self):
        if self.args.heurisctic_directed:
            edge_index = (self.data.edge_index).numpy()
        else:
            edge_index = to_undirected(self.data.edge_index).numpy()
        edge_weight = np.ones([edge_index.shape[1]], dtype=self.args.float)
        num_nodes = self.data.num_nodes
        adj = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        adj = remove_self_connection(adj)

        return adj

def dist_collate_fn(data):
    dist_edges = {}
    for key in data[0]:
        dist_edges[key] = np.hstack([item[key] for item in data])
    return dist_edges

def get_dist_edges(args, data, split_edge, posneg_split):
    ## path
    name = f'dist_edges_spd{args.max_dist}_cn_ja_aa_ra_neg{args.neg_size}'
    if args.use_val and posneg_split.split('_')[1] == 'test':
        name += '_useval'
    name += f'.pt'
    path = osp.join(args.dir_root, 'processed', name)
    ## load data
    reproduce = False
    if args.heurisctic_reproduce:
        if posneg_split == 'pos_train' or (posneg_split == 'pos_test' and args.use_val):
            # the algoristm will produce all posneg_split (i.e. pos_train, neg_train, pos_val, etc.) one time.
            # Therefore, we only need to reproduce at the beginning (i.e. pos_train).
            reproduce = True
    ## get dist_edges #####################################################################
    if osp.exists(path) and reproduce == False:
        dist_edges = torch.load(path)
    else:
        use_val = True if '_useval' in path else False
        data_dist = Distance_edges(args, data, split_edge, use_val)
        loader = DataLoader(data_dist, batch_size=args.num_workers, num_workers=args.num_workers, shuffle=False, collate_fn=dist_collate_fn)
        res, dist_edges = [], {}
        for re in tqdm(loader, ncols=50):
            res.append(re)
        for key in res[0]:
            dist_edges[key] = np.hstack([item[key] for item in res])
        ## save dist_edges to path ####################
        torch.save(dist_edges, path, pickle_protocol=4)
    #######################################################################################
    #######################################################################################
    ## process heuristics for model #######################################################
    ## impantort: do not change the order of the "if sentences" below
    def int_heuristics(heur, max, mag=1):
        heur = (heur*mag).astype(int)
        heur[heur < 0] = 0
        heur[heur > max] = max
        return heur
    raw_edges = dist_edges[posneg_split]
    i=2
    if args.use_dist:
        i +=1
    else:
        raw_edges = np.delete(raw_edges, i, 0)
    if args.use_cn:
        raw_edges[i] = int_heuristics(heur=raw_edges[i], max=args.max_cn, mag=1)
        i += 1
    else:
        raw_edges = np.delete(raw_edges, i, 0)
    if args.use_ja:
        raw_edges[i] = int_heuristics(heur=raw_edges[i], max=args.max_ja, mag=args.mag_ja)
        i += 1
    else:
        raw_edges = np.delete(raw_edges, i, 0)
    if args.use_aa:
        raw_edges[i] = int_heuristics(heur=raw_edges[i], max=args.max_aa, mag=args.mag_aa)
        i += 1
    else:
        raw_edges = np.delete(raw_edges, i, 0)
    if args.use_ra:
        raw_edges[i] = int_heuristics(heur=raw_edges[i], max=args.max_ra, mag=args.mag_ra)
        i += 1
    else:
        raw_edges = np.delete(raw_edges, i, 0)

    edges = torch.from_numpy(raw_edges.astype(int))

    return edges.t()


def get_edges(args, data, split_edge, posneg_split):
    posneg, split = posneg_split.split('_')
    if args.use_dist or args.use_cn or args.use_ja or args.use_aa or args.use_ra:
        edges = get_dist_edges(args, data, split_edge, posneg_split)
    else:
        if posneg_split == 'neg_train':
            edges = None
        else:
            edges = split_edge[split]['edge'] if posneg == 'pos' else split_edge[split]['edge_neg']
    return edges


def get_adj_degree(args, data, split_edge, posneg_split):
    ## path
    name = f'adj_hop{args.adj_hop}_neg{args.adj_neg}'
    if args.use_val and posneg_split.split('_')[1] == 'test':
        name += '_useval'
    name += f'.pt'
    path = osp.join(args.dir_root, 'processed', name)

    if osp.exists(path) and (posneg_split == 'pos_train' or (posneg_split == 'pos_test' and args.use_val))==False:
        adj_degree = torch.load(path)
    else:
        num_nodes = data.num_nodes
        adj_degree = {}
        # 1-hop adj ##########################################################################################
        edge_index = data.edge_index.numpy()
        edge_weight = data.edge_weight.numpy()
        adj = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes), dtype=args.float)
        adj = remove_self_connection(adj)
        adj_degree['1_hop'] = adj.copy()
        adj_degree['degree'] = np.asarray(adj.sum(axis=1))
        # high-hop adj for high-hop adj_gnn #####################################################################
        if args.adj_hop > 1:
            # base adj with weight 1 #########################################################################
            adj.data = np.ones((len(adj.data)), dtype=args.float)
            adj_t = adj.transpose()
            As, high_adjs = [], {}
            for i in range(args.adj_hop):
                A = adj if i == 0 else A.dot(adj_t)
                A.data = np.ones((A.data.shape[0]), dtype=args.float)
                A = remove_self_connection(A)
                As.append(A)
            del A
            print('remove non-shortest distance neighbors in high adjs ...')
            for i in range(args.adj_hop):
                a_hop = As[i]
                if i > 0:
                    for j in range(i):
                        a_hop = a_hop - As[j]
                        a_hop = a_hop.multiply(a_hop > 0)
                # remove self connection
                a_hop = remove_self_connection(a_hop)
                high_adjs[f'{i + 1}_hop'] = a_hop.astype(args.float)
        # neg neighbors sampling
        num_neg = int(num_nodes*args.adj_neg)
        if num_neg>0:
            neg_edges = get_negative_sampling(args, data, split_edge, num_neg_samples=num_neg)
            neg_edges = neg_edges.numpy()

        # adj_gnn as torch SparseTensor ##########################################################################
        ## add self connection
        row, col, v = [np.arange(num_nodes)], [np.arange(num_nodes)], [np.ones(num_nodes)]
        ## combine various-hop adjs into one
        if args.adj_hop == 1:
            a = adj_degree['1_hop'].tocoo()
            row.append(a.row)
            col.append(a.col)
            v.append(a.data)
        elif args.adj_hop > 1:
            for i in range(args.adj_hop):
                a = high_adjs[f'{i + 1}_hop'].tocoo()
                row.append(a.row)
                col.append(a.col)
                v.append(np.ones(len(a.row)) + i)
        if num_neg > 0:
            row.append(neg_edges[0])
            col.append(neg_edges[1])
            v.append(np.ones(len(neg_edges[0])) + args.adj_neg_dist)
        row, col, v = np.hstack(row), np.hstack(col), np.hstack(v)
        i = np.vstack((row, col))
        adj_gnn = torch.sparse_coo_tensor(i, v, (num_nodes, num_nodes)).coalesce()
        ## to torch_sparse.SparseTensor
        indices, values = adj_gnn.indices(), adj_gnn.values().float()
        adj_gnn = torch_sparse.SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=(adj_gnn.size(0), adj_gnn.size(1)), is_sorted=True)
        adj_gnn.storage.rowptr()
        adj_gnn.storage.csr2csc()
        adj_degree[f'adj_gnn'] = adj_gnn
        # ## add self connection to 1-hop adj ######################################################################
        # adj_degree['adj'] = adj_degree['adj'] + ssp.identity(adj.shape[0], dtype=args.float)

        # saving high_adjs data ############################################################################
        print(f'saving adj_degree data to {path}')
        torch.save(adj_degree, path, pickle_protocol=4)
    # print('finish loading high_adjs data')
    return adj_degree

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
    args.num_nodes = data.num_nodes = num_nodes = int(data.edge_index.max()) + 1
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

    # ## adj and node degree
    # edge_index = data.edge_index.numpy()
    # edge_weight = data.edge_weight.numpy()
    # adj = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes), dtype=args.float)
    # # first remove self connection and then add new self connection
    # adj[np.arange(num_nodes), np.arange(num_nodes)] = 0
    # adj.eliminate_zeros()
    # adj = adj + ssp.identity(num_nodes, dtype=args.float)
    # # node degree
    # data.node_degree = np.asarray(adj.sum(axis=1))
    # # adj in torch_sparse.SparseTensor
    # adj = adj.tocoo()
    # row, col, value = torch.Tensor(adj.row).long(), torch.Tensor(adj.col).long(), torch.Tensor(adj.data).float()
    # adj_gnn = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
    # adj_gnn.storage.rowptr()
    # adj_gnn.storage.csr2csc()
    # data.adj_gnn = adj_gnn

    return data, split_edge

class graph_prepare():
    def __init__(self, args, posneg_split):
        self.args = args
        self.posneg_split = posneg_split
        self.data, self.split_edge = get_dataset(self.args, self.posneg_split)
        self.x = self.data.x
        adj_degree = get_adj_degree(self.args, self.data, self.split_edge, self.posneg_split)
        self.adj_gnn = adj_degree['adj_gnn']
        self.degree = adj_degree['degree']
        self.degree[self.degree > args.max_degree] = args.max_degree
        ## edges ############################################################
        # with Timing(name='Times of validation: '):
        if posneg_split != 'neg_train':
            self.edges = get_edges(self.args, self.data, self.split_edge, self.posneg_split)
        else:
            ## setting initial total number of negs in training process
            edges = self.split_edge['train']['edge']
            train_size = edges.size(0)
            self.num_neg = int(train_size * self.args.neg_size)
            ## get neg edges
            if self.args.use_dist or self.args.use_cn or self.args.use_ja or self.args.use_aa or self.args.use_ra:
                self.edges_neg = get_edges(self.args, self.data, self.split_edge, self.posneg_split)
                # self.edges_neg = self.shuffle_edges(self.edges_neg)
                # self.dist_cn = self.edges_neg[:, 2:]
            else:
                self.edges_neg = get_negative_sampling(self.args, self.data, self.split_edge, num_neg_samples=self.num_neg).t()
            ## resample setting
            self.epoch_neg_size = int(min(train_size, self.args.batch_size * (self.args.batch_num + 1)))
            self.cnt_max = int(self.edges_neg.size(0) / self.epoch_neg_size)
            self.cnt = 0

        print(f'finish loading data_{posneg_split}')

    def resample_neg_edges(self):
        if self.cnt >= self.cnt_max:
            self.cnt = 0
            if self.args.use_dist or self.args.use_cn or self.args.use_ja or self.args.use_aa or self.args.use_ra:
                if self.args.heurisctic_reuse:
                    print('re-use dist edges for neg samples')
                    self.args.heurisctic_reproduce = False
                    self.edges_neg = get_edges(self.args, self.data, self.split_edge, self.posneg_split)
                    self.edges_neg = self.shuffle_edges(self.edges_neg)
                else:
                    print('re-produce dist edges for neg samples')
                    # reproduce edges by heurisctic_reproduce = True and 'pos_train'
                    self.args.heurisctic_reproduce = True
                    _ = get_edges(self.args, self.data, self.split_edge, 'pos_train')[0]
                    self.edges_neg = get_edges(self.args, self.data, self.split_edge, self.posneg_split)
            else:
                self.edges_neg = get_negative_sampling(self.args, self.data, self.split_edge, num_neg_samples=self.num_neg).t()
                # self.edges_neg = self.shuffle_edges(self.edges_neg)
            self.cnt_max = int(self.edges_neg.size(0) / self.epoch_neg_size)
        self.edges = self.edges_neg[self.epoch_neg_size * self.cnt:self.epoch_neg_size * (self.cnt + 1)]
        self.cnt += 1

    def shuffle_edges(self, edges):
        num_edge = edges.size(0)
        perm = np.random.permutation(num_edge)
        edges = edges[perm]
        return edges

##################################################################################################
##################################################################################################
# dataset preparing end ##########################################################################
##################################################################################################
##################################################################################################
