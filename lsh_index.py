import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorWithPadding

from LSH.citadel_utils import process_check_point
from LSH.lsh_model import LSHEncoder
from dataloader import BenchmarkDataset

class LSHIndexCoil1:
    def __init__(self, config):
        self.config = config
        self.transformer_model_dir = config.transformer_model_dir
        self.encode_loader = None
        self.context_encoder = None
        self.dataset = None

    def _prepare_data(self):
        tokenizer = BertTokenizer.from_pretrained(self.transformer_model_dir, use_fast=False)
        self.dataset = BenchmarkDataset(self.config, tokenizer)
        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=self.config.encode_batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer,
                max_length=self.config.max_seq_len,
                padding='max_length'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.config.dataloader_num_workers,
        )

    def _prepare_model(self):
        pass


class LSHIndex:
    def __init__(self,config):
        self.config = config
        self.transformer_model_dir = config.transformer_model_dir
        self.encode_loader = None
        self.context_encoder = None
        self.dataset = None

    def _prepare_data(self):
        tokenizer = BertTokenizer.from_pretrained(self.transformer_model_dir, use_fast=False)
        self.dataset = BenchmarkDataset(self.config, tokenizer)
        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=self.config.encode_batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer,
                max_length=self.config.max_seq_len,
                padding='max_length'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.config.dataloader_num_workers,
        )

    def _prepare_model(self):
        self.context_encoder = LSHEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout,
                                              tok_projection_dim=self.config.tok_projection_dim,
                                              cls_projection_dim=self.config.cls_projection_dim)

        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]

        checkpoint_dict = process_check_point(checkpoint_dict)

        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _encode(self):

        token_dense_repr_list = []
        token_sparse_repr_list = []
        cls_list = []
        num_token_list = []
        for batch in tqdm(self.encode_loader):
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        if k == "corpus_ids":
                            continue
                        contexts_ids_dict[k] = v.to(self.config.device)
                batch.data = contexts_ids_dict
                del contexts_ids_dict

                contexts_repr = self.context_encoder(batch)


            contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items()}
            token_dense_repr_list.append(contexts_repr["token_dense_repr"])
            # token_sparse_repr_list.append(contexts_repr["token_sparse_repr"])
            cls_list.append(contexts_repr["cls_repr"])
            num_token_list+=list(contexts_repr["num_token"].detach().cpu().numpy())



        all_dense_repr = torch.cat(token_dense_repr_list)
        # all_sparse_repr = torch.cat(token_sparse_repr_list)
        all_cls = torch.cat(cls_list)

        if self.config.hashing == "hyperspherical":
            hash_bins, hash_matrix , hash_v_mean, hash_v_std= self.hyperspherical_hashing(all_dense_repr, num_token_list)
        save_dir = os.path.join(self.config.index_dir, f'{self.config.dataset}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(hash_bins, r"{}/{}".format(save_dir, 'hash_bins.pt'))
        torch.save(hash_matrix, r"{}/{}".format(save_dir, 'hash_matrix.pt'))
        torch.save(all_cls, r"{}/{}".format(save_dir, 'all_cls.pt'))
        torch.save(hash_v_mean, r"{}/{}".format(save_dir, 'hash_v_mean.pt'))
        torch.save(hash_v_std, r"{}/{}".format(save_dir, 'hash_v_std.pt'))




    def hyperspherical_hashing(self, all_dense_repr, num_token_list):
        normalized_dense_repr, hash_v_mean, hash_v_std = normalization(all_dense_repr)

        hash_matrix = torch.rand(all_dense_repr.shape[1], self.config.num_hash)
        hash_value_matrix = normalized_dense_repr @ hash_matrix
        topk_indices = hash_value_matrix.topk( self.config.num_hash, dim=1).indices
        hash_bins = {}
        p = 0
        offset = num_token_list[p]
        for i , row in enumerate(tqdm(topk_indices)):

            if i > offset -1:
                p += 1
                offset += num_token_list[p]


            hash_value = tuple(row.tolist())
            if hash_value in list(hash_bins.keys()):
                hash_bins[hash_value]["dense_repr"].append((all_dense_repr[i],p))
                # hash_bins[hash_value]["sparse_repr"].append((all_sparse_repr[i],p))
            else:
                hash_bins[hash_value] = {}
                hash_bins[hash_value]["dense_repr"] = [(all_dense_repr[i],p)]
                # hash_bins[hash_value]["sparse_repr"] = [(all_sparse_repr[i],p)]

        for hash_value, values in hash_bins.items():
            hash_bins[hash_value]["dense_repr"] = Matrixing(values["dense_repr"])
            # hash_bins[hash_value]["sparse_repr"] = Matrixing(values["sparse_repr"])
        return hash_bins, hash_matrix, hash_v_mean, hash_v_std

    def run(self):
        self._prepare_model()
        self._prepare_data()
        self._encode()

def normalization(tensor):
    this_tensor = deepcopy(tensor)
    mean = this_tensor.mean(dim=0)
    std = this_tensor.std(dim=0)
    new_tensor = (this_tensor - mean) / std
    norms = torch.norm(new_tensor, p=2, dim=1, keepdim=True)
    return new_tensor/norms, mean, std

def Matrixing(repr_pair_list):
    repr = torch.stack([repr_pair[0] for repr_pair in repr_pair_list], dim=0)
    ivd = [repr_pair[1] for repr_pair in repr_pair_list]
    return (repr,ivd)


