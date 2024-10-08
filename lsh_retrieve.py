import gzip
import os
import perf_event
from copy import deepcopy

import faiss
import ir_measures
import pandas as pd
import torch
import torch_scatter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorWithPadding

from LSH.citadel_utils import process_check_point
from LSH.lsh_model import LSHEncoder
from LSH.transformer import HFTransform
from dataloader import LSHQueryDataset, LSHDataset
from utils import create_this_perf


class LSHRetrieve:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.topk = config.topk
        self.perf_path = None
        self.rank_path = None
        self.eval_path = None
        self.context_encoder = None
        self.searcher = None
        self.device = device

    def setup(self):
        checkpoint_dict = self._load_checkpoint()
        self._set_up_model(checkpoint_dict)
        self._sep_up_searcher()



    def _load_checkpoint(self):
        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]
        checkpoint_dict = process_check_point(checkpoint_dict)
        return checkpoint_dict

    def _set_up_model(self, checkpoint_dict):
        self.context_encoder = LSHEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout,
                                              tok_projection_dim=self.config.tok_projection_dim,
                                              cls_projection_dim=self.config.cls_projection_dim)

        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _sep_up_searcher(self):
        self.searcher = HypersphericalLSHIndex(self.config)
        self.searcher.load_index()
        self.searcher.set_scores()

    def _prepare_data(self):
        transform = HFTransform(self.config.transformer_model_dir, self.config.max_seq_len)
        self.dataset = LSHQueryDataset(self.config, transform)
        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )


    def _create_save_path(self):
        save_dir = r"{}/hypersphericalLSH/{}".format(self.config.results_save_to, self.config.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.perf_path = r"{}/{}".format(save_dir, "perf_results")
        self.rank_path = r"{}/{}".format(save_dir, "rank_results")
        self.eval_path = r"{}/{}".format(save_dir, "eval_results")
        if not os.path.exists(self.perf_path):
            os.makedirs(self.perf_path)
        if not os.path.exists(self.rank_path):
            os.makedirs(self.rank_path)
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

    def _retrieve(self):
        self._create_save_path()
        perf_encode = perf_event.PerfEvent()
        perf_retrival = perf_event.PerfEvent()
        all_query_match_scores = []
        all_query_inids = []
        all_perf = []
        for batch in tqdm(self.encode_loader):
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        contexts_ids_dict[k] = v.to(self.config.encode_device)
                batch.data = contexts_ids_dict
                del contexts_ids_dict
            perf_encode.startCounters()
            queries_repr = self.context_encoder(batch)
            queries_repr = {k: v.detach().cpu() for k, v in queries_repr.items()}
            token_dense_repr = queries_repr["token_dense_repr"].squeeze(0)
            token_dense_mask = torch.any(token_dense_repr != 0, dim=1)
            token_dense_repr = token_dense_repr[token_dense_mask]
            cls_repr = queries_repr["cls_repr"]
            perf_encode.stopCounters()
            perf_retrival.startCounters()
            batch_top_scores, batch_top_ids = self.searcher.search(cls_repr, token_dense_repr,)
            perf_retrival.stopCounters()
            this_perf = create_this_perf(perf_encode, perf_retrival)
            all_perf.append(this_perf)
            all_query_match_scores.append(batch_top_scores)
            all_query_inids.append(batch_top_ids)

        all_query_match_scores = torch.cat(all_query_match_scores, dim=0)
        all_query_exids = torch.cat(all_query_inids, dim=0)
        self._save_perf(all_perf)
        path = self._save_ranks(all_query_match_scores, all_query_exids)
        return path

    def _save_perf(self, all_perf: list):
        columns = ["encode_cycles", "encode_instructions",
                   "encode_L1_misses", "encode_LLC_misses",
                   "encode_L1_accesses", "encode_LLC_accesses",
                   "encode_branch_misses", "encode_task_clock",
                   "retrieval_cycles", "retrieval_instructions",
                   "retrieval_L1_misses", "retrieval_LLC_misses",
                   "retrieval_L1_accesses", "retrieval_LLC_accesses",
                   "retrieval_branch_misses", "retrieval_task_clock"]
        perf_df = pd.DataFrame(all_perf, columns=columns)
        perf_df.to_csv(r"{}/num_hash-{}.perf.csv".format(self.perf_path,
                                                                     str(self.config.num_hash)),
                       index=False)

    def _save_ranks(self, scores, indices):
        path = r"{}/num_hash-{}.run.gz".format(self.rank_path, str(self.config.prune_weight))
        rh = faiss.ResultHeap(scores.shape[0], self.topk)
        if self.device == "cpu":
            rh.add_result(-scores.numpy(), indices.numpy())
        elif self.device == "gpu":
            scores = scores.to("cpu").to(torch.float32)
            indices = indices.to("cpu").to(torch.int64)
            rh.add_result(-scores.numpy(), indices.numpy())
        rh.finalize()
        corpus_scores, corpus_indices = (-rh.D).tolist(), rh.I.tolist()

        qid_list = list(self.dataset.queries.keys())
        with gzip.open(path, 'wt') as fout:
            for i in range(len(corpus_scores)):
                q_id = qid_list[i]
                scores = corpus_scores[i]
                indices = corpus_indices[i]
                for j in range(len(scores)):
                    fout.write(f'{q_id} 0 {indices[j]} {j} {scores[j]} run\n')
        return path

    def evaluate(self, path):
        qrels = pd.read_csv(r"{}/{}.csv".format(self.config.label_json_dir, self.config.dataset))
        qrels["query_id"] = qrels["query_id"].astype(str)
        qrels["doc_id"] = qrels["doc_id"].astype(str)

        encode_dataset = LSHDataset(self.config, None)
        new_2_old = list(encode_dataset.corpus.keys())
        rank_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
        for i, r in rank_results_pd.iterrows():
            rank_results_pd.at[i, "doc_id"] = new_2_old[int(r["doc_id"])]
        eval_results = ir_measures.calc_aggregate(self.config.measure, qrels, rank_results_pd)
        eval_results["num_hash"] = self.config.num_hash
        return eval_results

    def run(self):
        self._prepare_data()
        eval_results_path = self._retrieve()
        eval_results= self.evaluate(eval_results_path)
        print(eval_results)
        return eval_results



class HypersphericalLSHIndex:
    def __init__(self, config):
        self.config = config
        self.hash_bins = None
        self.hash_matrix = None
        self.all_cls = None
        self.hash_v_mean = None
        self.hash_v_std = None
        self.sum_scores = None
        self.max_scores = None


    def load_index(self):
        index_dir = os.path.join(self.config.index_dir, f'{self.config.dataset}', str(self.config.num_hash))
        self.hash_bins = torch.load("{}/{}".format(index_dir, 'hash_bins.pt'), map_location="cpu")
        self.hash_matrix = torch.load(r"{}/{}".format(index_dir, 'hash_matrix.pt'), map_location="cpu")
        self.all_cls = torch.load(r"{}/{}".format(index_dir, 'all_cls.pt'), map_location="cpu")
        self.hash_v_mean = torch.load(r"{}/{}".format(index_dir, 'hash_v_mean.pt'), map_location="cpu")
        self.hash_v_std = torch.load(r"{}/{}".format(index_dir, 'hash_v_std.pt'), map_location="cpu")

    def set_scores(self):
        self.sum_scores = torch.zeros((1, self.all_cls.shape[0],), dtype=torch.float32)
        self.max_scores = torch.zeros((self.all_cls.shape[0],), dtype=torch.float32)

    def search(self, cls_vec, embeddings, topk=30):
        cls_vec = cls_vec.to(torch.float32)
        embeddings = embeddings.to(torch.float32)
        topk_indices = self.cal_hash_value(embeddings)

        self.cls_search(cls_vec)
        self.lsh_search(embeddings, topk_indices)
        top_scores, top_ids = self.sort(topk)
        self.sum_scores.fill_(0)
        return top_scores, top_ids

    def cls_search(self, cls_vec):
        self.sum_scores = self.compute_similarity(cls_vec, self.all_cls)

    def lsh_search(self, embeddings, topk_indices):

        for i, hash_key in enumerate(topk_indices):
            hash_key = tuple(hash_key.tolist())
            token_matrix = self.hash_bins[hash_key]["dense_repr"][0]
            token_pid = self.hash_bins[hash_key]["dense_repr"][1]
            token_score = self.compute_similarity(embeddings[i], token_matrix).relu_()
            token_pid_tensor = torch.Tensor(token_pid).to(torch.int64)
            torch_scatter.scatter_max(src=token_score, index=token_pid_tensor, out=self.max_scores, dim=-1)
            self.sum_scores[0] += self.max_scores
            self.max_scores.fill_(0)


    def cal_hash_value(self,embeddings):
        normalized_embeddings = self.normalization(embeddings)
        hash_value_matrix = normalized_embeddings @ self.hash_matrix
        topk_indices = hash_value_matrix.topk(self.hash_matrix.shape[1], dim=1).indices
        return topk_indices

    def normalization(self, tensor):
        this_tensor = deepcopy(tensor)
        new_tensor = (this_tensor -  self.hash_v_mean) / self.hash_v_std
        norms = torch.norm(new_tensor, p=2, dim=1, keepdim=True)
        return new_tensor / norms

    def compute_similarity(self, q_repr, ctx_repr):
        return torch.matmul(q_repr, ctx_repr.T)


    def sort(self, topk):
        top_scores, top_ids = self.sum_scores.topk(topk, dim=1)
        return top_scores, top_ids
