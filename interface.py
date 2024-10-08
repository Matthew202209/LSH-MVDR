import argparse

import pandas as pd
from ir_measures import *

from lsh_index import LSHIndex
from lsh_retrieve import LSHRetrieve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default=r"./data/corpus")
    parser.add_argument("--dataset", type=str, default=r"nfcorpus")
    # parser.add_argument("--data_dir", type=str, default=r"./data/corpus")
    # parser.add_argument("--ctx_embeddings_dir", type=str, default=r"./cache/LSH")
    parser.add_argument("--queries_dir", type=str, default=r"./data/query")
    parser.add_argument("--index_dir", type=str, default=r"./index/Hamming")
    parser.add_argument("--hashing", type=str, default="hamming")
    parser.add_argument("--transformer_model_dir", type=str, default=r"./checkpoints/bert-base-uncased")


    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tok_projection_dim", type=int, default=32)
    parser.add_argument("--cls_projection_dim", type=int, default=128)

    parser.add_argument("--check_point_path", type=str, default=r"./checkpoints/citadel.ckpt")
    parser.add_argument("--device", type=str, default=r"cuda:0")
    parser.add_argument("--encode_device", type=str, default=r"cuda:0")

    parser.add_argument("--add_context_id", type=bool, default=False)
    parser.add_argument("--num_hash", type=int, default=3)
    parser.add_argument("--weight_threshold", type=float, default=0.5)
    parser.add_argument("--prune_weight", type=float, default=0.8)

    parser.add_argument("--cls_dim", type=float, default=128)
    parser.add_argument("--dim", type=float, default=32)
    parser.add_argument("--sub_vec_dim", type=float, default=4)
    parser.add_argument("--num_centroids", type=float, default=16)
    parser.add_argument("--iter", type=float, default=5)
    parser.add_argument("--threshold", type=int, default=1000)
    parser.add_argument("--hamming_threshold", type=int, default=2)

    parser.add_argument("--query_json_dir", type=str, default=r'./data/query')
    parser.add_argument("--label_json_dir", type=str, default=r'./data/label')
    parser.add_argument("--results_save_to", type=str, default=r'./results_hamming')
    parser.add_argument("--portion", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--measure", type=list, default=[nDCG @ 10, RR @ 10, Success @ 10])
    args = parser.parse_args()

    # for num_hash in [4]:
    #     args.num_hash = num_hash
    #     lsh_indexer = LSHIndex(args)
    #     lsh_indexer.run()

    eval_list = []
    for num_hash in [4]:
        args.num_hash = num_hash
        h = LSHRetrieve(args)
        h.setup()
        eval_results = h.run()
        eval_list.append(eval_results)
        eval_df = pd.DataFrame(eval_list)
    eval_df.to_csv(r"{}/hypersphericalLSH/{}/eval_results/eval.csv".format(args.results_save_to, args.dataset), index=False)
