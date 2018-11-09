import sys
sys.path.append("evaluation")
import torch
# from configuration import get_config
from model import *
from dataloader import TextDataLoader
from evaluation.embedding import Embedding
from evaluation.datasets.similarity import fetch_MEN, fetch_MTurk, fetch_RW, fetch_SimLex999, fetch_TR9856, fetch_WS353
from evaluation.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from evaluation.evaluate import evaluate_similarity, evaluate_analogy

def evaluate(params, word2idx, is_similarity):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # text_loader = TextDataLoader(args.data_dir, args.dataset, args.batch_size,
    #                              args.window_size, args.neg_sample_size, args.is_character, args.num_worker)
    #
    # idx2word = text_loader.dataset.idx2word
    # word2idx = text_loader.dataset.word2idx
    #
    # params = torch.load(args.log_dir + 'model_best.pt', map_location=lambda storage, loc: storage)
    # print("Model loaded")

    embedding = params['center_embedding.weight']
    w = build_embedding_map(word2idx, embedding)
    if is_similarity:
        tasks = {
            "MTurk": fetch_MTurk(),
            "MEN": fetch_MEN(),
            "WS353": fetch_WS353(),
            "RW": fetch_RW(),
            "SIMLEX999": fetch_SimLex999()
        }
        # Calculate results using helper function
        for name, data in tasks.items():
            print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(w, data.X, data.y)))
    else:
        analogy_tasks = {
            "Google": fetch_google_analogy(),
            "MSR": fetch_msr_analogy()
        }
        analogy_results = {}
        for name, data in analogy_tasks.items():
            analogy_results[name] = evaluate_analogy(w, data.X, data.y)
            print("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))

def build_embedding_map(word2idx, embedding_matrix):
    embedding_map = {}
    for word in word2idx.keys():
        embedding_map[word] = embedding_matrix[torch.LongTensor([word2idx[word]])]
    return embedding_map


# if __name__ == "__main__":
    # evaluate(get_config(), False)
