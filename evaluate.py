import sys
sys.path.append("evaluation")
import torch
from configuration import get_config
from model import *
from dataloader import TextDataLoader
from evaluation.datasets.similarity import fetch_MEN, fetch_MTurk, fetch_RW, fetch_SimLex999, fetch_TR9856, fetch_WS353
from evaluation.analogy import *
from evaluation.evaluate import evaluate_similarity

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_loader = TextDataLoader(args.data_dir, args.dataset, args.batch_size,
                                 args.window_size, args.neg_sample_size, args.is_character, args.num_worker)

    idx2word = text_loader.dataset.idx2word
    word2idx = text_loader.dataset.word2idx

    # load trained model
    # args.model_name = 'sgns'
    # if args.model_name == 'sgns':
    #     model = skipgram(len(text_loader.dataset.vocabs), args.embed_size)
    # model = model.to(device)
    # model.load_state_dict(torch.load(args.log_dir + 'model_best.pt'))
    params = torch.load(args.log_dir + 'model_best.pt', map_location=lambda storage, loc: storage)
    print("Model loaded")

    # embedding = model.center_embedding
    embedding = params['center_embedding.weight']
    w_skipgram = build_embedding_map(word2idx, embedding)

    tasks = {
        "MTurk": fetch_MTurk(),
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "RW": fetch_RW(),
        "SIMLEX999": fetch_SimLex999()
    }

    # Calculate results using helper function
    for name, data in tasks.items():
        print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(w_skipgram, data.X, data.y)))


def build_embedding_map(word2idx, embedding_matrix):
    embedding_map = {}
    for word in word2idx.keys():
        embedding_map[word] = embedding_matrix[torch.LongTensor([word2idx[word]])]
    return embedding_map

if __name__ == "__main__":
    evaluate(get_config())

