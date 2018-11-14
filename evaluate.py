import sys
sys.path.append("evaluation")
import torch
from configuration import get_config
from dataloader import TestDataset
from evaluation.datasets.similarity import fetch_MEN, fetch_MTurk, fetch_RW, fetch_SimLex999, fetch_WS353
from evaluation.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from evaluation.evaluate import evaluate_similarity, evaluate_analogy
from model import *
from dataloader import *

def character_embedding(model, device, data_dir='./data', batch_size=2): # 여기
    test_loader = TestDataLoader(data_dir, batch_size)
    embeddings = []
    for words, length in test_loader:
        words = words.to(device)
        embedding = model.mlp_center(model.center_generator(words, length))
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, 0)
    embedding_map = {}
    for word, embedding in zip(test_loader.dataset.test_words, embeddings):
        embedding_map[word] = embedding
    return embedding_map

def evaluate(model, device, is_similarity, word2idx=None):
    if isinstance(model, skipgram):
        embedding = model.state_dict()['center_embedding.weight']
        w = build_embedding_map(word2idx, embedding)
    elif isinstance(model, word_embed_ng):
        w = character_embedding(model=model, device=device)
    else:
        w = build_embedding_map_pretrained(word2idx, model)
    if is_similarity:
        tasks = {
            "MTurk": fetch_MTurk(),
            "MEN": fetch_MEN(),
            "WS353": fetch_WS353(),
            "RW": fetch_RW(),
            "SIMLEX999": fetch_SimLex999()
        }
        similarity_results = {}
        # Calculate results using helper function
        for name, data in tasks.items():
            score, missing_words, total_words = evaluate_similarity(w, data.X, data.y)
            similarity_results[name] = (score, missing_words, total_words)
            print("Spearman correlation of scores on {} {}".format(name, score))
        return similarity_results
    else:
        analogy_tasks = {
            "Google": fetch_google_analogy(),
            "MSR": fetch_msr_analogy()
        }
        analogy_results = {}
        for name, data in analogy_tasks.items():
            score, missing_words, total_words = evaluate_analogy(w, data.X, data.y)
            analogy_results[name] = (score, missing_words, total_words)
            print("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))
        return analogy_results

def build_embedding_map(word2idx, embedding_matrix):
    embedding_map = {}
    for word in word2idx.keys():
        embedding_map[word] = embedding_matrix[torch.LongTensor([word2idx[word]])]
    return embedding_map

def build_embedding_map_pretrained(word2idx, embedding_matrix):
    embedding_map = {}
    for word in word2idx.keys():
        embedding_map[word] = embedding_matrix(torch.LongTensor([word2idx[word]]))
    return embedding_map
                                           

if __name__ == "__main__":
    args = get_config()
    text_loader = TextDataLoader(args.data_dir, args.dataset, args.batch_size,
                                 args.window_size, args.neg_sample_size, args.is_character, args.num_workers,
                                 args.remove_th, args.subsample_th)

    idx2word = text_loader.dataset.idx2word
    word2idx = text_loader.dataset.word2idx
    if False:
        # testing skipgram model
        model = skipgram(40000, args.embed_size)

        params = torch.load(args.log_dir + 'model_best.pt', map_location=lambda storage, loc: storage)
        print("Skipgram model loaded")
        print("Similarity test for skipgram model")
        evaluate(model=model, is_similarity=True, word2idx=word2idx)
        print("Analogy test for skipgram model")
        evaluate(model=model, is_similarity=False, word2idx=word2idx)

    if True:
        # embed:256, hidden:512, negative sampling:5
        args.embed_size = 256
        args.hidden_size = 512
        args.neg_sample_size = 5
        model = word_embed_ng(args.vocab_size, args.embed_size, args.hidden_size,
                              args.num_layer, args.dropout, args.mlp_size, args.neg_sample_size, args.bidirectional,
                              args.multigpu, args.device)
        params = torch.load(args.log_dir + 'rnn_model_best.pt', map_location=lambda storage, loc: storage)
        print("Character embedding model loaded")
        print("Similarity test for character embedding model")
        evaluate(model=model, is_similarity=True, word2idx=None)
        print("Analogy test for character embedding model")
        evaluate(model=model, is_similarity=False, word2idx=None)

