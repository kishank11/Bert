import argparse
import pandas as pd
import torch
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pyterrier as pt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Initialize PyTerrier
ensure_pyterrier_is_loaded()
if not pt.started():
    pt.init()

# Define argument parser
parser = argparse.ArgumentParser(description="BM25 + BERT Fusion Retrieval")
parser.add_argument("-i", "--input-dataset", required=True, help="Input dataset ID")
parser.add_argument("-o", "--output-dir", required=True, help="Output directory for results")
args = parser.parse_args()

# Load the dataset
dataset_id = args.input_dataset
output_dir = args.output_dir

pt_dataset = pt.get_dataset(f'irds:{dataset_id}')

# Build the BM25 index
indexer = pt.IterDictIndexer(
    output_dir + "/index",
    meta={'docno': 50, 'text': 4096},
    overwrite=True,
)
index = indexer.index(pt_dataset.get_corpus_iter())

bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# Define BERT re-ranker
class BERTReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def rerank(self, query, docs):
        inputs = self.tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1).cpu().numpy()
        return torch.sigmoid(torch.tensor(logits)).numpy()

bert_ranker = BERTReRanker()

# Re-rank with BERT
def rerank_with_bert(bm25_results, topics):
    reranked = []
    for qid, group in tqdm(bm25_results.groupby("qid"), desc="Re-ranking with BERT"):
        query = topics.loc[topics["qid"] == qid, "query"].values[0]
        docs = group["docid"].values
        doc_texts = [doc["text"] for doc in docs]
        scores = bert_ranker.rerank(query, doc_texts)
        group["score"] = scores
        reranked.append(group.sort_values(by="score", ascending=False))
    return pd.concat(reranked)

topics = pt_dataset.get_topics()
bm25_results = bm25(topics)
bert_results = rerank_with_bert(bm25_results, topics)

# Combine BM25 and BERT results
bm25_results.rename(columns={'score': 'score_bm25'}, inplace=True)
bert_results.rename(columns={'score': 'score_bert'}, inplace=True)
fusion_results = pd.merge(
    bm25_results, bert_results, on=["qid", "docid"], how="inner", suffixes=("_bm25", "_bert")
)

# Normalize and compute fusion scores
scaler = MinMaxScaler()
fusion_results["score_bm25"] = scaler.fit_transform(fusion_results[["score_bm25"]])
fusion_results["score_bert"] = scaler.fit_transform(fusion_results[["score_bert"]])
fusion_results["fusion_score"] = 0.1 * fusion_results["score_bm25"] + 0.9 * fusion_results["score_bert"]

# Rank results
fusion_results["rank"] = fusion_results["fusion_score"].rank(ascending=False)

# Save results
fusion_results[["qid", "docid", "rank", "fusion_score"]].to_csv(f"{output_dir}/run.txt", sep="\t", index=False)
