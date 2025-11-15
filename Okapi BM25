# Option 2 - Okapi BM25 Implementation
# Parameters: k1=1.6, b=0.75

import pandas as pd
from math import log

# ---------------------------
# Example input data
# ---------------------------

terms = [f"t{i}" for i in range(1, 11)]

data = {
    "t1": [3, 2, 1, 0, 2, 1],
    "t2": [0, 1, 0, 0, 0, 0],
    "t3": [5, 2, 0, 1, 0, 3],
    "t4": [0, 0, 1, 0, 0, 0],
    "t5": [2, 1, 0, 0, 1, 0],
    "t6": [0, 0, 0, 0, 0, 1],
    "t7": [1, 0, 3, 1, 2, 0],
    "t8": [0, 1, 0, 2, 0, 0],
    "t9": [4, 3, 2, 1, 0, 2],
    "t10": [0, 0, 0, 0, 1, 0]
}

docs_df = pd.DataFrame(data, index=[f"doc{i}" for i in range(1, 7)])
docs_df["doc_len"] = docs_df.sum(axis=1)

# Query terms
query_terms = ["t1", "t3", "t9", "t2"]

# ---------------------------
# BM25 Parameters
# ---------------------------

k1 = 1.6
b = 0.75
N = len(docs_df)

# ---------------------------
# Compute DF and IDF
# ---------------------------

df_counts = (docs_df[terms] > 0).sum(axis=0)
idf = {}

for t in terms:
    n_t = df_counts[t]
    idf[t] = log((N - n_t + 0.5) / (n_t + 0.5) + 1)

# ---------------------------
# Compute BM25 contributions
# ---------------------------

bm25_contrib = pd.DataFrame(0.0, index=docs_df.index, columns=query_terms)
avgdl = docs_df["doc_len"].mean()

for doc in docs_df.index:
    dl = docs_df.loc[doc, "doc_len"]

    for t in query_terms:
        tf = docs_df.loc[doc, t]

        if tf == 0:
            score = 0
        else:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf[t] * (numerator / denominator)

        bm25_contrib.loc[doc, t] = round(score, 6)

# ---------------------------
# Total BM25 score per document
# ---------------------------

bm25_contrib["BM25_score"] = bm25_contrib.sum(axis=1)
bm25_output = bm25_contrib[["BM25_score"] + query_terms]

# ---------------------------
# Ranking
# ---------------------------

ranking = bm25_output.sort_values("BM25_score", ascending=False)

# ---------------------------
# Print results
# ---------------------------

print("=== Input DataFrame ===")
print(docs_df)

print("\n=== BM25 Output ===")
print(bm25_output)

print("\n=== Ranking ===")
print(ranking)
