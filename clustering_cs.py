import os
import json
import random
import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from umap.umap_ import UMAP
import hdbscan
from sentence_transformers import SentenceTransformer
from collections import Counter
import logging
import subprocess
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("clustering_cs.log"),
        logging.StreamHandler(),
    ],
)


# Define the mapping of categories to broader groups
CATEGORY_GROUPS = {
    "physics": ["physics", "astro-ph", "quant-ph", "hep-th", "cond-mat", "gr-qc", "hep-ph", "nucl-th", "hep-ex", "hep-lat", "nucl-ex", "eess"],
    "math": ["math", "math-ph", "stat", "nlin"],
    "finances": ["econ", "q-fin"],
    "computer_science": ["cs"],
    "biology": ["q-bio"]
}

def update_status(message):
    """Log status updates."""
    logging.info(message)

# Updated TF-IDF keyword extraction
def extract_tfidf_keywords(documents, max_features=50, ngram_range=(1, 3), max_keywords=25):
    """
    Extracts TF-IDF keywords with n-gram support, stemming, and deduplication.
    Returns the top `max_keywords` based on TF-IDF scores.
    """
    if not documents:
        return []

    # TF-IDF vectorization with n-grams
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    keywords = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1  # Sum TF-IDF scores across all documents
    keyword_scores = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)

    # Extract and process keywords with stemming
    stemmed_keywords = {}
    for keyword, _ in keyword_scores:
        stem = stemmer.stem(keyword)
        if stem not in stemmed_keywords:
            stemmed_keywords[stem] = keyword

    # Return deduplicated, stemmed keywords limited to `max_keywords`
    deduplicated_keywords = list(stemmed_keywords.values())
    return deduplicated_keywords[:max_keywords]


# Function to map categories to broader groups
def map_to_broad_category(category_str):
    if not isinstance(category_str, str):
        return None  # Return None for invalid entries

    # Extract the main category
    categories = category_str.split()
    main_categories = [cat.split(".")[0] for cat in categories]
    main_category = main_categories[0]

    # Map the main category to a broader group
    for broad_category, category_list in CATEGORY_GROUPS.items():
        if main_category in category_list:
            return broad_category

    # If no match, return "others"
    return "others"

# Main pipeline function
def main():
    # Load the dataset
    update_status("Loading dataset...")
    df = pd.read_csv("preprocessed_data.csv")
    update_status("Loaded dataset.")
    df["broad_category"] = df["categories"].apply(map_to_broad_category)
    uncategorized_categories = (
        df.loc[df["broad_category"] == "others", "categories"]
        .apply(lambda x: x.split()[0].split(".")[0])
        .unique()
    )
    update_status("Uncategorized categories: {}".format(uncategorized_categories))

   
    # Files
    category = "computer_science"
    metrics_file = f"metrics_{category}.json"
    output_file = f"clusters_{category}.json"
    visualization_file = f"clustering_visualization_data_{category}.csv"
    param_path = "parameter_clustering.json" 

    # Load existing results if available
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            results_per_category = json.load(f)
    else:
        update_status("No parameter file found")

    params = results_per_category[category]
    
    # Clusterin
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            final_cluster_data = json.load(f)
    else:
        final_cluster_data = {}

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_summary = json.load(f)
    else:
        metrics_summary = {}

    visualization_data = []

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    try:
        update_status(f"Clustering for category: {category}")
        category_df = df[df["broad_category"] == category]
        content = category_df["cleaned_content_str"].tolist()

        # Skip if no content is available
        if not content:
            update_status(f"Skipping category {category} due to empty content.")

        embeddings = embedding_model.encode(content, show_progress_bar=True)

        params = results_per_category[category]  # Retrieve parameters for the category
        reducer = UMAP(
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            n_components=2,
            metric="cosine",
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
        )
        labels = clusterer.fit_predict(reduced_embeddings)

        vis_data = pd.DataFrame(
            {
                "category": category,
                "umap1": reduced_embeddings[:, 0],
                "umap2": reduced_embeddings[:, 1],
                "label": labels,
            }
        )
        visualization_data.append(vis_data)

        cluster_data = {"clusters": [], "noise": []}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_docs = category_df.loc[labels == cluster_id, "cleaned_content_str"].tolist()
            cluster_titles = category_df.loc[labels == cluster_id, "title"].tolist()
            top_keywords = extract_tfidf_keywords(cluster_docs)
            cluster_data["clusters"].append(
                {
                    "cluster_id": int(cluster_id),
                    "keywords": list(top_keywords),
                    "num_docs": len(cluster_docs),
                    "docs": cluster_docs,
                    "doc_titles": cluster_titles,
                }
            )

        noise_docs = category_df.loc[labels == -1, "cleaned_content_str"].tolist()
        cluster_data["noise"] = list(noise_docs)
        final_cluster_data[category] = cluster_data

        # Filter out noise for metrics
        core_embeddings = reduced_embeddings[labels != -1]
        core_labels = labels[labels != -1]

        if len(core_embeddings) > 1:  # Ensure there are enough samples
            dbi = float(davies_bouldin_score(core_embeddings, core_labels))
            silhouette = float(silhouette_score(core_embeddings, core_labels))
        else:
            dbi, silhouette = float("inf"), -1  # Default values for insufficient samples

        metrics_summary[category] = {"dbi": dbi, "silhouette": silhouette}
        update_status(f"DBI: {dbi}")
        update_status(f"silhouette: {silhouette}")

        # Save current category results incrementally
        with open(output_file, "w") as f:
            json.dump(final_cluster_data, f, indent=4)
        update_status(f"Results for category {category} saved to {output_file}")

        # Save metrics incrementally
        with open(metrics_file, "w") as f:
            json.dump(metrics_summary, f, indent=4)
        update_status(f"Metrics for category {category} saved to {metrics_file}")

    except Exception as e:
        update_status(f"Error processing category {category}: {e}")

    # Save visualization data
    if visualization_data:
        visualization_df = pd.concat(visualization_data)
        visualization_df.to_csv(visualization_file, index=False)
        update_status("Visualization data saved.")

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
