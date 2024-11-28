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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("clustering.log"),
        logging.StreamHandler(),
    ],
)


# Global constants
FILE = "arxiv-metadata-oai-snapshot.json"

def download_dataset():
    try:
        # Download the dataset using Kaggle CLI
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "Cornell-University/arxiv"],
            check=True
        )
        # Unzip the downloaded file
        subprocess.run(
            ["unzip", "-o", "arxiv.zip"],
            check=True
        )
        print("Dataset downloaded and extracted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def update_status(message):
    """Log status updates."""
    logging.info(message)

# Function to load and preprocess the dataset
def load_data(file_path):
    dataframe = {
        "title": [],
        "year": [],
        "authors": [],
        "categories": [],
        "journal-ref": [],
        "abstract": [],
    }
    with open(file_path) as f:
        for line in f:
            paper = json.loads(line)
            try:
                date = int(paper["update_date"].split("-")[0])
                if date > 2019:
                    dataframe["title"].append(paper["title"])
                    dataframe["year"].append(date)
                    dataframe["authors"].append(paper["authors"])
                    dataframe["categories"].append(paper["categories"])
                    dataframe["journal-ref"].append(paper["journal-ref"])
                    dataframe["abstract"].append(paper["abstract"])
            except:
                pass
    df = pd.DataFrame(dataframe)
    return df

# Preprocessing functions
def preprocess_text_fast(text, stop_words):
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def remove_least_used_fast(texts, N):
    all_words = [word for sublist in texts for word in sublist]
    word_counts = Counter(all_words)
    frequent_words = {word for word, count in word_counts.items() if count >= N}
    return [[word for word in sublist if word in frequent_words] for sublist in texts]

def extract_main_category(category_str):
    if not isinstance(category_str, str):
        return None
    categories = category_str.split()
    main_categories = [cat.split(".")[0] for cat in categories]
    return main_categories[0]

# Function to extract TF-IDF keywords
def extract_tfidf_keywords(documents, max_features=10):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(documents)
    top_keywords = vectorizer.get_feature_names_out()
    return top_keywords

# Main pipeline function
def main():
    # Load the dataset
   #  d# # ownload_dataset()
    update_status("Loading dataset...")
   # df = load_data(FILE)
    # uncomment while testing
    #df = df.sample(frac=0.001, random_state=42)

    # Preprocess the dataset
    #stop_words = set(stopwords.words("english"))
    #df["content"] = df["title"] + " " + df["abstract"]
    #df["cleaned_content"] = [
    #    preprocess_text_fast(text, stop_words) for text in df["content"]
    #]
    #df["main_category"] = df["categories"].apply(extract_main_category)
    #df["cleaned_content"] = remove_least_used_fast(df["cleaned_content"], N=10)
    #df["cleaned_content_str"] = df["cleaned_content"].apply(lambda x: " ".join(x))
    df = pd.read_csv("preprocessed_data.csv")
    update_status("Loaded dataset...")
   
   # Optimize parameters for each category
   
   # File to save progress
    results_file = "optimization_results.json"

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_per_category = json.load(f)
    else:
        results_per_category = {}
   
    categories_to_test = df["main_category"].unique().tolist()
    update_status(f"Number of categories (total): {len(categories_to_test)}")
    results_per_category = {}
    df_optimize = df.sample(frac=0.1, random_state=42)
    df_optimize = df_optimize[df_optimize["main_category"].isin(categories_to_test)]

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    param_space = {
        "n_neighbors": [5, 10, 15, 30, 50],
        "min_dist": [0.01, 0.1, 0.3, 0.5],
        "min_cluster_size": [5, 10, 20, 50],
        "min_samples": [1, 5, 10],
    }
    random_combinations = [
        {
            "n_neighbors": random.choice(param_space["n_neighbors"]),
            "min_dist": random.choice(param_space["min_dist"]),
            "min_cluster_size": random.choice(param_space["min_cluster_size"]),
            "min_samples": random.choice(param_space["min_samples"]),
        }
        for _ in range(10)
    ]
    # Reorder categories_to_test to prioritize 'adap-org'
    failing_category = "adap-org"
    if failing_category in categories_to_test:
        categories_to_test.remove(failing_category)
    categories_to_test = [failing_category] + categories_to_test
    for category in categories_to_test:
       # Skip already processed categories
        if category in results_per_category:
            update_status(f"Skipping already processed category: {category}")
            continue
        update_status(f"Optimizing for category: {category}")
        category_df = df[df["main_category"] == category]
        content = category_df["cleaned_content_str"].tolist()
        embeddings = embedding_model.encode(content, show_progress_bar=True)

        category_results = []
        for params in random_combinations:
            try:
                reducer = UMAP(
                    n_neighbors=params["n_neighbors"],
                    min_dist=params["min_dist"],
                    n_components=5,
                    metric="cosine",
                )
                reduced_embeddings = reducer.fit_transform(embeddings)

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=params["min_cluster_size"],
                    min_samples=params["min_samples"],
                )
                labels = clusterer.fit_predict(reduced_embeddings)

                if len(set(labels)) > 1:
                    dbi = davies_bouldin_score(reduced_embeddings, labels)
                else:
                    dbi = float("inf")
                category_results.append({**params, "dbi": dbi})
            except Exception as e:
                update_status(f"Error during optimization for category '{category}' with params {params}: {e}")
                continue
        if category_results:    
            best_params = min(category_results, key=lambda x: x["dbi"])
            results_per_category[category] = best_params
            # Save progress after each category
            with open(results_file, "w") as f:
                json.dump(results_per_category, f, indent=4)

            update_status(f"Completed optimization for category: {category}")
        else:
            update_status(f"No valid results for category '{category}'. Skipping.")
    
    update_status("Parameter optimization completed.")

    # Final clustering and saving results
    final_cluster_data = {}
    visualization_data = []
    metrics_summary = {}
    for category, params in results_per_category.items():
        update_status(f"Clustering for category: {category}")
        category_df = df[df["main_category"] == category]
        content = category_df["cleaned_content_str"].tolist()
        embeddings = embedding_model.encode(content, show_progress_bar=True)

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
            {"category": category, "umap1": reduced_embeddings[:, 0], "umap2": reduced_embeddings[:, 1], "label": labels}
        )
        visualization_data.append(vis_data)

        cluster_data = {"clusters": [], "noise": []}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_docs = category_df.loc[labels == cluster_id, "cleaned_content_str"].tolist()
            top_keywords = extract_tfidf_keywords(cluster_docs)
            cluster_data["clusters"].append({
                "cluster_id": int(cluster_id),
                "keywords": list(top_keywords),
                "num_docs": len(cluster_docs),
                "docs": cluster_docs
            })

        noise_docs = category_df.loc[labels == -1, "cleaned_content_str"].tolist()
        cluster_data["noise"] = list(noise_docs)
        final_cluster_data[category] = cluster_data

        core_embeddings = reduced_embeddings[labels != -1]
        core_labels = labels[labels != -1]
        dbi = davies_bouldin_score(core_embeddings, core_labels)
        silhouette = silhouette_score(core_embeddings, core_labels)
        metrics_summary[category] = {"dbi": dbi, "silhouette": silhouette}

    # Save clustering data to JSON
    output_file = "hdbscan_cluster_results.json"
    with open(output_file, "w") as f:
        json.dump(final_cluster_data, f, indent=4)
    update_status(f"Cluster data saved to {output_file}")

    # Save visualization data
    visualization_df = pd.concat(visualization_data)
    visualization_df.to_csv("clustering_visualization_data.csv", index=False)
    update_status("Visualization data saved.")

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
