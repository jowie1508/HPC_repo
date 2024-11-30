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

# Define the mapping of categories to broader groups
CATEGORY_GROUPS = {
    "physics": ["physics", "astro-ph", "quant-ph", "hep-th", "cond-mat", "gr-qc", "hep-ph", "nucl-th", "hep-ex", "hep-lat", "nucl-ex", "eess"],
    "math": ["math", "math-ph", "stat", "nlin"],
    "finances": ["econ", "q-fin"],
    "computer_science": ["cs"],
    "biology": ["q-bio"]
}

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
    # download_dataset()
    update_status("Loading dataset...")
    # df = load_data(FILE)
    # uncomment while testing
    # df = df.sample(frac=0.001, random_state=42)

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

    df["broad_category"] = df["categories"].apply(map_to_broad_category)
    uncategorized_categories = (
        df.loc[df["broad_category"] == "others", "categories"]
        .apply(lambda x: x.split()[0].split(".")[0])
        .unique()
    )
    update_status("Uncategorized categories:")

   
   # Optimize parameters for each category
   
   # File to save progress
    results_file = "optimization_results.json"

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_per_category = json.load(f)
    else:
        results_per_category = {}
   
    categories_to_test = df["broad_category"].unique().tolist()
    update_status(f"Number of categories (total): {len(categories_to_test)}")
    
    df_optimize = df.sample(frac=0.1, random_state=42)
    df_optimize = df_optimize[df_optimize["broad_category"].isin(categories_to_test)]

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

    for category in categories_to_test:
       # Skip already processed categories
        if category in results_per_category:
            update_status(f"Skipping already processed category: {category}")
            continue
        update_status(f"Optimizing for category: {category}")
        category_df = df[df["broad_category"] == category]
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
    categories_to_run = list(results_per_category.keys())

    # Load existing results if the output file exists
    output_file = "hdbscan_cluster_results.json"
    metrics_file = "hdbscan_metrics.json"

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

    # Clustering logic
    for category in categories_to_run:
        try:
            # Skip if the category is already processed
            if category in final_cluster_data:
                update_status(f"Skipping category {category}, already processed.")
                continue

            update_status(f"Clustering for category: {category}")
            category_df = df[df["broad_category"] == category]
            content = category_df["cleaned_content_str"].tolist()

            # Skip if no content is available
            if not content:
                update_status(f"Skipping category {category} due to empty content.")
                continue

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
                dbi = davies_bouldin_score(core_embeddings, core_labels)
                silhouette = silhouette_score(core_embeddings, core_labels)
            else:
                dbi, silhouette = float("inf"), -1  # Default values for insufficient samples

            metrics_summary[category] = {"dbi": dbi, "silhouette": silhouette}

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
            continue

    # Save visualization data
    if visualization_data:
        visualization_df = pd.concat(visualization_data)
        visualization_df.to_csv("clustering_visualization_data.csv", index=False)
        update_status("Visualization data saved.")

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
