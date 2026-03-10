"""SentenceTransformers + KMeans clustering."""
# hardshell/analysis/clustering.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def compute_semantic_clusters(df: pd.DataFrame, text_column: str, n_clusters: int = 5):
    """
    Uses Sentence-Transformers to embed Agent A's internal messages and 
    clusters them to identify behavioral archetypes (e.g., 'Innocent Summary' vs 'Weaponized Task').
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Embedding {len(df)} messages for semantic clustering...")
    embeddings = model.encode(df[text_column].tolist(), show_progress_bar=True)
    
    # Perform KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['cluster_id'] = kmeans.fit_predict(embeddings)
    
    # Optional: Reduce dimensionality for 2D visualization later
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(embeddings)
    df['pca_1'] = pca_features[:, 0]
    df['pca_2'] = pca_features[:, 1]
    
    return df