import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the features from the JSON file
def load_features(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Extract the embeddings (features) for each track_id
def extract_embeddings(data):
    track_embeddings = {}
    
    for entry in data:
        track_id = entry['track_id']
        features = entry['features']
        
        if track_id not in track_embeddings:
            track_embeddings[track_id] = []
        
        track_embeddings[track_id].append(features)
    
    return track_embeddings

# Perform dimensionality reduction (PCA or t-SNE)
def reduce_dimensions(embeddings, method='PCA', n_components=2):
    scaler = StandardScaler()
    
    all_embeddings = np.array([embedding for track in embeddings.values() for embedding in track])
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)
    
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Unsupported method: choose 'PCA' or 'TSNE'")
    
    reduced_embeddings = reducer.fit_transform(all_embeddings_scaled)
    
    return reduced_embeddings

# Visualize the embeddings in 2D
def visualize_embeddings(reduced_embeddings, track_embeddings):
    plt.figure(figsize=(10, 8))
    
    # Assign a color for each track_id
    track_colors = {track_id: np.random.rand(3,) for track_id in track_embeddings}
    
    for track_id, embeddings in track_embeddings.items():
        track_reduced = reduced_embeddings[:len(embeddings)]
        reduced_embeddings = reduced_embeddings[len(embeddings):]
        
        plt.scatter(track_reduced[:, 0], track_reduced[:, 1], label=f"Track {track_id}", 
                    color=track_colors[track_id], alpha=0.7)
    
    # plt.legend()
    plt.title('2D Visualization of Track Embeddings')
    plt.xlabel('PCA/TSNE Component 1')
    plt.ylabel('PCA/TSNE Component 2')
    plt.show()

# Plot the centroids of each track
def plot_centroids(reduced_embeddings, track_embeddings):
    plt.figure(figsize=(10, 8))
    
    # Assign a color for each track_id
    track_colors = {track_id: np.random.rand(3,) for track_id in track_embeddings}
    
    # Calculate and plot the centroid for each track
    for track_id, embeddings in track_embeddings.items():
        track_reduced = reduced_embeddings[:len(embeddings)]
        reduced_embeddings = reduced_embeddings[len(embeddings):]
        
        # Calculate the centroid of the track (mean of reduced embeddings)
        centroid = np.mean(track_reduced, axis=0)
        
        # Plot the centroid
        plt.scatter(centroid[0], centroid[1], color=track_colors[track_id], marker='x', s=200, label=f"Centroid of Track {track_id}")
    
    plt.legend()
    plt.title('Centroids of Track Embeddings')
    plt.xlabel('PCA/TSNE Component 1')
    plt.ylabel('PCA/TSNE Component 2')
    plt.show()

# KNN Matching between tracks (based on the embeddings)
def knn_match(embeddings, n_neighbors=5):
    all_embeddings = np.array([embedding for track in embeddings.values() for embedding in track])
    
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(all_embeddings)
    
    # You can then use knn to find the closest embeddings
    distances, indices = knn.kneighbors(all_embeddings)
    
    return distances, indices

# Main function to load data, reduce dimensions, and perform KNN
def main():
    file_path = 'output/s03_c011/features.json'  # Specify the path to your features.json file
    data = load_features(file_path)
    
    track_embeddings = extract_embeddings(data)
    
    # Reduce dimensions for visualization (PCA or TSNE)
    reduced_embeddings = reduce_dimensions(track_embeddings, method='TSNE', n_components=2)
    
    # Visualize the 2D embeddings
    visualize_embeddings(reduced_embeddings, track_embeddings)
    
    # Plot the centroids
    plot_centroids(reduced_embeddings, track_embeddings)
    
    # Perform KNN matching
    # distances, indices = knn_match(track_embeddings)
    
    # Print the closest matches for each embedding (this can be modified)
    print("Closest matches (distances and indices):")
    # print(distances)
    # print(indices)

if __name__ == "__main__":
    main()
