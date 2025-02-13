import tenseal as ts
import networkx as nx
import numpy as np

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.BFV,  # Change to BGV if needed
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        plain_modulus=1032193  # Ensure plain modulus is a prime > max(feature values)
    )
    context.generate_galois_keys()
    return context

def create_graph(edges_df):
    G = nx.Graph()
    users = edges_df["User"].unique()
    songs = edges_df["Song"].unique()
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(songs, bipartite=1)
    G.add_edges_from(edges_df.to_records(index=False))
    return G

def scale_features(features, scale_factor=1000):
    return {key: (np.array(value) * scale_factor).astype(int).tolist() for key, value in features.items()}

def encrypt_features(context, features):
    return {key: ts.bfv_vector(context, value) for key, value in features.items()}

def modular_inverse(value, modulus):
    """Compute modular inverse using Fermat's Little Theorem (modular arithmetic trick)."""
    return pow(value, -1, modulus) if value != 0 else 1  # Avoid division by zero

def gnn_layer(graph, user_features, song_features, context):
    plain_modulus = 1032193  # Must match context's plain modulus
    updated_user_features = {}
    updated_song_features = {}

    for user in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]:
        aggregated = ts.bfv_vector(context, [0] * len(user_features[user].decrypt()))
        for neighbor in graph.neighbors(user):
            aggregated += song_features[neighbor]
        neighbor_count = len(list(graph.neighbors(user)))

        if neighbor_count > 0:
            inv_neighbor_count = modular_inverse(neighbor_count, plain_modulus)
            aggregated *= inv_neighbor_count  # Multiply by modular inverse instead of dividing
        updated_user_features[user] = aggregated
    
    for song in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]:
        aggregated = ts.bfv_vector(context, [0] * len(song_features[song].decrypt()))
        for neighbor in graph.neighbors(song):
            aggregated += user_features[neighbor]
        neighbor_count = len(list(graph.neighbors(song)))

        if neighbor_count > 0:
            inv_neighbor_count = modular_inverse(neighbor_count, plain_modulus)
            aggregated *= inv_neighbor_count  # Multiply by modular inverse instead of dividing
        updated_song_features[song] = aggregated

    return updated_user_features, updated_song_features

def recommend(graph, encrypted_user_features, encrypted_song_features):
    recommendations = {}
    for user in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]:
        scores = {}
        for song in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]:
            score = encrypted_user_features[user].dot(encrypted_song_features[song]).decrypt()[0]
            scores[song] = score
        recommendations[user] = scores
    return recommendations