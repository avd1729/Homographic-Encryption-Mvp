import tenseal as ts
import pandas as pd
import networkx as nx

# Step 1: Create Encryption Context
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,  # Use CKKS for encrypted floating-point arithmetic
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40  # Precision scale
    context.generate_galois_keys()  # For rotations and encrypted aggregations
    return context

# Step 2: Load Graph Data from CSV
def create_graph(edges_csv):
    # Load edges from CSV
    edges_df = pd.read_csv(edges_csv)
    
    # Create a bipartite graph
    G = nx.Graph()
    users = edges_df["User"].unique()
    songs = edges_df["Song"].unique()
    
    G.add_nodes_from(users, bipartite=0)  # User nodes
    G.add_nodes_from(songs, bipartite=1)  # Song nodes
    G.add_edges_from(edges_df.to_records(index=False))  # Add edges
    return G

# Step 3: Load Node Features from Separate CSVs
def create_node_features(user_features_csv, song_features_csv):
    # Load user features
    user_features_df = pd.read_csv(user_features_csv)
    user_features = {
        row["User"]: row.iloc[1:].values.tolist()
        for _, row in user_features_df.iterrows()
    }
    
    # Load song features
    song_features_df = pd.read_csv(song_features_csv)
    song_features = {
        row["Song"]: row.iloc[1:].values.tolist()
        for _, row in song_features_df.iterrows()
    }
    
    return user_features, song_features

# Step 4: Encrypt Features
def encrypt_features(context, user_features, song_features):
    encrypted_user_features = {u: ts.ckks_vector(context, f) for u, f in user_features.items()}
    encrypted_song_features = {s: ts.ckks_vector(context, f) for s, f in song_features.items()}
    return encrypted_user_features, encrypted_song_features

# Step 5: GNN Layer (Message Passing and Aggregation)
def gnn_layer(graph, user_features, song_features, context):
    updated_user_features = {}
    updated_song_features = {}
    
    # Update user features based on connected songs
    for user in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]:
        aggregated = ts.ckks_vector(context, [0] * len(user_features[user].decrypt()))
        neighbors = list(graph.neighbors(user))  # Songs connected to the user
        for neighbor in neighbors:
            aggregated += song_features[neighbor]
        if len(neighbors) > 0:
            aggregated *= (1 / len(neighbors))  # Multiply by reciprocal instead of division
        updated_user_features[user] = aggregated
    
    # Update song features based on connected users
    for song in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]:
        aggregated = ts.ckks_vector(context, [0] * len(song_features[song].decrypt()))
        neighbors = list(graph.neighbors(song))  # Users connected to the song
        for neighbor in neighbors:
            aggregated += user_features[neighbor]
        if len(neighbors) > 0:
            aggregated *= (1 / len(neighbors))  # Multiply by reciprocal instead of division
        updated_song_features[song] = aggregated

    return updated_user_features, updated_song_features

# Step 6: Recommend songs for each user
def recommend(graph, encrypted_user_features, encrypted_song_features):
    recommendations = {}
    for user in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]:
        scores = {}
        for song in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]:
            # Compute similarity as dot product
            score = encrypted_user_features[user].dot(encrypted_song_features[song])
            
            # Decrypt the score and extract the scalar value
            decrypted_score = score.decrypt()
            
            # If the result is a list with one value, extract the scalar
            scores[song] = decrypted_score[0] if isinstance(decrypted_score, list) else decrypted_score
        
        recommendations[user] = scores
    return recommendations

# Main Workflow
if __name__ == "__main__":
    # Step 1: Create encryption context
    context = create_context()

    # Step 2: Load graph from CSV
    edges_csv = "edges.csv"  # Replace with your actual file path
    graph = create_graph(edges_csv)

    # Step 3: Load node features from separate CSVs
    user_features_csv = "user_features.csv"  # Replace with actual file path
    song_features_csv = "song_features.csv"  # Replace with actual file path
    user_features, song_features = create_node_features(user_features_csv, song_features_csv)

    # Step 4: Encrypt node features
    encrypted_user_features, encrypted_song_features = encrypt_features(context, user_features, song_features)

    # Step 5: Apply GNN layer
    print("Applying GNN layer on encrypted data...")
    updated_user_features, updated_song_features = gnn_layer(graph, encrypted_user_features, encrypted_song_features, context)

    # Step 6: Recommend songs for each user
    print("Generating recommendations...")
    recommendations = recommend(graph, updated_user_features, updated_song_features)

    # Step 7: Display recommendations
    for user, scores in recommendations.items():
        print(f"\nRecommendations for {user}:")
        for song, score in scores.items():
            print(f"  {song}: {score:.4f}")
