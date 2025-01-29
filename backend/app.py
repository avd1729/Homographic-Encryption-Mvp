from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import networkx as nx
import tenseal as ts

app = Flask(__name__)
CORS(app)

# GNN logic from previous code
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
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

def encrypt_features(context, features):
    return {key: ts.ckks_vector(context, value) for key, value in features.items()}

def gnn_layer(graph, user_features, song_features, context):
    updated_user_features = {}
    updated_song_features = {}
    
    for user in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]:
        aggregated = ts.ckks_vector(context, [0] * len(user_features[user].decrypt()))
        for neighbor in graph.neighbors(user):
            aggregated += song_features[neighbor]
        if len(list(graph.neighbors(user))) > 0:
            aggregated *= 1 / len(list(graph.neighbors(user))) 
        updated_user_features[user] = aggregated
    
    for song in [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]:
        aggregated = ts.ckks_vector(context, [0] * len(song_features[song].decrypt()))
        for neighbor in graph.neighbors(song):
            aggregated += user_features[neighbor]
        if len(list(graph.neighbors(user)))  > 0:
            aggregated *= 1 / len(list(graph.neighbors(user))) 
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

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    # Read uploaded files
    edges_file = request.files["edges"]
    user_features_file = request.files["user_features"]
    song_features_file = request.files["song_features"]

    edges_df = pd.read_csv(edges_file)
    user_features_df = pd.read_csv(user_features_file)
    song_features_df = pd.read_csv(song_features_file)

    # Prepare data
    graph = create_graph(edges_df)
    user_features = user_features_df.set_index("User").T.to_dict(orient="list")
    song_features = song_features_df.set_index("Song").T.to_dict(orient="list")

    # Encryption and recommendations
    context = create_context()
    encrypted_user_features = encrypt_features(context, user_features)
    encrypted_song_features = encrypt_features(context, song_features)
    updated_user_features, updated_song_features = gnn_layer(graph, encrypted_user_features, encrypted_song_features, context)
    recommendations = recommend(graph, updated_user_features, updated_song_features)

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
