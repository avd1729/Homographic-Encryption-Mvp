from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from utils.util import create_context, create_graph, encrypt_features, gnn_layer, recommend


load_dotenv()

app = Flask(__name__)
CORS(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/recommend", methods=["POST"])
@login_required
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
