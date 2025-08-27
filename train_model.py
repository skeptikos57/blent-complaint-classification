"""Script d'entraînement du modèle de classification de plaintes pour une compagnie d'assurance.

Ce script :
1. Charge un dataset de plaintes clients avec leurs catégories de produits
2. Transforme les textes en vecteurs numériques avec Word2Vec
3. Entraîne un réseau de neurones (CNN + LSTM) pour prédire la catégorie
4. Sauvegarde les modèles entraînés pour une utilisation ultérieure
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement avec override=True pour forcer le rechargement
load_dotenv(override=True)

# Configuration pour réduire les messages d'information de TensorFlow
# IMPORTANT : Ces lignes DOIVENT être avant l'import de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 3 = Afficher seulement les erreurs (pas les warnings/infos)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Désactive les optimisations oneDNN (réduit les messages)

# Imports des bibliothèques nécessaires
import numpy as np  # Pour les calculs numériques et les tableaux
import pandas as pd  # Pour manipuler les données sous forme de tableaux
import matplotlib.pyplot as plt  # Pour créer des graphiques (pas utilisé actuellement)
import tensorflow as tf  # Framework de deep learning pour créer le réseau de neurones

# Imports des modules spécifiques
from dotenv import load_dotenv  # Pour charger les variables d'environnement depuis le fichier .env
from gensim.models import Word2Vec  # Pour créer des embeddings de mots (Word2Vec)
from keras import layers  # Pour construire les couches du réseau de neurones
from spacy.tokenizer import Tokenizer  # Pour découper le texte (pas utilisé actuellement)
from spacy.lang.en import English  # Pour le traitement du texte en anglais
from sklearn.model_selection import train_test_split  # Pour diviser les données en train/test
from sklearn.utils.class_weight import compute_class_weight  # Pour gérer le déséquilibre des classes
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint  # Pour le monitoring et la sauvegarde
import json  # Pour lire le fichier d'analyse des classes

# Configuration des hyperparamètres (avec valeurs par défaut si non définies)
W2V_SIZE = int(os.getenv("W2V_SIZE", 100))  # Dimension des vecteurs Word2Vec
W2V_MIN_COUNT = int(os.getenv("W2V_MIN_COUNT", 3))  # Fréquence minimale des mots
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))  # Nombre max de mots par commentaire
NB_COMMENT = int(os.getenv("NB_COMMENT", 10000))  # Nombre de commentaires à utiliser

def tokenize_corpus(texts):
    """Transforme une liste de textes de plaintes en liste de mots (tokens).
    
    Cette fonction prépare le texte pour Word2Vec :
    - Met tout en minuscules pour uniformiser
    - Découpe chaque texte en mots individuels
    - Enlève la ponctuation (virgules, points, etc.)
    
    Args:
        texts: Liste de textes de plaintes (textes bruts)
    
    Returns:
        Liste de listes de mots. Chaque sous-liste = les mots d'une plainte
        Exemple: [["credit", "report", "error"], ["mortgage", "payment", "issue"]]
    """
    tokens = []  # Liste qui contiendra tous les textes tokenisés
    nlp = English()  # Créer un objet pour analyser l'anglais
    
    for text in texts:
        if pd.isna(text):  # Gérer les valeurs manquantes
            text = ""
        text = str(text).lower()  # Convertir en minuscules
        # nlp(text) découpe le texte et analyse chaque mot
        # x.text récupère le mot, x.is_punct vérifie si c'est de la ponctuation
        tokens.append([x.text for x in nlp(text) if not x.is_punct])
    return tokens

def fit_word2vec(tokens):
    """Entraîne un modèle Word2Vec pour transformer les mots en vecteurs numériques.
    
    Word2Vec apprend à représenter chaque mot comme un vecteur de nombres.
    Les mots similaires auront des vecteurs proches (ex: "excellent" et "super").
    
    Args:
        tokens: Liste de listes de mots provenant de tokenize_corpus
    
    Returns:
        Modèle Word2Vec entraîné
    """
    # Création et entraînement du modèle Word2Vec
    w2v = Word2Vec(
        sentences=tokens,  # Les commentaires tokenisés
        vector_size=W2V_SIZE,  # Taille des vecteurs (100 dimensions par défaut)
        min_count=W2V_MIN_COUNT,  # Ignore les mots qui apparaissent moins de 3 fois
        window=5,  # Contexte : regarde 5 mots avant et après pour apprendre
        workers=2  # Utilise 2 threads pour accélérer l'entraînement
    )
    
    # Sauvegarder le modèle pour pouvoir le réutiliser plus tard
    w2v.wv.save("models/w2v.wv")
    
    # Afficher des statistiques sur le vocabulaire appris
    print(f"\n📊 Informations sur le modèle Word2Vec:")
    print(f"- Taille du vocabulaire: {len(w2v.wv)} mots uniques")
    
    return w2v

def text2vec(tokens, w2v):
    """Convertit chaque texte de plainte en une matrice de vecteurs Word2Vec.
    
    Transforme les mots en nombres pour que le réseau de neurones puisse les comprendre.
    Chaque commentaire devient une matrice de taille fixe (100 x 64).
    
    Args:
        tokens: Liste de listes de mots
        w2v: Modèle Word2Vec entraîné
    
    Returns:
        Array numpy 3D de forme (nb_commentaires, 100, 64)
        - Dimension 1: Chaque commentaire
        - Dimension 2: Les 100 dimensions du vecteur Word2Vec
        - Dimension 3: Les 64 mots maximum par commentaire
    """
    # Pré-allouer le tableau numpy pour économiser la mémoire
    # Utilisation de float32 au lieu de float64 pour diviser la mémoire par 2
    n_samples = len(tokens)
    X = np.zeros((n_samples, W2V_SIZE, MAX_LENGTH), dtype=np.float32)
    
    # Pour chaque commentaire tokenisé
    for i, row_tokens in enumerate(tokens):
        # Remplir la matrice avec les vecteurs des mots (max 64 mots)
        for j in range(min(MAX_LENGTH, len(row_tokens))):
            # Vérifier si le mot existe dans le vocabulaire
            if row_tokens[j] in w2v.wv:
                # Mettre le vecteur du mot j dans la colonne j
                X[i, :, j] = w2v.wv[row_tokens[j]]
            # Si le mot n'est pas dans le vocabulaire, on laisse des zéros
        
        # Afficher la progression tous les 1000 textes
        if i % 1000 == 0:
            print(f"{(i * 100) / len(tokens):.1f}% effectué.")
    
    return X

def prepare_labels(data, class_mapping):
    """Prépare les étiquettes (labels) pour l'entraînement.
    
    Transforme les catégories de produits en indices numériques puis en one-hot encoding.
    
    Args:
        data: DataFrame avec colonne 'product'
        class_mapping: Dictionnaire mappant les noms de classes aux indices
    
    Returns:
        Array one-hot encoded (nb_classes colonnes : une seule vaut 1, les autres 0)
    """
    # Mapper les catégories de produits aux indices numériques
    y_indices = data['product'].map(class_mapping)
    
    # Transformer en one-hot encoding
    num_classes = len(class_mapping)
    y = np.zeros((len(y_indices), num_classes))
    for i, idx in enumerate(y_indices):
        if not pd.isna(idx):
            y[i, int(idx)] = 1
    
    return y

def create_rnn(num_classes):
    """Crée l'architecture du réseau de neurones pour la classification.
    
    Architecture hybride CNN + LSTM :
    - CNN (Convolution) : Détecte des patterns locaux dans le texte
    - LSTM : Comprend les séquences et le contexte
    - Dense : Couches finales pour la classification
    
    Returns:
        Modèle Keras non compilé
    """
    return tf.keras.Sequential([
        # Couche d'entrée : attend des matrices de taille (100, 64)
        layers.Input(shape=(W2V_SIZE, MAX_LENGTH)),
        
        # Convolution 1D : détecte des patterns de 3 mots consécutifs
        # 64 filtres pour mieux capturer la diversité des plaintes
        layers.Convolution1D(64, kernel_size=3, padding='same', activation='relu'),
        
        # MaxPooling : réduit la taille de moitié en gardant les valeurs importantes
        layers.MaxPool1D(2),
        
        # Convolution supplémentaire pour capturer plus de patterns
        layers.Convolution1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool1D(2),
        
        # LSTM : réseau récurrent qui comprend les séquences
        # 256 neurones pour gérer la complexité accrue
        layers.LSTM(256, activation="tanh", return_sequences=True),
        
        # Dropout : désactive aléatoirement 20% des neurones (évite le surapprentissage)
        layers.Dropout(0.2),
        
        # Deuxième couche LSTM
        layers.LSTM(128, activation="tanh"),
        
        # Dropout supplémentaire
        layers.Dropout(0.2),
        
        # Dense : couche classique avec 128 neurones
        layers.Dense(128, activation="relu"),
        
        # Dropout avant la sortie
        layers.Dropout(0.1),
        
        # Couche de sortie : autant de neurones que de catégories
        # Softmax garantit que la somme des probabilités = 1
        layers.Dense(num_classes, activation="softmax")
    ])
            
def load_class_mapping():
    """Charge le mapping des classes depuis le fichier d'analyse.
    
    Returns:
        tuple: (class_mapping dict, num_classes int, class_weights dict)
    """
    with open('data/prepared/analyse_result.json', 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    # Créer le mapping nom de classe -> indice
    class_mapping = {}
    class_weights = {}
    for i, classe in enumerate(analysis['classes']):
        class_mapping[classe['nom']] = i
        # Utiliser l'inverse du pourcentage comme poids pour compenser le déséquilibre
        # Gérer les classes avec pourcentage nul ou très faible
        pourcentage = max(classe['pourcentage'] / 100.0, 0.0001)  # Minimum 0.01%
        class_weights[i] = 1.0 / pourcentage
    
    # Sauvegarder le mapping pour l'utiliser dans predict.py
    with open('models/class_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({
            'class_mapping': class_mapping,
            'index_to_class': {str(v): k for k, v in class_mapping.items()},
            'num_classes': len(class_mapping)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📈 Classes identifiées: {len(class_mapping)} catégories")
    print(f"Catégories les plus fréquentes:")
    for classe in analysis['classes'][:5]:
        print(f"  - {classe['nom'][:50]}: {classe['pourcentage']:.2f}%")
    
    return class_mapping, len(class_mapping), class_weights

def main():
    """Fonction principale qui orchestre tout l'entraînement."""
    
    # ÉTAPE 0 : Charger le mapping des classes
    class_mapping, num_classes, class_weights = load_class_mapping()
    
    # ÉTAPE 1 : Chargement des données
    # Lit le fichier CSV préparé par prepare_data.py
    data_path = 'data/prepared/complaints_processed.csv'
    if not os.path.exists(data_path):
        print(f"Erreur: Le fichier {data_path} n'existe pas.")
        print("Veuillez d'abord exécuter prepare_data.py pour préparer les données.")
        return
    
    print(f"\n📂 Chargement des données depuis {data_path}...")
    print(f"Limite configurée: {NB_COMMENT} plaintes")
    
    # Charger seulement le nombre nécessaire de lignes
    data = pd.read_csv(data_path, nrows=NB_COMMENT)
    
    # Vérifier les colonnes
    if 'complaint_text' not in data.columns or 'product' not in data.columns:
        print("Erreur: Le fichier CSV doit contenir les colonnes 'complaint_text' et 'product'")
        return
    
    print(f"Données chargées: {len(data)} plaintes")
    
    # ÉTAPE 2 : Préparation du texte
    # Vérifier si on doit utiliser le cache ou regénérer
    import pickle
    import sys
    
    use_cache = True
    if len(sys.argv) > 1 and sys.argv[1] == '--regenerate':
        use_cache = False
        print("🔄 Mode regénération : création de nouveaux tokens et Word2Vec")
    
    tokens_cache_path = 'models/tokens_cache.pkl'
    
    if use_cache and os.path.exists(tokens_cache_path) and os.path.exists('models/w2v.wv'):
        # Charger depuis le cache
        print("\n📦 Chargement des tokens depuis le cache...")
        with open(tokens_cache_path, 'rb') as f:
            tokens_data = pickle.load(f)
        tokens = tokens_data['tokens']
        print(f"✅ {len(tokens)} tokens chargés depuis le cache")
        
        print("\n📦 Chargement du modèle Word2Vec depuis le cache...")
        from gensim.models import KeyedVectors
        w2v_vectors = KeyedVectors.load("models/w2v.wv")
        # Créer un objet Word2Vec factice pour la compatibilité
        w2v = type('obj', (object,), {'wv': w2v_vectors})()
        print(f"✅ Word2Vec chargé avec {len(w2v.wv)} mots")
    else:
        # Générer et sauvegarder
        print("\n🔤 Tokenisation des textes...")
        tokens = tokenize_corpus(data["complaint_text"])
        
        # Sauvegarder les tokens
        tokens_data = {
            'tokens': tokens,
            'max_length': MAX_LENGTH,
            'w2v_size': W2V_SIZE,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(tokens_cache_path, 'wb') as f:
            pickle.dump(tokens_data, f)
        print(f"💾 Tokens sauvegardés dans {tokens_cache_path}")
        
        # ÉTAPE 3 : Création des embeddings Word2Vec
        # Apprend à représenter chaque mot comme un vecteur
        w2v = fit_word2vec(tokens)
    
    # ÉTAPE 4 & 5 : Vectorisation et labels (avec cache)
    vectors_cache_path = 'models/vectors_cache.pkl'
    labels_cache_path = 'models/labels_cache.pkl'
    
    if use_cache and os.path.exists(vectors_cache_path) and os.path.exists(labels_cache_path):
        # Charger les vecteurs et labels depuis le cache
        print("\n📦 Chargement des vecteurs et labels depuis le cache...")
        
        with open(vectors_cache_path, 'rb') as f:
            X = pickle.load(f)
        print(f"✅ Vecteurs chargés : shape {X.shape}")
        
        with open(labels_cache_path, 'rb') as f:
            y = pickle.load(f)
        print(f"✅ Labels chargés : shape {y.shape}")
    else:
        # Générer et sauvegarder
        # ÉTAPE 4 : Vectorisation des textes
        print("\n🎯 Vectorisation des textes...")
        X = text2vec(tokens, w2v)
        
        # Sauvegarder les vecteurs
        with open(vectors_cache_path, 'wb') as f:
            pickle.dump(X, f)
        print(f"💾 Vecteurs sauvegardés dans {vectors_cache_path}")
        
        # ÉTAPE 5 : Préparation des labels
        print("\n🏷️ Préparation des labels...")
        y = prepare_labels(data, class_mapping)
        
        # Sauvegarder les labels
        with open(labels_cache_path, 'wb') as f:
            pickle.dump(y, f)
        print(f"💾 Labels sauvegardés dans {labels_cache_path}")
    
    # ÉTAPE 6 : Division train/test
    # 75% pour l'entraînement, 25% pour le test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # ÉTAPE 7 : Création et configuration du modèle
    print(f"\n🤖 Création du modèle pour {num_classes} classes...")
    rnn = create_rnn(num_classes)
    
    # Compilation : définit comment le modèle va apprendre
    rnn.compile(
        optimizer='adam',  # Algorithme d'optimisation (ajuste les poids)
        loss="categorical_crossentropy",  # Fonction de perte pour classification multi-classes
        metrics=['categorical_accuracy']  # Métrique à surveiller (% de bonnes prédictions)
    )
    
    # Affiche un résumé de l'architecture du modèle
    rnn.summary()
    
    # ÉTAPE 8 : Entraînement
    print("\n🏋️ Entraînement du modèle...")
    
    # Callbacks pour le monitoring
    tensorboard_callback = TensorBoard("logs/rnn_complaints_classification")
    
    # Early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Checkpoint pour sauvegarder le meilleur modèle
    checkpoint = ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Calculer les poids de classes pour gérer le déséquilibre
    # Convertir y_train en indices pour compute_class_weight
    y_train_indices = np.argmax(y_train, axis=1)
    unique_classes = np.unique(y_train_indices)
    class_weight_array = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=y_train_indices
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weight_array)}
    
    # Entraînement du modèle
    _ = rnn.fit(  # history peut être utilisé pour analyser l'entraînement
        x=X_train, y=y_train,  # Données d'entraînement
        validation_data=(X_test, y_test),  # Données de validation
        epochs=20,  # Nombre de passes sur les données (augmenté pour plus de classes)
        batch_size=32,  # Traite 32 textes à la fois
        class_weight=class_weight_dict,  # Poids pour gérer le déséquilibre
        callbacks=[tensorboard_callback, early_stopping, checkpoint]  # Pour le monitoring
    )
    
    # ÉTAPE 9 : Sauvegarde
    # Sauvegarde le modèle entraîné pour utilisation future
    # Utiliser la variable d'environnement pour le nom du fichier
    output_file = os.getenv('OUTPUT_FILE', 'complaint_classifier')
    model_path = f"models/{output_file}.keras"
    rnn.save(model_path)
    print(f"\n✅ Modèle sauvegardé dans {model_path}")
    print("✅ Meilleur modèle sauvegardé dans models/best_model.keras")
    print("\n📊 Pour visualiser l'entraînement: tensorboard --logdir=logs")

# Point d'entrée du script
if __name__ == "__main__":
    # Lance le processus d'entraînement
    main()