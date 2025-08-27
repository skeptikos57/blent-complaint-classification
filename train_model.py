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
    Version optimisée pour les plaintes financières (vocabulaire technique).
    
    Args:
        tokens: Liste de listes de mots provenant de tokenize_corpus
    
    Returns:
        Modèle Word2Vec entraîné
    """
    import multiprocessing
    
    # Création et entraînement du modèle Word2Vec avec paramètres optimisés
    w2v = Word2Vec(
        sentences=tokens,           # Les commentaires tokenisés
        vector_size=W2V_SIZE,        # Taille des vecteurs (200 recommandé)
        min_count=W2V_MIN_COUNT,     # Ignore les mots rares (3 est bon)
        window=7,                    # Augmenté de 5 à 7 pour mieux capturer le contexte financier
        sg=1,                        # Skip-gram (1) au lieu de CBOW (0) - meilleur pour vocabulaire technique
        hs=0,                        # Hierarchical softmax désactivé
        negative=10,                 # Negative sampling augmenté de 5 à 10 pour meilleure précision
        alpha=0.025,                 # Taux d'apprentissage initial
        min_alpha=0.0001,           # Taux d'apprentissage minimal
        epochs=15,                   # Augmenté de 5 à 15 pour meilleur apprentissage (était implicite)
        workers=multiprocessing.cpu_count() - 1,  # Utilise tous les CPU disponibles - 1
        seed=42                      # Pour la reproductibilité
    )
    
    # Sauvegarder le modèle pour pouvoir le réutiliser plus tard
    w2v.wv.save("models/w2v.wv")
    
    # Afficher des statistiques sur le vocabulaire appris
    print(f"\n📊 Informations sur le modèle Word2Vec:")
    print(f"  • Taille du vocabulaire: {len(w2v.wv)} mots uniques")
    print(f"  • Dimensions des vecteurs: {W2V_SIZE}")
    print(f"  • Fenêtre de contexte: 7 mots")
    print(f"  • Algorithme: Skip-gram avec negative sampling")
    print(f"  • Epochs d'entraînement: 15")
    
    # Exemples de similarités pour vérifier la qualité
    try:
        test_words = ['credit', 'payment', 'account', 'debt', 'loan']
        print("\n🔍 Test de similarité (qualité des embeddings):")
        for word in test_words:
            if word in w2v.wv:
                similar = w2v.wv.most_similar(word, topn=3)
                print(f"  • '{word}' → {[w for w, _ in similar]}")
    except:
        pass
    
    return w2v

def text2vec(tokens, w2v):
    """Convertit chaque texte de plainte en une matrice de vecteurs Word2Vec.
    Version optimisée pour grandes quantités de données.
    
    Transforme les mots en nombres pour que le réseau de neurones puisse les comprendre.
    Chaque commentaire devient une matrice de taille fixe (W2V_SIZE x MAX_LENGTH).
    
    Args:
        tokens: Liste de listes de mots
        w2v: Modèle Word2Vec entraîné
    
    Returns:
        Array numpy 3D de forme (nb_commentaires, W2V_SIZE, MAX_LENGTH)
    """
    n_samples = len(tokens)
    
    # Estimation de la mémoire nécessaire
    memory_gb = (n_samples * W2V_SIZE * MAX_LENGTH * 4) / (1024**3)
    print(f"  Vectorisation de {n_samples} textes...")
    print(f"  Dimensions: {W2V_SIZE} x {MAX_LENGTH}")
    print(f"  Mémoire estimée: {memory_gb:.2f} GB")
    
    if memory_gb > 16:
        print(f"  ⚠️  ATTENTION: Utilisation mémoire très élevée!")
        print(f"  Suggestions:")
        print(f"    - Réduire NB_COMMENT (actuellement {n_samples})")
        print(f"    - Réduire W2V_SIZE (actuellement {W2V_SIZE})")
        print(f"    - Réduire MAX_LENGTH (actuellement {MAX_LENGTH})")
    
    # Pré-allouer le tableau numpy pour économiser la mémoire
    # Utilisation de float32 au lieu de float64 pour diviser la mémoire par 2
    X = np.zeros((n_samples, W2V_SIZE, MAX_LENGTH), dtype=np.float32)
    
    # Traitement par batch pour meilleure gestion mémoire
    batch_size = 5000
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        
        # Pour chaque commentaire dans le batch
        for i in range(batch_start, batch_end):
            row_tokens = tokens[i]
            # Remplir la matrice avec les vecteurs des mots (limité à MAX_LENGTH)
            for j in range(min(MAX_LENGTH, len(row_tokens))):
                # Vérifier si le mot existe dans le vocabulaire
                if row_tokens[j] in w2v.wv:
                    # Mettre le vecteur du mot j dans la colonne j
                    X[i, :, j] = w2v.wv[row_tokens[j]]
                # Si le mot n'est pas dans le vocabulaire, on laisse des zéros
        
        # Afficher la progression par batch
        progress = (batch_end * 100) / n_samples
        print(f"  {progress:.1f}% effectué ({batch_end}/{n_samples} textes)")
    
    print(f"  ✓ Vectorisation terminée : {X.nbytes / 1024**3:.2f} GB utilisés")
    
    return X

def prepare_labels(data, class_mapping):
    """Prépare les étiquettes (labels) pour l'entraînement.
    
    Transforme les catégories de produits en indices numériques puis en one-hot encoding.
    Version optimisée pour grandes quantités de données.
    
    Args:
        data: DataFrame avec colonne 'product'
        class_mapping: Dictionnaire mappant les noms de classes aux indices
    
    Returns:
        Array one-hot encoded (nb_classes colonnes : une seule vaut 1, les autres 0)
    """
    print(f"  Préparation de {len(data)} labels...")
    
    # Mapper les catégories de produits aux indices numériques
    y_indices = data['product'].map(class_mapping)
    
    # Version vectorisée plus rapide (sans boucle Python)
    num_classes = len(class_mapping)
    n_samples = len(y_indices)
    
    # Utiliser float32 au lieu de float64 pour économiser la mémoire
    y = np.zeros((n_samples, num_classes), dtype=np.float32)
    
    # Remplir uniquement les indices valides (vectorisé)
    valid_mask = ~y_indices.isna()
    valid_indices = y_indices[valid_mask].astype(int).values
    valid_positions = np.where(valid_mask)[0]
    
    # Remplissage vectorisé (beaucoup plus rapide que la boucle)
    y[valid_positions, valid_indices] = 1
    
    print(f"  ✓ Labels préparés : shape {y.shape}, mémoire : {y.nbytes / 1024**2:.1f} MB")
    
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
    
    # ÉTAPE 6 : Division train/test avec stratification
    # 75% pour l'entraînement, 25% pour le test
    # Stratification pour garder les proportions de classes
    from sklearn.model_selection import train_test_split
    
    # Convertir y en indices pour stratification
    y_indices = np.argmax(y, axis=1)
    
    # Vérifier si stratification possible
    from collections import Counter
    class_counts = Counter(y_indices)
    min_class_count = min(class_counts.values())
    
    if min_class_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y_indices
        )
        print("✅ Stratification appliquée pour maintenir les proportions")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        print("⚠️ Stratification impossible (classes trop rares)")
    
    # ÉTAPE 7 : Création et configuration du modèle
    print(f"\n🤖 Création du modèle pour {num_classes} classes...")
    rnn = create_rnn(num_classes)
    
    # Compilation : définit comment le modèle va apprendre
    # Optimiseur avec taux d'apprentissage ajusté pour 21 classes
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0005)  # Réduit de 0.001 pour apprentissage plus stable
    
    rnn.compile(
        optimizer=optimizer,  # Algorithme d'optimisation avec taux ajusté
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
        patience=10,  # Augmenté de 3 à 10 pour laisser plus de temps
        restore_best_weights=True,
        verbose=1  # Affiche quand il s'active
    )
    
    # Checkpoint pour sauvegarder le meilleur modèle
    checkpoint = ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Réduction du taux d'apprentissage si plateau
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Divise le LR par 2
        patience=5,   # Après 5 epochs sans amélioration
        min_lr=0.00001,
        verbose=1
    )
    
    # Calculer les poids de classes pour gérer le déséquilibre EXTRÊME
    # Convertir y_train en indices pour compute_class_weight
    y_train_indices = np.argmax(y_train, axis=1)
    unique_classes = np.unique(y_train_indices)
    
    # Compter les échantillons par classe
    from collections import Counter
    class_counts = Counter(y_train_indices)
    total_samples = len(y_train_indices)
    n_classes_present = len(unique_classes)
    
    # Calculer les poids avec stratégie agressive pour déséquilibre extrême
    class_weight_dict = {}
    max_count = max(class_counts.values())
    
    print("\n⚖️ Poids des classes (compensation du déséquilibre):")
    for class_id in range(num_classes):
        if class_id in class_counts:
            count = class_counts[class_id]
            # Stratégie sqrt pour éviter des poids trop extrêmes
            # mais toujours favoriser les minorités
            weight = np.sqrt(max_count / count)
            # Limiter les poids pour éviter l'instabilité
            weight = min(weight, 50.0)  # Cap à 50x
            class_weight_dict[class_id] = weight
            
            if count < 100:  # Classes très minoritaires
                print(f"  ⚠️ Classe {class_id}: {count} exemples → poids {weight:.2f}x")
        else:
            # Classe absente dans train
            class_weight_dict[class_id] = 0.0
            print(f"  ❌ Classe {class_id}: ABSENTE dans l'entraînement!")
    
    print(f"\n📊 Classes présentes: {n_classes_present}/{num_classes}")
    print(f"📊 Poids moyen appliqué: {np.mean(list(class_weight_dict.values())):.2f}x")
    
    # Entraînement du modèle
    _ = rnn.fit(  # history peut être utilisé pour analyser l'entraînement
        x=X_train, y=y_train,  # Données d'entraînement
        validation_data=(X_test, y_test),  # Données de validation
        epochs=30,  # Augmenté à 30 pour permettre plus d'apprentissage (21 classes)
        batch_size=64,  # Augmenté à 64 pour stabiliser l'apprentissage
        class_weight=class_weight_dict,  # Poids pour gérer le déséquilibre
        callbacks=[tensorboard_callback, early_stopping, checkpoint, reduce_lr],  # Pour le monitoring
        verbose=1  # Affiche la progression détaillée
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