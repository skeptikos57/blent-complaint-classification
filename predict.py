"""Script de prédiction de la catégorie de produit pour une plainte client.

Ce script :
1. Charge les modèles pré-entraînés (RNN et Word2Vec)
2. Prend une plainte en entrée via la ligne de commande
3. Transforme le texte en vecteurs numériques
4. Prédit la catégorie de produit financier concerné

Usage:
    python predict.py "Votre plainte ici"
"""

import os
import sys  # Pour récupérer les arguments de ligne de commande

# Configuration pour réduire les messages de TensorFlow
# IMPORTANT : Ces lignes DOIVENT être avant l'import de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 3 = Afficher seulement les erreurs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Désactive les optimisations oneDNN

# Imports des bibliothèques nécessaires
import numpy as np  # Pour les calculs numériques et tableaux
import tensorflow as tf  # Pour charger et utiliser le modèle de deep learning
from gensim.models import KeyedVectors  # Pour charger le modèle Word2Vec sauvegardé
from spacy.lang.fr import French  # Pour analyser et découper le texte en français
from dotenv import load_dotenv  # Pour charger les variables d'environnement
import json  # Pour charger le mapping des classes

# Configuration supplémentaire pour réduire les messages de TensorFlow
tf.get_logger().setLevel('ERROR')  # Ne montre que les erreurs, pas les warnings

def load_models():
    """Charge les modèles pré-entraînés nécessaires pour la prédiction.
    
    Cette fonction charge :
    - Le modèle RNN : le réseau de neurones qui fait la prédiction
    - Le modèle Word2Vec : pour convertir les mots en vecteurs numériques
    - Le tokenizer Spacy : pour découper le texte en mots
    
    Returns:
        Tuple contenant (modèle_rnn, modèle_word2vec, tokenizer_spacy)
    """
    # Charge le modèle de deep learning sauvegardé après l'entraînement
    # Utiliser la variable d'environnement pour le nom du fichier
    output_file = os.getenv('OUTPUT_FILE', 'complaint_classifier')
    model_path = f"models/{output_file}.keras"
    rnn = tf.keras.models.load_model(model_path)
    print(f"✅ Modèle RNN chargé depuis {model_path}")
    
    # Charge le vocabulaire Word2Vec (les vecteurs de tous les mots appris)
    w2v = KeyedVectors.load("models/w2v.wv")
    print("✅ Modèle Word2Vec chargé depuis models/w2v.wv")
    
    # Charge le mapping des classes
    with open('models/class_mapping.json', 'r', encoding='utf-8') as f:
        class_data = json.load(f)
    print("✅ Mapping des classes chargé depuis models/class_mapping.json")
    
    # Initialise l'analyseur de texte anglais (les plaintes sont en anglais)
    from spacy.lang.en import English
    nlp = English()
    
    return rnn, w2v, nlp, class_data

def process_comment(comment, w2v, nlp, W2V_SIZE, MAX_LENGTH):
    """Transforme un commentaire texte en matrice de vecteurs Word2Vec.
    
    Cette fonction fait exactement la même transformation que pendant
    l'entraînement pour que le modèle comprenne le nouveau commentaire.
    
    Args:
        comment: Le texte du commentaire à analyser
        w2v: Modèle Word2Vec pour convertir les mots en vecteurs
        nlp: Tokenizer Spacy pour découper le texte
        W2V_SIZE: Taille des vecteurs Word2Vec (100)
        MAX_LENGTH: Nombre maximum de mots à considérer (64)
    
    Returns:
        Matrice numpy de forme (100, 64) représentant le commentaire
    """
    # Met le texte en minuscules pour uniformiser
    comment = comment.lower()
    
    # Découpe le texte en mots et enlève la ponctuation
    tokens = [x.text for x in nlp(comment) if not x.is_punct]
    
    # Crée une matrice vide (100 lignes x 64 colonnes) remplie de zéros
    row = np.zeros((W2V_SIZE, MAX_LENGTH))
    
    # Remplit la matrice avec les vecteurs de chaque mot
    for j in range(min(MAX_LENGTH, len(tokens))):
        try:
            # Place le vecteur du mot j dans la colonne j
            row[:, j] = w2v[tokens[j]]
        except KeyError:
            # Si le mot n'a pas été vu pendant l'entraînement, on laisse des zéros
            print(f"Le mot '{tokens[j]}' ne fait pas partie du vocabulaire.")
    
    return row

def predict(comment, rnn, w2v, nlp, W2V_SIZE, MAX_LENGTH, class_data):
    """Prédit la catégorie de produit d'une plainte et retourne les résultats.
    
    Processus de prédiction :
    1. Transforme le texte en matrice de vecteurs
    2. Fait passer cette matrice dans le réseau de neurones
    3. Récupère les probabilités pour chaque catégorie de produit
    4. Choisit la catégorie avec la plus haute probabilité
    
    Args:
        comment: Texte de la plainte à analyser
        rnn: Modèle de réseau de neurones entraîné
        w2v: Modèle Word2Vec
        nlp: Tokenizer Spacy
        W2V_SIZE: Dimension des vecteurs (100)
        MAX_LENGTH: Longueur max (64)
        class_data: Mapping des classes de produits
    
    Returns:
        Tuple contenant:
        - La catégorie de produit prédite
        - Le niveau de confiance en pourcentage
        - Les probabilités détaillées pour chaque catégorie
    """
    # Transforme le commentaire en matrice numérique
    processed_comment = process_comment(comment, w2v, nlp, W2V_SIZE, MAX_LENGTH)
    
    # Fait la prédiction avec le modèle
    # np.asarray([...]) ajoute une dimension car le modèle attend un batch
    # verbose=0 pour ne pas afficher de messages pendant la prédiction
    prediction = rnn.predict(np.asarray([processed_comment]), verbose=0)
    
    # Interprétation des résultats
    # Le modèle retourne 21 probabilités (une pour chaque catégorie de produit)
    
    # argmax trouve l'indice de la plus haute probabilité
    predicted_class = np.argmax(prediction[0])
    
    # Récupère le nom de la catégorie depuis le mapping
    category_name = class_data['index_to_class'][str(predicted_class)]
    
    # Convertit la probabilité en pourcentage
    confidence = prediction[0][predicted_class] * 100
    
    return category_name, confidence, prediction[0]


def main():
    """Fonction principale qui orchestre la prédiction.
    
    Étapes :
    1. Charge les modèles sauvegardés
    2. Récupère le commentaire depuis la ligne de commande
    3. Fait la prédiction
    4. Affiche les résultats de façon claire
    """
    # ÉTAPE 1 : Chargement des modèles
    # Charge le RNN, Word2Vec, le tokenizer et le mapping des classes
    rnn, w2v, nlp, class_data = load_models()
    
    # ÉTAPE 2 : Récupération du commentaire
    # Vérifie qu'un commentaire a été fourni en argument
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"votre commentaire ici\"")
        print("Exemple: python predict.py \"Ce film était vraiment excellent !\"")
        sys.exit(1)  # Quitte le programme avec un code d'erreur
    
    # Joint tous les arguments pour former le commentaire complet
    # (au cas où il y aurait des espaces dans le commentaire)
    comment = " ".join(sys.argv[1:])
    
    # ÉTAPE 3 : Prédiction
    # Analyse le commentaire et prédit la catégorie de produit
    category, confidence, probabilities = predict(comment, rnn, w2v, nlp, W2V_SIZE, MAX_LENGTH, class_data)
    
    # ÉTAPE 4 : Affichage des résultats
    print(f"\n📊 Analyse de la plainte:")
    print(f"Texte: \"{comment[:150]}{'...' if len(comment) > 150 else ''}\"")
    
    # Affiche la catégorie principale et la confiance
    print(f"\n📁 Catégorie détectée: {category}")
    print(f"💪 Confiance: {confidence:.1f}%")
    
    # Affiche les 5 catégories les plus probables
    print(f"\n📈 Top 5 catégories probables:")
    top_indices = np.argsort(probabilities)[-5:][::-1]
    for idx in top_indices:
        cat_name = class_data['index_to_class'][str(idx)]
        prob = probabilities[idx] * 100
        if prob > 1:  # N'affiche que les probabilités > 1%
            print(f"  • {cat_name[:50]}: {prob:.1f}%")

# Point d'entrée du script
# Ce code ne s'exécute que si on lance directement ce fichier
# (pas si on l'importe dans un autre fichier)
if __name__ == "__main__":
    # Charge les variables d'environnement depuis le fichier .env
    # Cela permet de configurer les paramètres sans modifier le code
    load_dotenv(override=True)
    
    # Récupère les paramètres de configuration
    # Ces valeurs doivent être identiques à celles utilisées pour l'entraînement
    W2V_SIZE = int(os.getenv("W2V_SIZE", 100))  # Taille des vecteurs Word2Vec
    W2V_MIN_COUNT = int(os.getenv("W2V_MIN_COUNT", 3))  # Fréquence min (pas utilisé ici)
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))  # Nombre max de mots par commentaire
    
    # Lance la fonction principale
    main()