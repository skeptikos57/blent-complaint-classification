"""Script pour générer et visualiser la matrice de confusion du modèle.

Ce script :
1. Charge le modèle entraîné et les données de test
2. Fait des prédictions sur l'ensemble de test
3. Génère une matrice de confusion
4. Crée des visualisations et calcule les métriques de performance
5. Sauvegarde les résultats dans un fichier

Usage:
    python confusion_matrix.py
    python confusion_matrix.py --samples 5000  # Limiter le nombre d'échantillons
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import json
from datetime import datetime

# Configuration pour réduire les messages de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Charger les variables d'environnement
load_dotenv(override=True)

# Configuration des paramètres
W2V_SIZE = int(os.getenv("W2V_SIZE", 100))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "complaint_classifier")


def load_models_and_data():
    """Charge les modèles et le mapping des classes."""
    print("\n📂 Chargement des modèles...")
    
    # Charger le modèle de classification
    model_path = f"models/{OUTPUT_FILE}.keras"
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Le modèle {model_path} n'existe pas.")
        print("Veuillez d'abord entraîner le modèle avec train_model.py")
        sys.exit(1)
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Modèle chargé depuis {model_path}")
    
    # Charger Word2Vec
    w2v = KeyedVectors.load("models/w2v.wv")
    print("✅ Word2Vec chargé")
    
    # Charger le mapping des classes
    with open('models/class_mapping.json', 'r', encoding='utf-8') as f:
        class_data = json.load(f)
    print("✅ Mapping des classes chargé")
    
    return model, w2v, class_data


def load_and_prepare_data(max_samples=None):
    """Charge et prépare les données de test."""
    print("\n📊 Chargement des données...")
    
    # Charger les données
    data_path = 'data/prepared/complaints_processed.csv'
    if not os.path.exists(data_path):
        print(f"❌ Erreur: Le fichier {data_path} n'existe pas.")
        print("Veuillez d'abord préparer les données avec prepare_data.py")
        sys.exit(1)
    
    data = pd.read_csv(data_path)
    
    # Limiter le nombre d'échantillons si demandé
    if max_samples and len(data) > max_samples:
        data = data.sample(n=max_samples, random_state=42)
        print(f"📉 Données limitées à {max_samples} échantillons")
    
    print(f"✅ {len(data)} échantillons chargés")
    
    return data


def tokenize_text(text):
    """Tokenise un texte en mots."""
    import re
    # Simple tokenization
    text = text.lower()
    # Garder seulement lettres et espaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Diviser en mots
    words = text.split()
    return words


def text_to_vector(text, w2v):
    """Convertit un texte en matrice de vecteurs Word2Vec."""
    # Tokeniser le texte
    tokens = tokenize_text(text)
    
    # Créer la matrice de vecteurs
    matrix = np.zeros((W2V_SIZE, MAX_LENGTH), dtype=np.float32)
    
    # Remplir avec les vecteurs des mots
    for j, word in enumerate(tokens[:MAX_LENGTH]):
        if word in w2v:  # KeyedVectors n'a pas d'attribut wv
            matrix[:, j] = w2v[word]
    
    return matrix


def predict_batch(model, texts, w2v, batch_size=32):
    """Fait des prédictions par batch pour économiser la mémoire."""
    predictions = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"\n🔮 Génération des prédictions ({n_batches} batches)...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_vectors = []
        
        for text in batch_texts:
            vector = text_to_vector(text, w2v)
            batch_vectors.append(vector)
        
        batch_array = np.array(batch_vectors)
        batch_preds = model.predict(batch_array, verbose=0)
        predictions.extend(batch_preds)
        
        # Afficher la progression
        progress = min(i + batch_size, len(texts))
        print(f"  {progress}/{len(texts)} échantillons traités ({progress*100/len(texts):.1f}%)", end='\r')
    
    print(f"\n✅ Prédictions terminées pour {len(texts)} échantillons")
    return np.array(predictions)


def generate_confusion_matrix(y_true, y_pred, class_names):
    """Génère et affiche la matrice de confusion."""
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Créer la figure
    plt.figure(figsize=(20, 16))
    
    # Utiliser une palette de couleurs adaptée
    # Pour une meilleure lisibilité avec beaucoup de classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Nombre de prédictions'})
    
    plt.title('Matrice de Confusion - Classification des Plaintes\n', fontsize=16, fontweight='bold')
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.ylabel('Classe Réelle', fontsize=12)
    
    # Rotation des labels pour meilleure lisibilité
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/confusion_matrix_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Matrice de confusion sauvegardée dans {filename}")
    
    return cm


def generate_normalized_matrix(y_true, y_pred, class_names):
    """Génère une matrice de confusion normalisée (en pourcentages)."""
    
    # Calculer la matrice de confusion normalisée
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Créer la figure
    plt.figure(figsize=(20, 16))
    
    # Afficher avec des pourcentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Pourcentage (%)'})
    
    plt.title('Matrice de Confusion Normalisée - Classification des Plaintes\n', fontsize=16, fontweight='bold')
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.ylabel('Classe Réelle', fontsize=12)
    
    # Rotation des labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/confusion_matrix_normalized_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"💾 Matrice normalisée sauvegardée dans {filename}")
    
    return cm_normalized


def calculate_metrics(y_true, y_pred, class_names):
    """Calcule et affiche les métriques de performance."""
    
    print("\n📈 Métriques de Performance")
    print("=" * 60)
    
    # Accuracy globale
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Accuracy globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Rapport de classification détaillé
    # Obtenir les labels uniques présents dans les données
    unique_labels = sorted(set(y_true) | set(y_pred))
    target_names_filtered = [class_names[i] for i in unique_labels]
    
    report = classification_report(y_true, y_pred, 
                                  labels=unique_labels,
                                  target_names=target_names_filtered,
                                  output_dict=True,
                                  zero_division=0)
    
    # Sauvegarder le rapport dans un fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"models/classification_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Rapport détaillé sauvegardé dans {report_file}")
    
    # Afficher les métriques par classe (top 5 meilleures et pires)
    class_metrics = []
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            class_metrics.append({
                'Classe': class_name[:40],  # Tronquer les noms longs
                'Précision': metrics['precision'],
                'Rappel': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': int(metrics['support'])
            })
    
    # Convertir en DataFrame pour un affichage plus joli
    df_metrics = pd.DataFrame(class_metrics)
    df_metrics = df_metrics.sort_values('F1-Score', ascending=False)
    
    print("\n🏆 Top 5 classes (meilleur F1-Score):")
    print(df_metrics.head(5).to_string(index=False))
    
    print("\n⚠️ Bottom 5 classes (plus faible F1-Score):")
    print(df_metrics.tail(5).to_string(index=False))
    
    # Métriques moyennes pondérées
    print("\n📊 Métriques moyennes pondérées:")
    print(f"  • Précision: {report['weighted avg']['precision']:.4f}")
    print(f"  • Rappel: {report['weighted avg']['recall']:.4f}")
    print(f"  • F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    return report


def analyze_confusion_patterns(cm, class_names):
    """Analyse les principales confusions dans la matrice."""
    
    print("\n🔍 Analyse des Confusions Principales")
    print("=" * 60)
    
    # Pour chaque classe, trouver avec quoi elle est le plus confondue
    confusions = []
    
    # La matrice peut être plus petite que le nombre total de classes
    # si certaines classes ne sont pas présentes dans l'échantillon
    matrix_size = cm.shape[0]
    
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j and i < len(class_names) and j < len(class_names) and cm[i, j] > 0:  # Ignorer la diagonale et les zéros
                confusions.append({
                    'Vraie classe': class_names[i][:30],
                    'Prédite comme': class_names[j][:30],
                    'Nombre': cm[i, j],
                    'Pourcentage': (cm[i, j] / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
                })
    
    # Trier par nombre de confusions
    confusions = sorted(confusions, key=lambda x: x['Nombre'], reverse=True)
    
    # Afficher les 10 principales confusions
    print("\nTop 10 des confusions les plus fréquentes:")
    for i, conf in enumerate(confusions[:10], 1):
        print(f"{i:2d}. {conf['Vraie classe']} → {conf['Prédite comme']}")
        print(f"    {conf['Nombre']} fois ({conf['Pourcentage']:.1f}% des cas)")
    
    # Identifier les classes problématiques
    print("\n⚠️ Classes les plus difficiles à prédire:")
    diagonal = np.diag(cm)
    row_sums = cm.sum(axis=1)
    accuracies = diagonal / (row_sums + 1e-10)  # Éviter division par zéro
    
    # Limiter aux classes présentes dans la matrice
    difficult_classes = [(class_names[i], acc) for i, acc in enumerate(accuracies) if i < len(class_names)]
    difficult_classes.sort(key=lambda x: x[1])
    
    # Afficher jusqu'à 5 classes difficiles
    for class_name, acc in difficult_classes[:min(5, len(difficult_classes))]:
        print(f"  • {class_name[:40]}: {acc*100:.1f}% de précision")


def main():
    """Fonction principale."""
    
    print("=" * 60)
    print("🎯 GÉNÉRATION DE LA MATRICE DE CONFUSION")
    print("=" * 60)
    
    # Parser les arguments de ligne de commande
    max_samples = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--samples' and len(sys.argv) > 2:
            max_samples = int(sys.argv[2])
    
    try:
        # Charger les modèles et données
        model, w2v, class_data = load_models_and_data()
        data = load_and_prepare_data(max_samples)
        
        # Préparer les données
        print("\n🔄 Préparation des données...")
        
        # Mapper les catégories aux indices
        class_mapping = class_data['class_mapping']
        data['label'] = data['product'].map(class_mapping)
        
        # Supprimer les lignes avec des labels manquants
        data = data.dropna(subset=['label'])
        data['label'] = data['label'].astype(int)
        
        # Diviser en train/test (on utilise les mêmes proportions que l'entraînement)
        # Vérifier si on peut faire une stratification
        min_class_count = data['label'].value_counts().min()
        if min_class_count >= 2:
            _, test_data = train_test_split(data, test_size=0.2, 
                                           random_state=42, 
                                           stratify=data['label'])
        else:
            # Si certaines classes ont trop peu d'exemples, pas de stratification
            print("⚠️ Certaines classes ont trop peu d'exemples, stratification désactivée")
            _, test_data = train_test_split(data, test_size=0.2, 
                                           random_state=42)
        
        print(f"✅ {len(test_data)} échantillons de test préparés")
        
        # Faire les prédictions
        X_test = test_data['complaint_text'].values
        y_true = test_data['label'].values
        
        # Prédire par batch
        predictions = predict_batch(model, X_test, w2v)
        y_pred = np.argmax(predictions, axis=1)
        
        # Préparer les noms de classes (tronqués pour l'affichage)
        class_names = [class_data['index_to_class'][str(i)][:30] 
                      for i in range(len(class_data['index_to_class']))]
        
        # Générer les matrices de confusion
        print("\n📊 Génération des visualisations...")
        cm = generate_confusion_matrix(y_true, y_pred, class_names)
        _ = generate_normalized_matrix(y_true, y_pred, class_names)  # Génère et sauvegarde la matrice normalisée
        
        # Calculer les métriques
        _ = calculate_metrics(y_true, y_pred, class_names)  # Génère et sauvegarde le rapport de classification
        
        # Analyser les patterns de confusion
        analyze_confusion_patterns(cm, class_names)
        
        # Afficher les graphiques
        print("\n✅ Analyse terminée!")
        print("📊 Pour voir les graphiques, ouvrez les fichiers PNG dans le dossier models/")
        
        # Optionnel : afficher les graphiques directement
        try:
            plt.show()
        except:
            pass  # Si pas d'environnement graphique
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()