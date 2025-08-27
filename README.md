# Classification de Plaintes Service Client - Projet Éducatif

Système de classification automatique des plaintes clients utilisant le Deep Learning (CNN + LSTM) et Word2Vec. Ce projet éducatif démontre comment utiliser l'apprentissage profond pour catégoriser automatiquement des textes dans 21 catégories de produits financiers différents.

## 📋 Description

Ce projet utilise l'intelligence artificielle pour classifier automatiquement les plaintes des clients dans différentes catégories de produits financiers (cartes de crédit, prêts immobiliers, comptes bancaires, etc.). Le système analyse le texte des plaintes en anglais et prédit la catégorie appropriée avec un niveau de confiance, permettant ainsi un traitement plus rapide et efficace des demandes clients.

### Prérequis de Connaissances

Ce projet est idéal pour les étudiants et développeurs ayant :
- Des bases en **Python** (numpy, pandas)
- Une compréhension basique du **Machine Learning**
- Un intérêt pour le **Deep Learning** et le **NLP**
- Envie d'apprendre les **réseaux de neurones** (CNN, LSTM)

## 🎯 Catégories Supportées

Le système peut classifier les plaintes dans 21 catégories de produits financiers, incluant :
- Credit reporting and repair services
- Debt collection
- Mortgage
- Credit cards
- Checking/savings accounts
- Student loans
- Vehicle loans
- Payday loans
- Money transfers
- Et plus encore...

## 🚀 Installation Rapide

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Au moins 8 GB de RAM pour l'entraînement complet (4 GB pour le mode réduit)
- GPU NVIDIA avec CUDA (optionnel, mais recommandé)

### Installation

1. **Cloner le projet**
```bash
git clone <url-du-projet>
cd support-classification
```

2. **Créer et activer l'environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Télécharger le modèle de langue anglaise pour Spacy**
```bash
python -m spacy download en_core_web_sm
```

5. **Créer les dossiers nécessaires**
```bash
mkdir -p data/raw data/prepared models logs
```

## ⚙️ Configuration

Le fichier `.env` contient tous les paramètres configurables :

```env
# Chemin vers le fichier de données
INPUT_FILE=data/raw/complaints.csv

# Paramètres d'entraînement
NB_COMMENT=10000        # Nombre d'exemples à utiliser (10k pour test, 50k+ pour production)
W2V_SIZE=100           # Dimension des vecteurs Word2Vec
W2V_MIN_COUNT=3        # Fréquence minimale des mots
MAX_LENGTH=64          # Longueur maximale des textes (64 pour économiser la RAM, 256 pour meilleure précision)

# Nom du modèle
OUTPUT_FILE=complaint_classifier
```

**Conseils de configuration :**
- **Développement/Test** : `NB_COMMENT=10000`, `MAX_LENGTH=64` (utilise ~2GB RAM)
- **Production** : `NB_COMMENT=50000`, `MAX_LENGTH=256` (utilise ~8GB RAM, meilleure précision)

## 📊 Workflow Complet

### 1. Préparation des Données

```bash
# Analyser la distribution des classes dans vos données
python analyze_classes.py

# Préparer et nettoyer les données
python prepare_data.py
```

### 2. Entraînement du Modèle

```bash
# Lancer l'entraînement
python train_model.py

# Surveiller l'entraînement avec TensorBoard (dans un autre terminal)
tensorboard --logdir=logs
# Puis ouvrir http://localhost:6006
```

L'entraînement :
- Charge les données depuis `data/prepared/complaints_processed.csv`
- Crée des embeddings Word2Vec pour comprendre le sens des mots
- Entraîne un réseau de neurones hybride CNN + LSTM
- Sauvegarde automatiquement le meilleur modèle
- Génère les fichiers :
  - `models/complaint_classifier.keras` : Modèle principal
  - `models/best_model.keras` : Meilleur modèle (validation)
  - `models/w2v.wv` : Embeddings Word2Vec
  - `models/class_mapping.json` : Mapping des catégories

### 3. Prédiction

```bash
# Prédire la catégorie d'une plainte
python predict.py "I have an issue with my credit card payment"

# Exemples
python predict.py "My mortgage application was rejected without explanation"
python predict.py "I received unauthorized charges on my debit card"
python predict.py "The debt collector keeps calling me at work"
```

Output exemple :
```
📊 Analyse de la plainte:
Texte: "I have an issue with my credit card payment"

📁 Catégorie détectée: Credit card or prepaid card
💪 Confiance: 87.3%

📈 Top 5 catégories probables:
  • Credit card or prepaid card: 87.3%
  • Credit card: 8.1%
  • Debt collection: 2.5%
  • Bank account or service: 1.2%
  • Consumer Loan: 0.9%
```

## 📁 Structure du Projet

```
support-classification/
│
├── data/
│   ├── raw/                         # Données brutes
│   │   └── complaints.csv           # Dataset original (300k+ plaintes)
│   └── prepared/                    # Données prétraitées
│       ├── complaints_processed.csv # Données nettoyées
│       └── analyse_result.json      # Statistiques des classes
│
├── models/                          # Modèles sauvegardés
│   ├── complaint_classifier.keras   # Modèle principal
│   ├── best_model.keras            # Meilleur modèle (validation)
│   ├── w2v.wv                      # Embeddings Word2Vec
│   └── class_mapping.json          # Mapping des catégories
│
├── logs/                           # Logs TensorBoard
│
├── venv/                           # Environnement virtuel Python (non versionné)
│
├── prepare_data.py                 # Préparation et nettoyage des données
├── analyze_classes.py              # Analyse de la distribution des classes
├── train_model.py                  # Entraînement du modèle
├── predict.py                      # Prédictions sur nouveaux textes
├── confusion_matrix.py             # Génération de la matrice de confusion
│
├── .env                            # Variables d'environnement
├── requirements.txt                # Dépendances Python
└── README.md                       # Documentation (ce fichier)
```

## 🏗️ Architecture du Modèle

Le système utilise une architecture de deep learning sophistiquée :

1. **Word2Vec (100 dimensions)** : Comprend le sens sémantique des mots
2. **Conv1D (2 couches)** : Détecte les patterns locaux dans le texte
3. **MaxPooling** : Réduit la dimensionnalité et garde les features importantes
4. **LSTM (2 couches, 256+128 unités)** : Comprend le contexte et les séquences
5. **Dropout (0.5)** : Prévient le surapprentissage
6. **Dense (128 unités)** : Couche de décision
7. **Sortie (21 classes)** : Classification finale avec softmax

**Caractéristiques :**
- ~650k paramètres entraînables
- Support GPU (CUDA) pour entraînement rapide
- Early stopping pour éviter le surapprentissage
- Sauvegarde automatique du meilleur modèle

## 📈 Performances

### Métriques typiques
- **Accuracy** : 40-60% sur 21 classes (avec 10k exemples)
- **Accuracy** : 70-85% sur 21 classes (avec 50k+ exemples)
- **Temps d'entraînement** :
  - CPU : 30-60 minutes (50k exemples)
  - GPU : 5-10 minutes (50k exemples)

### Optimisation des performances
1. **Plus de données** : Utiliser `NB_COMMENT=50000` ou plus
2. **Séquences plus longues** : `MAX_LENGTH=256` pour plus de contexte
3. **Plus d'époques** : Modifier `epochs=20` dans train_model.py
4. **Ajuster l'architecture** : Ajouter des couches ou augmenter les unités

## 🔧 Résolution de Problèmes

### Erreur de mémoire (Killed)
**Symptôme** : Le script s'arrête avec "Killed" pendant la vectorisation

**Solutions** :
1. Réduire `NB_COMMENT` dans `.env` (ex: 10000)
2. Réduire `MAX_LENGTH` dans `.env` (ex: 64)
3. Utiliser `dtype=float32` (déjà configuré)
4. Fermer d'autres applications
5. Augmenter le swap Linux si nécessaire

### Faible précision
**Solutions** :
1. Augmenter `NB_COMMENT` pour plus de données
2. Augmenter `MAX_LENGTH` pour plus de contexte
3. Entraîner plus longtemps (augmenter epochs)
4. Vérifier le déséquilibre des classes avec `analyze_classes.py`

### GPU non détecté
**Solutions** :
1. Installer CUDA et cuDNN compatibles
2. Installer tensorflow-gpu : `pip install tensorflow-gpu`
3. Vérifier avec : `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## 🚀 Utilisation Avancée

### Entraînement avec paramètres personnalisés

Modifier directement dans `.env` ou créer plusieurs fichiers de config :

```bash
# Créer une config de production
cp .env .env.production
# Éditer .env.production avec des valeurs plus élevées

# Utiliser la config de production
cp .env.production .env
python train_model.py
```

### API REST (exemple)

```python
# api.py (exemple simple avec Flask)
from flask import Flask, request, jsonify
import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_category():
    text = request.json['text']
    category, confidence, _ = predict.predict_complaint(text)
    return jsonify({
        'category': category,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(port=5000)
```


## 📊 Monitoring et Métriques

### TensorBoard

Pendant l'entraînement, surveillez :
- **Loss** : Doit diminuer progressivement
- **Accuracy** : Doit augmenter progressivement
- **Val_loss vs Train_loss** : Si l'écart augmente = surapprentissage

```bash
tensorboard --logdir=logs --port=6006
```

### Métriques personnalisées

Le modèle génère automatiquement :
- Matrice de confusion (après entraînement)
- Précision par classe
- F1-score pondéré
- Rapport de classification détaillé

## 📊 Matrice de Confusion

### Qu'est-ce qu'une matrice de confusion ?

Une **matrice de confusion** est un outil essentiel pour évaluer les performances d'un modèle de classification. Elle présente sous forme de tableau le nombre de prédictions correctes et incorrectes pour chaque catégorie.

### Génération de la matrice

```bash
# Générer la matrice avec toutes les données de test
python confusion_matrix.py

# Ou limiter le nombre d'échantillons pour des tests rapides
python confusion_matrix.py --samples 5000
```

### Visualisations générées

Le script `confusion_matrix.py` génère plusieurs analyses :

1. **Matrice de confusion standard** : Affiche le nombre absolu de prédictions pour chaque combinaison classe réelle/classe prédite
2. **Matrice de confusion normalisée** : Affiche les pourcentages pour mieux comprendre les taux d'erreur
3. **Métriques de performance** :
   - Accuracy globale du modèle
   - Précision, rappel et F1-score par catégorie
   - Top 5 des meilleures et pires classes
4. **Analyse des confusions** : Identifie les erreurs de classification les plus fréquentes

### Interprétation des résultats

- **Diagonale principale** : Prédictions correctes (plus les valeurs sont élevées, mieux c'est)
- **Hors diagonale** : Erreurs de classification (indiquent quelles catégories sont confondues)
- **Classes problématiques** : Les catégories avec beaucoup de confusions suggèrent des similarités dans le langage utilisé

### Exemple de résultats typiques

Avec 10k échantillons d'entraînement :
- **Accuracy globale** : 25-40% sur 21 classes
- **Meilleures classes** : "Credit reporting" (50%+ F1-score)
- **Confusions fréquentes** : Les produits de crédit similaires (cartes, prêts) sont souvent confondus

### Fichiers générés

Les résultats sont sauvegardés dans le dossier `models/` :
- `confusion_matrix_[timestamp].png` : Matrice visuelle avec nombres absolus
- `confusion_matrix_normalized_[timestamp].png` : Matrice en pourcentages
- `classification_report_[timestamp].json` : Métriques détaillées en JSON

## 📚 Objectifs Pédagogiques

Ce projet éducatif permet d'apprendre :

1. **Deep Learning** : Comprendre l'architecture CNN + LSTM
2. **NLP (Natural Language Processing)** : Traitement du langage naturel avec Word2Vec
3. **Classification Multi-classes** : Gérer 21 catégories différentes
4. **Préparation des Données** : Nettoyage et preprocessing de textes
5. **Gestion du Déséquilibre** : Traiter des classes déséquilibrées
6. **Optimisation Mémoire** : Gérer des datasets volumineux efficacement
7. **MLOps Basique** : Configuration, monitoring avec TensorBoard, versioning des modèles

### Exercices Suggérés

Pour approfondir votre apprentissage :

1. **Expérimentez avec les hyperparamètres** dans `.env`
2. **Analysez l'impact** du nombre d'exemples d'entraînement sur la précision
3. **Comparez les performances** avec différentes valeurs de MAX_LENGTH
4. **Visualisez les métriques** dans TensorBoard pendant l'entraînement
5. **Testez le modèle** avec vos propres exemples de plaintes
6. **Étudiez la matrice de confusion** pour comprendre les erreurs du modèle

## ⚠️ Notes Importantes

1. **Dataset** : Le système est entraîné sur des données publiques de plaintes financières (CFPB)
2. **Langue** : Actuellement optimisé pour l'anglais uniquement
3. **RGPD** : Assurez-vous de respecter les réglementations sur les données personnelles
4. **Biais** : Le modèle peut refléter les biais présents dans les données d'entraînement

## 📝 Licence

MIT License - Voir fichier LICENSE pour plus de détails

## 📧 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Email : [votre-email]
- Documentation : [lien-vers-docs]

---

**Dernière mise à jour** : Novembre 2024
**Version** : 1.0.0
**Type** : Projet Éducatif 📚
**Niveau** : Intermédiaire/Avancé