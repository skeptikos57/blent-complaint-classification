# Classification de Demandes au Service Client

Système de classification automatique des demandes au service client d'une compagnie d'assurance utilisant le Deep Learning (CNN + LSTM) et Word2Vec.

## 📋 Description

Ce projet utilise l'intelligence artificielle pour classifier automatiquement les plaintes des clients dans différentes catégories de produits d'assurance. Le système analyse le texte des plaintes et prédit la catégorie appropriée, permettant ainsi un traitement plus rapide et plus efficace des demandes.

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Au moins 4 GB de RAM pour l'entraînement

### Étapes d'installation

1. **Cloner le projet**
```bash
git clone <url-du-projet>
cd support-classification
```

2. **Créer un environnement virtuel**

Option A - Méthode standard :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

Option B - Si erreur avec ensurepip :
```bash
# Sur Ubuntu/Debian
sudo apt-get install python3-venv python3-pip
python3 -m venv venv --without-pip
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python

# Ou avec virtualenv
pip install virtualenv
virtualenv venv
source venv/bin/activate
```

Option C - Utiliser directement pip sans environnement virtuel (non recommandé) :
```bash
pip install --user -r requirements.txt
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Télécharger le modèle de langue anglaise pour Spacy**
```bash
python -m spacy download en_core_web_md
```

5. **Créer les dossiers nécessaires**
```bash
mkdir -p data/raw data/prepared models logs
```

6. **Configuration** (optionnel)

Modifier le fichier `.env` pour ajuster les paramètres :
```env
NB_COMMENT = 50000      # Nombre de commentaires pour l'entraînement
W2V_SIZE = 100         # Dimension des vecteurs Word2Vec
W2V_MIN_COUNT = 3      # Fréquence minimale des mots
MAX_LENGTH = 64        # Longueur maximale des commentaires
```

## 📊 Analyse des Données

Avant l'entraînement, il est recommandé d'analyser la distribution des classes dans votre dataset :

```bash
python analyze_classes.py
```

Ce script :
- Lit le fichier `data/raw/complaints.csv`
- Analyse la distribution des catégories de produits
- Calcule les statistiques (nombre d'observations, pourcentages, déséquilibre)
- Sauvegarde les résultats dans `data/prepared/analyse_result.json`

### Structure du fichier CSV attendu

Le fichier `complaints.csv` doit contenir au minimum une colonne :
- `Product` : La catégorie du produit concerné par la plainte

## 🤖 Entraînement du Modèle

### Préparation des données

Placer votre fichier de données `complaints.csv` dans le dossier `data/raw/`. 

⚠️ **Note importante** : Le script `train_model.py` actuel est configuré pour un dataset de sentiments de films (`text_sentiment.csv`). Pour l'adapter aux plaintes d'assurance, vous devrez :

1. Modifier le nom du fichier dans `train_model.py` :
```python
# Ligne 197 - Remplacer :
data = pd.read_csv("data/text_sentiment.csv").dropna()
# Par :
data = pd.read_csv("data/raw/complaints.csv").dropna()
```

2. Ajuster les colonnes selon votre structure de données :
   - Colonne de texte : remplacer `"comment"` par le nom de votre colonne de plaintes
   - Colonnes de classes : adapter la fonction `merge_feelings()` pour vos catégories

### Lancer l'entraînement

```bash
python train_model.py
```

Le script va :
1. Charger les données depuis le CSV
2. Tokeniser les textes (découpage en mots)
3. Créer des embeddings Word2Vec
4. Transformer les textes en vecteurs numériques
5. Entraîner un modèle hybride CNN + LSTM
6. Sauvegarder les modèles dans le dossier `models/`

### Suivi de l'entraînement

Pour visualiser les métriques d'entraînement en temps réel :

```bash
tensorboard --logdir=logs
```

Puis ouvrir http://localhost:6006 dans votre navigateur.

## 🔮 Prédiction

Une fois le modèle entraîné, vous pouvez faire des prédictions sur de nouveaux textes :

```bash
python predict.py "Texte de la plainte client à analyser"
```

Exemple :
```bash
python predict.py "Je n'arrive pas à faire une réclamation pour mon assurance auto suite à un accident"
```

Le script affichera :
- La catégorie prédite
- Le niveau de confiance (en %)
- Les probabilités détaillées pour chaque catégorie

## 📁 Structure du Projet

```
support-classification/
│
├── data/
│   ├── raw/                 # Données brutes
│   │   └── complaints.csv    # Dataset des plaintes
│   └── prepared/             # Données préparées
│       └── analyse_result.json  # Résultats de l'analyse
│
├── models/                   # Modèles entraînés
│   ├── w2v.wv               # Modèle Word2Vec
│   └── comment_sentiment_rnn.keras  # Modèle de classification
│
├── logs/                     # Logs TensorBoard
│
├── analyze_classes.py        # Script d'analyse des classes
├── train_model.py           # Script d'entraînement
├── predict.py               # Script de prédiction
├── requirements.txt         # Dépendances Python
├── .env                     # Configuration
└── README.md               # Ce fichier
```

## 🔧 Personnalisation

### Adapter le modèle à vos données

Pour utiliser ce système avec vos propres données de plaintes :

1. **Format des données** : Assurez-vous que votre CSV contient :
   - Une colonne de texte avec les plaintes
   - Une colonne de catégories/labels

2. **Modifier `train_model.py`** :
   - Ligne 197 : Chemin du fichier CSV
   - Ligne 204 : Nom de la colonne de texte
   - Fonction `merge_feelings()` : Adapter selon vos labels

3. **Ajuster les hyperparamètres** dans `.env` selon vos besoins

### Architecture du modèle

Le modèle utilise une architecture hybride :
- **Word2Vec** : Pour créer des embeddings sémantiques des mots
- **CNN (Convolution 1D)** : Pour détecter des patterns locaux dans le texte
- **LSTM** : Pour comprendre les séquences et le contexte
- **Dense layers** : Pour la classification finale

## ⚠️ Notes Importantes

1. **Mémoire** : L'entraînement avec 50 000 commentaires nécessite environ 4 GB de RAM
2. **GPU** : Le code supporte CUDA pour accélérer l'entraînement sur GPU NVIDIA
3. **Temps d'entraînement** : Environ 30-60 minutes sur CPU, 5-10 minutes sur GPU
4. **Déséquilibre des classes** : Si certaines catégories sont sous-représentées, considérez :
   - L'utilisation de `class_weight` dans le fit
   - Des techniques de sur-échantillonnage (SMOTE)
   - Des métriques adaptées (Weighted F1-Score)

## 📈 Performances

Les performances du modèle dépendent de :
- La qualité et quantité des données
- L'équilibre entre les classes
- Les hyperparamètres choisis

Utilisez TensorBoard pour monitorer :
- Loss (entraînement et validation)
- Accuracy
- Overfitting (écart train/validation)

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche pour votre feature
3. Commit vos changements
4. Push vers votre branche
5. Ouvrir une Pull Request

## 📝 License

[Insérer votre license ici]

## 📧 Contact

[Vos informations de contact]