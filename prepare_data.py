#!/usr/bin/env python3
"""
Script de prétraitement et nettoyage des données pour la classification de réclamations
Génère un fichier nettoyé et prétraité prêt pour l'entraînement du modèle
"""

import pandas as pd
import numpy as np
import re
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent
INPUT_FILE = os.getenv('INPUT_FILE', 'data/raw/complaints.csv')
RAW_DATA_PATH = PROJECT_ROOT / INPUT_FILE
PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'prepared' / 'complaints_processed.csv'
STATS_PATH = PROJECT_ROOT / 'data' / 'prepared' / 'preprocessing_stats.json'
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)  # Créer le dossier logs s'il n'existe pas

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paramètres de prétraitement
MIN_TEXT_LENGTH = 20  # Longueur minimale du texte après nettoyage
MAX_TEXT_LENGTH = 5000  # Longueur maximale du texte (pour éviter les outliers)
CHUNK_SIZE = 10000  # Taille des chunks pour le traitement


class TextPreprocessor:
    """Classe pour le prétraitement du texte"""
    
    def __init__(self):
        """Initialise les patterns de nettoyage"""
        # Patterns pour l'anonymisation - ORDRE TRÈS IMPORTANT pour éviter les conflits
        # Les patterns sont appliqués dans l'ordre, donc les plus spécifiques d'abord
        
        # Note: (?i) rend le pattern case-insensitive pour X/x
        # ORDRE CRITIQUE : Les patterns les plus spécifiques doivent être traités EN PREMIER
        self.patterns = {
            # === PATTERNS TRÈS SPÉCIFIQUES (traiter en tout premier) ===
            
            # SSN - DOIT être avant les dates car XXX-XX-XXXX ressemble à une date
            'ssn_anon': r'(?i)\b(xxx|XXX)[\-](xx|XX)[\-](xxxx|\d{4}|XXXX)\b',  # XXX-XX-XXXX
            'ssn_partial': r'(?i)\b\d{3}[\-](xx|XX)[\-](xxxx|\d{4}|XXXX)\b',  # 123-XX-XXXX
            'ssn_real': r'\b\d{3}[\-]\d{2}[\-]\d{4}\b',  # 123-45-6789
            
            # TÉLÉPHONES - DOIT être avant les dates car XXX-XXX-XXXX ressemble à une date
            'phone_anon_full': r'(?i)\b(xxx|XXX)[\-](xxx|XXX)[\-](xxxx|XXXX|\d{4})\b',  # XXX-XXX-XXXX
            'phone_anon_with_parens': r'(?i)\(\s*(xxx|XXX)\s*\)[\s\-]*(xxx|XXX)[\-](xxxx|XXXX|\d{4})',  # (XXX) XXX-XXXX
            'phone_anon_partial': r'(?i)\b(xxx|XXX)[\-](xxxx|XXXX|\d{4})\b',  # XXX-XXXX
            'phone_partial_real': r'\b\d{3}[\-]\d{4}\b',  # 666-1666
            'phone_real': r'\b(?:\(\d{3}\)[\s\-]?\d{3}[\-]\d{4}|\d{3}[\-]\d{3}[\-]\d{4})\b',  # (123) 456-7890
            
            # === DATES (après SSN et téléphones pour éviter les conflits) ===
            # Dates avec TIRETS mais seulement XX-XX-année (pas XX-XX-XXXX qui est SSN)
            'dates_dash_year': r'(?i)\b(xx|XX)[\-](xx|XX)[\-](\d{4}|\d{2})\b',  # XX-XX-2020, xx-xx-2020
            'dates_full_anon': r'(?i)\b(xx?xx?|XX?XX?)[/](xx?xx?|XX?XX?)[/](\d{4}|\d{2}|(xx?xx?|XX?XX?))\b',  # XX/XX/2022 avec SLASH
            'dates_partial_anon': r'(?i)\b(xx?xx?|XX?XX?)[/](xx?xx?|XX?XX?)[/](xx?xx?|XX?XX?)\b',  # XX/XX/XX avec SLASH
            'dates_year_first': r'(?i)\b(\d{4}|(xx?xx?|XX?XX?))[/](xx?xx?|XX?XX?)[/](xx?xx?|XX?XX?)\b',  # 2022/XX/XX ou XXXX/XX/XX
            'dates_real': r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',  # 6/12/18, 12-25-2022
            
            # === MONTANTS MONÉTAIRES ===
            'money_brackets': r'\{\$[\d,]+\.?\d*\}',  # {$560.00}
            'money_normal': r'\$[\d,]+\.?\d*',  # $560.00
            
            # === NUMÉROS DE COMPTE/CARTE (patterns très spécifiques d'abord) ===
            # Case-insensitive pour les X
            'account_long': r'(?i)\b(xxxx|XXXX)[\-](xxxxxxxxxxxx|XXXXXXXXXXXX)\b',  # XXXX-XXXXXXXXXXXX exactement
            'account_card_full': r'(?i)\b(xxxx|XXXX)[\s\-](xxxx|XXXX)[\s\-](xxxx|XXXX)[\s\-](xxxx|XXXX|\d{4})\b',  # XXXX XXXX XXXX XXXX
            'account_prefixed_upper': r'(?i)\b([a-wyz]{3,}x{8,})\b',  # INDIGOXXXXXXXX - préfixe lettres (sans x) + 8+ X
            'account_16_exact': r'(?i)\b(xxxxxxxxxxxxxxxx|XXXXXXXXXXXXXXXX)\b',  # Exactement 16 X (sans capturer "code")
            'account_real': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # 1234-5678-9012-3456
            
            # (Ces patterns sont déjà définis plus haut, on les supprime ici pour éviter les doublons)
            
            # === CODES ET RÉFÉRENCES ===
            'ref_code': r'(?i)\b[A-Z]{3,}[\-]\d{4}[\-]\d{4}\b',  # USC-6802-6805 (case insensitive maintenant)
            'ref_mixed': r'(?i)\b(xx?xx?|XX?XX?)\s*\|\s*(xx|XX)[/\-](xx|XX)[/\-](\d{4}|(xx?xx?|XX?XX?))\b',  # XXXX | XX/XX/XXXX
            
            # === ANNÉES ISOLÉES (traiter AVANT les patterns génériques) ===
            # Détecte XXXX quand précédé de mots clés temporels - CAPTURE LE MOT CLÉ AUSSI
            'year_with_since': r'(?i)\b(since\s+)(xxxx|XXXX)\b',  # Since XXXX
            'year_with_in': r'(?i)\b(in\s+)(xxxx|XXXX)\b',  # In XXXX
            'year_with_from': r'(?i)\b(from\s+)(xxxx|XXXX)\b',  # From XXXX
            'year_with_until': r'(?i)\b(until\s+)(xxxx|XXXX)\b',  # Until XXXX
            'year_with_before': r'(?i)\b(before\s+)(xxxx|XXXX)\b',  # Before XXXX
            'year_with_after': r'(?i)\b(after\s+)(xxxx|XXXX)\b',  # After XXXX
            'year_with_during': r'(?i)\b(during\s+)(xxxx|XXXX)\b',  # During XXXX
            'year_with_year': r'(?i)\b(year\s+)(xxxx|XXXX)\b',  # Year XXXX
            'year_real': r'\b(19|20)\d{2}\b',  # 1977, 2022
            
            # === PATTERNS GÉNÉRIQUES (traiter en dernier) ===
            # Patterns de X multiples - case insensitive
            # Pattern "XXXX XXXX" (deux groupes seulement) - AVANT les patterns individuels
            'x_paired': r'(?i)\b(xxxx|XXXX)[\s,;]+?(xxxx|XXXX)\b(?![\s\-](xxxx|XXXX))',  # XXXX XXXX mais pas XXXX XXXX XXXX
            # Les patterns longs d'abord (mais après x_paired)
            'x_12': r'(?i)\b(xxxxxxxxxxxx|XXXXXXXXXXXX)\b',  # Exactement 12 X
            'x_8': r'(?i)\b(xxxxxxxx|XXXXXXXX)\b',  # Exactement 8 X
            'x_6': r'(?i)\b(xxxxxx|XXXXXX)\b',  # Exactement 6 X
            'x_5': r'(?i)\b(xxxxx|XXXXX)\b',  # Exactement 5 X
            'x_4': r'(?i)\b(xxxx|XXXX)\b',  # Exactement 4 X isolés
            'x_generic': r'(?i)\b(x{3,}|X{3,})\b',  # 3 X ou plus
            
            # === EMAILS ET URLS ===
            'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            'urls': r'(?:https?://|www\.)[^\s]+',
            
            # === NETTOYAGE FINAL ===
            'extra_spaces': r'\s+',
            'special_chars': r'[^\w\s.,!?;:\'\"-]'  # Garde seulement certains caractères spéciaux
        }
        
        # Tokens de remplacement correspondants
        self.replacements = {
            # SSN (en premier)
            'ssn_anon': '[SSN]',
            'ssn_partial': '[SSN]',
            'ssn_real': '[SSN]',
            
            # Téléphones
            'phone_anon_full': '[PHONE]',
            'phone_anon_with_parens': '[PHONE]',
            'phone_anon_partial': '[PHONE]',
            'phone_partial_real': '[PHONE]',
            'phone_real': '[PHONE]',
            
            # Dates
            'dates_dash_year': '[DATE]',
            'dates_full_anon': '[DATE]',
            'dates_partial_anon': '[DATE]',
            'dates_year_first': '[DATE]',
            'dates_real': '[DATE]',
            
            # Montants
            'money_brackets': '[AMOUNT]',
            'money_normal': '[AMOUNT]',
            
            # Comptes
            'account_long': '[ACCOUNT]',
            'account_card_full': '[ACCOUNT]',
            'account_prefixed_upper': '[ACCOUNT]',
            'account_16_exact': '[ACCOUNT]',
            'account_real': '[ACCOUNT]',
            
            # Références
            'ref_code': '[REF]',
            'ref_mixed': '[ID] | [DATE]',
            
            # Années
            'year_with_since': r'\1[YEAR]',  # Since XXXX → Since [YEAR] (garde "since ")
            'year_with_in': r'\1[YEAR]',  # In XXXX → In [YEAR] (garde "in ")
            'year_with_from': r'\1[YEAR]',  # From XXXX → From [YEAR]
            'year_with_until': r'\1[YEAR]',  # Until XXXX → Until [YEAR]
            'year_with_before': r'\1[YEAR]',  # Before XXXX → Before [YEAR]
            'year_with_after': r'\1[YEAR]',  # After XXXX → After [YEAR]
            'year_with_during': r'\1[YEAR]',  # During XXXX → During [YEAR]
            'year_with_year': r'\1[YEAR]',  # Year XXXX → Year [YEAR]
            'year_real': '[YEAR]',
            
            # Patterns X génériques
            'x_paired': '[ID] [ID]',  # XXXX XXXX → [ID] [ID]
            'x_12': '[ID]',
            'x_8': '[ID]',
            'x_6': '[ID]',
            'x_5': '[ID]',
            'x_4': '[ID]',
            'x_generic': '[ID]',
            
            # Autres
            'email': '[EMAIL]',
            'urls': '[URL]'
        }
    
    def clean_text(self, text):
        """
        Nettoie et normalise le texte
        
        Args:
            text (str): Texte brut à nettoyer
            
        Returns:
            str: Texte nettoyé
        """
        if pd.isna(text):
            return ""
        
        # Convertir en string si nécessaire
        text = str(text)
        
        # Mettre en minuscules
        text = text.lower()
        
        # Remplacer les patterns identifiés
        for pattern_name, pattern in self.patterns.items():
            if pattern_name in self.replacements:
                text = re.sub(pattern, self.replacements[pattern_name], text)
        
        # Nettoyer les espaces multiples
        text = re.sub(self.patterns['extra_spaces'], ' ', text)
        
        # Supprimer les espaces en début et fin
        text = text.strip()
        
        return text
    
    def validate_text(self, text):
        """
        Valide que le texte est utilisable pour l'entraînement
        
        Args:
            text (str): Texte à valider
            
        Returns:
            bool: True si le texte est valide
        """
        if not text or len(text) < MIN_TEXT_LENGTH:
            return False
        
        if len(text) > MAX_TEXT_LENGTH:
            return False
        
        # Vérifier qu'il reste du contenu après nettoyage
        # (pas seulement des tokens de remplacement)
        tokens_count = sum(1 for token in self.replacements.values() if token in text)
        words_count = len(text.split())
        
        # Au moins 50% du texte doit être du contenu réel
        if tokens_count > words_count * 0.5:
            return False
        
        return True


def process_chunk(chunk, preprocessor, stats):
    """
    Traite un chunk de données
    
    Args:
        chunk (DataFrame): Chunk de données à traiter
        preprocessor (TextPreprocessor): Instance du préprocesseur
        stats (dict): Dictionnaire des statistiques
        
    Returns:
        DataFrame: Chunk traité
    """
    processed_chunk = []
    
    for _, row in chunk.iterrows():
        # Nettoyer le texte
        cleaned_text = preprocessor.clean_text(row['Consumer complaint narrative'])
        
        # Valider le texte
        if preprocessor.validate_text(cleaned_text):
            processed_chunk.append({
                'product': row['Product'],
                'complaint_text': cleaned_text,
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split())
            })
            stats['kept'] += 1
        else:
            stats['removed'] += 1
            if cleaned_text:
                stats['removed_reasons']['invalid_length'] += 1
            else:
                stats['removed_reasons']['empty'] += 1
    
    return pd.DataFrame(processed_chunk)


def main():
    """Fonction principale de prétraitement"""
    
    logger.info("=" * 80)
    logger.info("DÉBUT DU PRÉTRAITEMENT DES DONNÉES")
    logger.info("=" * 80)
    
    # Vérifier que le fichier source existe
    if not RAW_DATA_PATH.exists():
        logger.error(f"Fichier source introuvable: {RAW_DATA_PATH}")
        sys.exit(1)
    
    # Créer le dossier processed si nécessaire
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialiser le préprocesseur et les statistiques
    preprocessor = TextPreprocessor()
    stats = {
        'start_time': datetime.now().isoformat(),
        'total_rows': 0,
        'kept': 0,
        'removed': 0,
        'removed_reasons': {
            'empty': 0,
            'invalid_length': 0
        },
        'class_distribution': {},
        'text_stats': {
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0,
            'min_words': float('inf'),
            'max_words': 0,
            'avg_words': 0
        }
    }
    
    # Traiter les données par chunks
    logger.info(f"Lecture du fichier: {RAW_DATA_PATH}")
    logger.info(f"Traitement par chunks de {CHUNK_SIZE} lignes")
    
    processed_chunks = []
    chunk_count = 0
    
    # Utiliser tqdm pour afficher la progression
    with tqdm(desc="Traitement des chunks", unit="chunk") as pbar:
        for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE):
            chunk_count += 1
            stats['total_rows'] += len(chunk)
            
            # Traiter le chunk
            processed_chunk = process_chunk(chunk, preprocessor, stats)
            
            if not processed_chunk.empty:
                processed_chunks.append(processed_chunk)
            
            # Mettre à jour la progression
            pbar.update(1)
            pbar.set_postfix({
                'Total': stats['total_rows'],
                'Conservés': stats['kept'],
                'Supprimés': stats['removed']
            })
            
            # Log périodique
            if chunk_count % 10 == 0:
                logger.info(f"Chunks traités: {chunk_count} | "
                          f"Lignes: {stats['total_rows']:,} | "
                          f"Conservées: {stats['kept']:,}")
    
    logger.info(f"\nTotal chunks traités: {chunk_count}")
    
    # Concaténer tous les chunks traités
    if processed_chunks:
        logger.info("Concaténation des données traitées...")
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        
        # Calculer les statistiques finales
        logger.info("Calcul des statistiques...")
        
        # Distribution des classes
        class_counts = df_processed['product'].value_counts()
        stats['class_distribution'] = {
            str(product): int(count) 
            for product, count in class_counts.items()
        }
        stats['num_classes'] = len(class_counts)
        
        # Statistiques du texte
        stats['text_stats']['min_length'] = int(df_processed['text_length'].min())
        stats['text_stats']['max_length'] = int(df_processed['text_length'].max())
        stats['text_stats']['avg_length'] = float(df_processed['text_length'].mean())
        stats['text_stats']['min_words'] = int(df_processed['word_count'].min())
        stats['text_stats']['max_words'] = int(df_processed['word_count'].max())
        stats['text_stats']['avg_words'] = float(df_processed['word_count'].mean())
        
        # Supprimer les colonnes temporaires
        df_processed = df_processed.drop(columns=['text_length', 'word_count'])
        
        # Sauvegarder les données traitées
        logger.info(f"Sauvegarde des données dans: {PROCESSED_DATA_PATH}")
        df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
        
        # Ajouter les statistiques finales
        stats['end_time'] = datetime.now().isoformat()
        stats['output_file'] = str(PROCESSED_DATA_PATH)
        stats['retention_rate'] = round(stats['kept'] / stats['total_rows'] * 100, 2)
        
        # Sauvegarder les statistiques
        with open(STATS_PATH, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Afficher le résumé
        logger.info("\n" + "=" * 80)
        logger.info("RÉSUMÉ DU PRÉTRAITEMENT")
        logger.info("=" * 80)
        logger.info(f"📊 Statistiques générales:")
        logger.info(f"  • Lignes totales traitées: {stats['total_rows']:,}")
        logger.info(f"  • Lignes conservées: {stats['kept']:,} ({stats['retention_rate']}%)")
        logger.info(f"  • Lignes supprimées: {stats['removed']:,}")
        logger.info(f"    - Vides: {stats['removed_reasons']['empty']:,}")
        logger.info(f"    - Longueur invalide: {stats['removed_reasons']['invalid_length']:,}")
        
        logger.info(f"\n📝 Statistiques du texte:")
        logger.info(f"  • Longueur moyenne: {stats['text_stats']['avg_length']:.0f} caractères")
        logger.info(f"  • Longueur min/max: {stats['text_stats']['min_length']}/{stats['text_stats']['max_length']}")
        logger.info(f"  • Mots moyens: {stats['text_stats']['avg_words']:.0f}")
        logger.info(f"  • Mots min/max: {stats['text_stats']['min_words']}/{stats['text_stats']['max_words']}")
        
        logger.info(f"\n📁 Fichiers générés:")
        logger.info(f"  • Données: {PROCESSED_DATA_PATH}")
        logger.info(f"  • Statistiques: {STATS_PATH}")
        
        logger.info("\n✅ Prétraitement terminé avec succès!")
        
    else:
        logger.error("Aucune donnée valide après prétraitement!")
        sys.exit(1)


if __name__ == "__main__":
    main()