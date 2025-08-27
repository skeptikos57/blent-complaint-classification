#!/usr/bin/env python3
"""
Script pour analyser la distribution des classes dans complaints.csv
Extrait toutes les catégories (Product) et compte le nombre d'observations
"""

import pandas as pd
import sys
from pathlib import Path
import json
import os
from dotenv import load_dotenv


def analyze_class_distribution(filename='data/raw/complaints.csv', chunk_size=10000):
    """
    Analyse la distribution des classes dans le fichier CSV
    
    Args:
        filename (str): Nom du fichier CSV à analyser
        chunk_size (int): Taille des chunks pour la lecture (défaut 10000)
    
    Returns:
        dict: Distribution des classes avec counts et pourcentages
    """
    filepath = Path(filename)
    
    if not filepath.exists():
        print(f"Erreur: Le fichier {filename} n'existe pas")
        return None
    
    print(f"Analyse de la distribution des classes dans {filename}...")
    print(f"Lecture par chunks de {chunk_size} lignes\n")
    print("-" * 80)
    
    # Dictionnaire pour stocker les comptages
    class_counts = {}
    total_rows = 0
    chunks_processed = 0
    
    try:
        # Lecture par chunks pour efficacité mémoire
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=['Product']):
            chunks_processed += 1
            
            # Compter les occurrences dans ce chunk
            for product, count in chunk['Product'].value_counts().items():
                class_counts[product] = class_counts.get(product, 0) + count
                total_rows += count
            
            # Affichage progression
            if chunks_processed % 10 == 0:
                print(f"Chunks traités: {chunks_processed} | Lignes analysées: {total_rows:,}")
        
        print(f"\nAnalyse terminée!")
        print(f"Total chunks traités: {chunks_processed}")
        print(f"Total observations: {total_rows:,}")
        print("-" * 80)
        
        # Calculer les statistiques
        results = calculate_statistics(class_counts, total_rows)
        
        # Afficher les résultats
        display_results(results)
        
        # Sauvegarder les résultats
        save_results(results)
        
        return results
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        return None


def calculate_statistics(class_counts, total_rows):
    """
    Calcule les statistiques pour chaque classe
    
    Args:
        class_counts (dict): Comptages par classe
        total_rows (int): Nombre total d'observations
    
    Returns:
        dict: Statistiques détaillées
    """
    # Trier par nombre d'observations (décroissant)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    results = {
        'total_observations': total_rows,
        'nombre_classes': len(class_counts),
        'classes': []
    }
    
    for class_name, count in sorted_classes:
        percentage = (count / total_rows) * 100
        weight_alpha = count / total_rows  # αᵢ pour le Weighted F1 Score
        
        results['classes'].append({
            'nom': class_name,
            'nombre_observations': count,
            'pourcentage': round(percentage, 2),
            'poids_alpha': round(weight_alpha, 4)
        })
    
    return results


def display_results(results):
    """
    Affiche les résultats de l'analyse
    
    Args:
        results (dict): Résultats de l'analyse
    """
    print("\n" + "=" * 100)
    print("RÉSULTATS DE L'ANALYSE")
    print("=" * 100)
    
    print(f"\n📊 Statistiques générales:")
    print(f"  • Nombre total d'observations: {results['total_observations']:,}")
    print(f"  • Nombre de classes distinctes: {results['nombre_classes']}")
    
    # Afficher TOUTES les classes
    print(f"\n📈 Distribution complète des {results['nombre_classes']} classes:")
    print(f"\n{'Rang':<5} {'Classe':<55} {'Observations':<15} {'%':<10} {'αᵢ':<10}")
    print("-" * 95)
    
    for i, classe in enumerate(results['classes'], 1):
        nom = classe['nom'][:52] + "..." if len(classe['nom']) > 55 else classe['nom']
        print(f"{i:<5} {nom:<55} {classe['nombre_observations']:<15,} {classe['pourcentage']:<10.3f} {classe['poids_alpha']:<10.5f}")
    
    print("-" * 95)
    
    # Statistiques sur la distribution
    print("\n📉 Analyse de la distribution:")
    counts = [c['nombre_observations'] for c in results['classes']]
    
    # Classe la plus représentée
    max_class = results['classes'][0]
    print(f"\n  • Classe majoritaire:")
    print(f"    - Nom: {max_class['nom']}")
    print(f"    - Observations: {max_class['nombre_observations']:,} ({max_class['pourcentage']:.2f}%)")
    
    # Classe la moins représentée
    min_class = results['classes'][-1]
    print(f"\n  • Classe minoritaire:")
    print(f"    - Nom: {min_class['nom']}")
    print(f"    - Observations: {min_class['nombre_observations']:,} ({min_class['pourcentage']:.2f}%)")
    
    # Moyenne et médiane
    import statistics
    mean_count = statistics.mean(counts)
    median_count = statistics.median(counts)
    std_count = statistics.stdev(counts) if len(counts) > 1 else 0
    
    print(f"\n  • Statistiques de distribution:")
    print(f"    - Moyenne d'observations par classe: {mean_count:,.0f}")
    print(f"    - Médiane d'observations par classe: {median_count:,.0f}")
    print(f"    - Écart-type: {std_count:,.0f}")
    
    # Déséquilibre des classes
    imbalance_ratio = max_class['nombre_observations'] / min_class['nombre_observations']
    print(f"    - Ratio déséquilibre (max/min): {imbalance_ratio:.1f}x")
    
    # Classes avec moins de 1% des données
    small_classes = [c for c in results['classes'] if c['pourcentage'] < 1.0]
    if small_classes:
        print(f"\n  • Classes minoritaires (<1% des données): {len(small_classes)} classes")
        for c in small_classes[:5]:  # Afficher les 5 premières
            print(f"    - {c['nom'][:40]}: {c['nombre_observations']} obs ({c['pourcentage']:.3f}%)")
        if len(small_classes) > 5:
            print(f"    ... et {len(small_classes) - 5} autres")


def save_results(results):
    """
    Sauvegarde les résultats dans un fichier JSON
    
    Args:
        results (dict): Résultats à sauvegarder
    """
    try:
        # Définir le chemin de sauvegarde
        output_file = Path('data/prepared/analyse_result.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Résultats JSON sauvegardés dans: {output_file}")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde: {e}")


def main():
    """Fonction principale"""
    # Charger les variables d'environnement
    load_dotenv()
    
    # Récupérer le chemin du fichier depuis les variables d'environnement
    filename = os.getenv('INPUT_FILE', 'data/raw/complaints.csv')
    chunk_size = 10000
    
    # Récupérer les arguments si fournis (permet d'override les valeurs par défaut)
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            chunk_size = int(sys.argv[2])
        except ValueError:
            print("La taille de chunk doit être un entier")
            sys.exit(1)
    
    # Analyser la distribution
    results = analyze_class_distribution(filename, chunk_size)
    
    if results:
        print("\n" + "=" * 100)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("=" * 100)
        print(f"\n📌 Informations importantes pour le projet:")
        print(f"  • {results['nombre_classes']} classes à prédire")
        print(f"  • Les poids αᵢ ont été calculés pour le Weighted F1 Score")
        print(f"  • Attention au déséquilibre des classes lors de l'entraînement")
        print(f"  • Considérer des techniques de rééquilibrage (SMOTE, class_weight, etc.)")


if __name__ == "__main__":
    main()