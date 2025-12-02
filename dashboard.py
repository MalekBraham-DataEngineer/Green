import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Espaces Verts Parisiens",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2E8B57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2E8B57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .model-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .accessibility-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class GreenSpaceAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
        
    def prepare_data(self):
        """Préparation complète des données"""
        # Nettoyage de base
        self.df = self.df[self.df['annee_ouverture'] > 1600]
        
        # Gestion des valeurs manquantes
        numeric_cols = ['surface_totale_reelle', 'surface_horticole', 'nb_entites', 
                       'perimeter', 'poly_area', 'densite_espaces', 'ratio_horticole']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
        
        # Calculs dérivés si manquants
        if 'densite_espaces' not in self.df.columns or self.df['densite_espaces'].sum() == 0:
            self.df['densite_espaces'] = np.where(
                self.df['surface_totale_reelle'] > 0,
                self.df['nb_entites'] / self.df['surface_totale_reelle'],
                0
            )
        
        if 'ratio_horticole' not in self.df.columns or self.df['ratio_horticole'].sum() == 0:
            self.df['ratio_horticole'] = np.where(
                self.df['surface_totale_reelle'] > 0,
                self.df['surface_horticole'] / self.df['surface_totale_reelle'],
                0
            )
        
        # ================================================
        #  Utiliser directement taille_categorie existante
        # ================================================
        if 'taille_categorie' in self.df.columns:
            # Filtrer uniquement les catégories Petit, Moyen, Grand
            valid_categories = ['Petit', 'Moyen', 'Grand']
            self.df = self.df[self.df['taille_categorie'].isin(valid_categories)]
            # Créer une copie pour compatibilité
            self.df['categorie_taille'] = self.df['taille_categorie']
        
        # Décennie pour analyse temporelle
        self.df['decade'] = (self.df['annee_ouverture'] // 10) * 10
        
        # Classification par densité
        if 'densite_espaces' in self.df.columns:
            median_densite = self.df['densite_espaces'].median()
            self.df['type_amenagement'] = np.where(
                self.df['densite_espaces'] > median_densite,
                'Haute densité équipements',
                'Faible densité équipements'
            )
        
        # Score de qualité composite (0-100)
        self.calculate_quality_score()
        
        # Créer un identifiant unique pour chaque espace
        if 'nom_ev' not in self.df.columns:
            # Utiliser une combinaison d'arrondissement et d'index
            self.df['nom_ev'] = [
                f"Espace_{int(row['arrondissement'] if pd.notna(row['arrondissement']) else 0)}_{i+1}" 
                for i, (_, row) in enumerate(self.df.iterrows())
            ]
    
    def calculate_quality_score(self):
        """Calcul d'un score de qualité composite"""
        # Normalisation des métriques
        scaler = StandardScaler()
        
        features = ['surface_totale_reelle', 'densite_espaces', 'ratio_horticole']
        valid_features = [f for f in features if f in self.df.columns]
        
        if len(valid_features) > 0:
            # Remplacer les inf par 0
            for col in valid_features:
                self.df[col] = self.df[col].replace([np.inf, -np.inf], 0)
            
            # Filtrer les valeurs NaN
            data_for_scaling = self.df[valid_features].fillna(0)
            
            # Vérifier qu'on a des données
            if len(data_for_scaling) > 0 and not data_for_scaling.isnull().all().all():
                normalized = scaler.fit_transform(data_for_scaling)
                
                # Poids : surface (30%), densité (40%), ratio horticole (30%)
                weights = [0.3, 0.4, 0.3][:len(valid_features)]
                weights = np.array(weights) / sum(weights)  # Normaliser les poids
                
                # Score composite
                scores = normalized @ weights
                
                # Normaliser entre 0 et 100
                min_score = scores.min()
                max_score = scores.max()
                
                if max_score > min_score:  # Éviter division par zéro
                    self.df['score_qualite'] = (
                        (scores - min_score) / (max_score - min_score) * 100
                    )
                else:
                    self.df['score_qualite'] = 50  # Score neutre
            else:
                self.df['score_qualite'] = 50
        else:
            self.df['score_qualite'] = 50  # Score neutre par défaut
    
    def get_key_metrics(self):
        """Métriques clés globales"""
        metrics = {
            'total_espaces': len(self.df),
            'surface_totale': 0,
            'surface_moyenne': 0,
            'surface_mediane': 0,
            'densite_moyenne': 0,
            'ratio_horticole_moyen': 0,
            'age_moyen': 0,
            'score_qualite_moyen': 0
        }
        
        if 'surface_totale_reelle' in self.df.columns:
            metrics['surface_totale'] = self.df['surface_totale_reelle'].sum() / 10000  # en hectares
            metrics['surface_moyenne'] = self.df['surface_totale_reelle'].mean()
            metrics['surface_mediane'] = self.df['surface_totale_reelle'].median()
        
        if 'densite_espaces' in self.df.columns:
            # Exclure les valeurs infinies pour le calcul
            densite_valide = self.df[self.df['densite_espaces'] < np.inf]['densite_espaces']
            if len(densite_valide) > 0:
                metrics['densite_moyenne'] = densite_valide.mean() * 10000  # par hectare
        
        if 'ratio_horticole' in self.df.columns:
            ratio_valide = self.df[self.df['ratio_horticole'] <= 1]['ratio_horticole']
            if len(ratio_valide) > 0:
                metrics['ratio_horticole_moyen'] = ratio_valide.mean() * 100
        
        if 'age_espace' in self.df.columns:
            metrics['age_moyen'] = self.df['age_espace'].mean()
        
        if 'score_qualite' in self.df.columns:
            metrics['score_qualite_moyen'] = self.df['score_qualite'].mean()
        
        return metrics
    
    def analyze_by_arrondissement(self):
        """Analyse détaillée par arrondissement"""
        if 'arrondissement' not in self.df.columns:
            return None
            
        # Définir les colonnes d'agrégation disponibles
        agg_dict = {}
        
        if 'surface_totale_reelle' in self.df.columns:
            agg_dict['surface_totale_reelle'] = ['count', 'sum', 'mean', 'median']
        
        if 'densite_espaces' in self.df.columns:
            agg_dict['densite_espaces'] = 'mean'
        
        if 'ratio_horticole' in self.df.columns:
            agg_dict['ratio_horticole'] = 'mean'
        
        if 'age_espace' in self.df.columns:
            agg_dict['age_espace'] = 'mean'
        
        if 'score_qualite' in self.df.columns:
            agg_dict['score_qualite'] = 'mean'
        
        if 'nb_entites' in self.df.columns:
            agg_dict['nb_entites'] = 'sum'
        
        if not agg_dict:
            return None
        
        arrond_stats = self.df.groupby('arrondissement').agg(agg_dict).round(2)
        arrond_stats.columns = ['_'.join(col).strip() for col in arrond_stats.columns.values]
        
        # Renommer pour clarté
        rename_dict = {}
        if 'surface_totale_reelle_count' in arrond_stats.columns:
            rename_dict['surface_totale_reelle_count'] = 'nombre_espaces'
        if 'surface_totale_reelle_sum' in arrond_stats.columns:
            rename_dict['surface_totale_reelle_sum'] = 'surface_totale_m2'
        if 'surface_totale_reelle_mean' in arrond_stats.columns:
            rename_dict['surface_totale_reelle_mean'] = 'surface_moyenne_m2'
        if 'surface_totale_reelle_median' in arrond_stats.columns:
            rename_dict['surface_totale_reelle_median'] = 'surface_mediane_m2'
        if 'densite_espaces_mean' in arrond_stats.columns:
            rename_dict['densite_espaces_mean'] = 'densite_moyenne'
        if 'ratio_horticole_mean' in arrond_stats.columns:
            rename_dict['ratio_horticole_mean'] = 'ratio_horticole_moyen'
        if 'age_espace_mean' in arrond_stats.columns:
            rename_dict['age_espace_mean'] = 'age_moyen'
        if 'score_qualite_mean' in arrond_stats.columns:
            rename_dict['score_qualite_mean'] = 'score_qualite'
        if 'nb_entites_sum' in arrond_stats.columns:
            rename_dict['nb_entites_sum'] = 'total_equipements'
        
        arrond_stats = arrond_stats.rename(columns=rename_dict)
        
        # Trier par surface totale
        if 'surface_totale_m2' in arrond_stats.columns:
            arrond_stats = arrond_stats.sort_values('surface_totale_m2', ascending=False)
        
        return arrond_stats
    
    def temporal_evolution(self):
        """Évolution temporelle avec indicateurs clés"""
        if 'annee_ouverture' not in self.df.columns:
            return pd.DataFrame()
        
        # Préparer les colonnes d'agrégation
        agg_dict = {'surface_totale_reelle': ['count', 'sum']}
        
        if 'surface_totale_reelle' in self.df.columns:
            agg_dict['surface_totale_reelle'] = ['count', 'sum', 'mean']
        
        if 'densite_espaces' in self.df.columns:
            agg_dict['densite_espaces'] = 'mean'
        
        if 'ratio_horticole' in self.df.columns:
            agg_dict['ratio_horticole'] = 'mean'
        
        if 'score_qualite' in self.df.columns:
            agg_dict['score_qualite'] = 'mean'
        
        yearly = self.df.groupby('annee_ouverture').agg(agg_dict).reset_index()
        
        # Gérer les noms de colonnes
        if isinstance(yearly.columns, pd.MultiIndex):
            yearly.columns = ['_'.join(col).strip() for col in yearly.columns.values]
        
        # Renommer les colonnes
        rename_map = {}
        for col in yearly.columns:
            if 'surface_totale_reelle_count' in col:
                rename_map[col] = 'nombre'
            elif 'surface_totale_reelle_sum' in col:
                rename_map[col] = 'surface_totale'
            elif 'surface_totale_reelle_mean' in col:
                rename_map[col] = 'surface_moyenne'
            elif 'densite_espaces_mean' in col:
                rename_map[col] = 'densite_moyenne'
            elif 'ratio_horticole_mean' in col:
                rename_map[col] = 'ratio_horticole'
            elif 'score_qualite_mean' in col:
                rename_map[col] = 'score_qualite'
            elif 'annee_ouverture' in col:
                rename_map[col] = 'annee'
        
        yearly = yearly.rename(columns=rename_map)
        
        # Assurer que les colonnes nécessaires existent
        if 'nombre' in yearly.columns:
            yearly['nombre_cumule'] = yearly['nombre'].cumsum()
        if 'surface_totale' in yearly.columns:
            yearly['surface_cumulee'] = yearly['surface_totale'].cumsum()
        
        return yearly
    
    def clustering_analysis(self, n_clusters=4):
        """Clustering des espaces verts par profil"""
        features = ['surface_totale_reelle', 'densite_espaces', 'ratio_horticole', 'age_espace']
        
        # Vérifier quelles features sont disponibles
        available_features = [f for f in features if f in self.df.columns]
        
        if len(available_features) < 2:
            return pd.DataFrame()
        
        # Préparer les données
        X = self.df[available_features].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Transformation logarithmique pour les features positives
        X_log = X.copy()
        for col in ['surface_totale_reelle', 'densite_espaces']:
            if col in X_log.columns and X_log[col].min() >= 0:
                X_log[col] = np.log1p(X_log[col])
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)
        
        # K-Means
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
            self.df['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Profil de chaque cluster
            cluster_summary = self.df.groupby('cluster')[available_features].mean()
            cluster_summary['count'] = self.df.groupby('cluster').size()
            
            return cluster_summary
        except:
            return pd.DataFrame()
    
    def correlation_analysis(self):
        """Matrice de corrélation des variables clés"""
        corr_vars = ['surface_totale_reelle', 'densite_espaces', 'ratio_horticole', 
                     'age_espace', 'nb_entites', 'score_qualite']
        
        available_vars = [v for v in corr_vars if v in self.df.columns]
        
        if len(available_vars) > 1:
            # Remplacer inf et -inf par NaN puis supprimer
            temp_df = self.df[available_vars].replace([np.inf, -np.inf], np.nan)
            temp_df = temp_df.dropna()
            
            if len(temp_df) > 1:
                corr_matrix = temp_df.corr()
                return corr_matrix
        return None
    
    # ================================================
    # Fonction identify_outliers améliorée
    # ================================================
    def identify_outliers(self):
        """Identification des espaces remarquables par arrondissement"""
        outliers = {
            'top_surface': None,
            'top_densite': None,
            'top_qualite': None,
            'low_qualite': None
        }
        
        # 1. Top surface (plus grandes surfaces)
        if 'surface_totale_reelle' in self.df.columns:
            # Exclure les valeurs infinies
            valid_surface = self.df[self.df['surface_totale_reelle'] < np.inf]
            if len(valid_surface) >= 5:
                top_surface_df = valid_surface.nlargest(5, 'surface_totale_reelle')[
                    ['nom_ev', 'arrondissement', 'surface_totale_reelle', 'score_qualite']
                ].copy()
                outliers['top_surface'] = top_surface_df
        
        # 2. Top densité (plus d'équipements par m²)
        if 'densite_espaces' in self.df.columns:
            # Exclure les valeurs infinies et négatives
            valid_densite = self.df[
                (self.df['densite_espaces'] < np.inf) & 
                (self.df['densite_espaces'] >= 0)
            ]
            if len(valid_densite) >= 5:
                top_densite_df = valid_densite.nlargest(5, 'densite_espaces')[
                    ['nom_ev', 'arrondissement', 'densite_espaces', 'score_qualite']
                ].copy()
                outliers['top_densite'] = top_densite_df
        
        # 3. Top qualité (meilleurs scores)
        if 'score_qualite' in self.df.columns:
            # S'assurer que le score est numérique
            self.df['score_qualite'] = pd.to_numeric(self.df['score_qualite'], errors='coerce')
            valid_scores = self.df.dropna(subset=['score_qualite'])
            
            if len(valid_scores) >= 5:
                top_qualite_df = valid_scores.nlargest(5, 'score_qualite')[
                    ['nom_ev', 'arrondissement', 'score_qualite', 'surface_totale_reelle']
                ].copy()
                outliers['top_qualite'] = top_qualite_df
        
        # 4. Basse qualité (scores les plus faibles)
        if 'score_qualite' in self.df.columns:
            valid_scores = self.df.dropna(subset=['score_qualite'])
            
            if len(valid_scores) >= 5:
                low_qualite_df = valid_scores.nsmallest(5, 'score_qualite')[
                    ['nom_ev', 'arrondissement', 'score_qualite', 'age_espace']
                ].copy()
                outliers['low_qualite'] = low_qualite_df
        
        return outliers
    
    def analyze_accessibility(self):
        """Analyse complète de l'accessibilité des espaces verts"""
        accessibility = {
            'total_surface': self.df['surface_totale_reelle'].sum() if 'surface_totale_reelle' in self.df.columns else 0,
            'open_spaces': 0,
            'closed_spaces': 0,
            'unknown_status': 0,
            'fenced_spaces': 0,
            'open_surface': 0,
            'closed_surface': 0,
            'accessibility_rate': 0
        }
        
        if 'ouvert_ferme' in self.df.columns:
            open_df = self.df[self.df['ouvert_ferme'] == 1]
            closed_df = self.df[self.df['ouvert_ferme'] == 0]
            
            accessibility['open_spaces'] = len(open_df)
            accessibility['closed_spaces'] = len(closed_df)
            accessibility['unknown_status'] = len(self.df[self.df['ouvert_ferme'].isna()])
            
            if 'surface_totale_reelle' in open_df.columns:
                accessibility['open_surface'] = open_df['surface_totale_reelle'].sum()
            if 'surface_totale_reelle' in closed_df.columns:
                accessibility['closed_surface'] = closed_df['surface_totale_reelle'].sum()
            
            if accessibility['total_surface'] > 0:
                accessibility['accessibility_rate'] = (
                    accessibility['open_surface'] / accessibility['total_surface'] * 100
                )
        
        # Analyse par clôture si disponible
        if 'presence_cloture' in self.df.columns:
            fence_stats = self.df.groupby('presence_cloture').agg({
                'surface_totale_reelle': ['count', 'sum', 'mean'],
                'score_qualite': 'mean',
                'ratio_horticole': 'mean'
            }).round(2)
            
            fence_stats.columns = ['count', 'total_surface', 'avg_surface', 'avg_score', 'avg_ratio_horticole']
            accessibility['fence_stats'] = fence_stats
        
        return accessibility
    
    # ================================================
    #Fonction analyze_horticultural_ratio améliorée
    # ================================================
    def analyze_horticultural_ratio(self):
        """Analyse détaillée du ratio horticole par arrondissement"""
        horticultural_analysis = {}
        
        # Vérifier que la colonne existe
        if 'ratio_horticole' not in self.df.columns:
            return horticultural_analysis
        
        # Nettoyer les données
        self.df['ratio_horticole'] = pd.to_numeric(self.df['ratio_horticole'], errors='coerce')
        valid_ratio = self.df.dropna(subset=['ratio_horticole'])
        valid_ratio = valid_ratio[valid_ratio['ratio_horticole'] <= 1]  # Ratio entre 0 et 1
        
        if len(valid_ratio) == 0:
            return horticultural_analysis
        
        # Statistiques globales
        horticultural_analysis['global_mean'] = valid_ratio['ratio_horticole'].mean() * 100
        horticultural_analysis['global_median'] = valid_ratio['ratio_horticole'].median() * 100
        horticultural_analysis['global_std'] = valid_ratio['ratio_horticole'].std() * 100
        
        # Analyse par taille (Petit/Moyen/Grand uniquement)
        if 'taille_categorie' in valid_ratio.columns:
            # Filtrer uniquement nos 3 catégories
            taille_filtered = valid_ratio[valid_ratio['taille_categorie'].isin(['Petit', 'Moyen', 'Grand'])]
            
            if len(taille_filtered) > 0:
                size_analysis = taille_filtered.groupby('taille_categorie')['ratio_horticole'].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(3)
                size_analysis[['mean', 'median', 'std']] = size_analysis[['mean', 'median', 'std']] * 100
                horticultural_analysis['by_size'] = size_analysis
        
        # Analyse par arrondissement
        if 'arrondissement' in valid_ratio.columns:
            arrond_analysis = valid_ratio.groupby('arrondissement')['ratio_horticole'].agg([
                'mean', 'median', 'std', 'count'
            ]).round(3)
            arrond_analysis[['mean', 'median', 'std']] = arrond_analysis[['mean', 'median', 'std']] * 100
            horticultural_analysis['by_arrondissement'] = arrond_analysis
        
        # Évolution temporelle
        if 'annee_ouverture' in valid_ratio.columns:
            temporal_analysis = valid_ratio.groupby('annee_ouverture')['ratio_horticole'].mean().reset_index()
            temporal_analysis['ratio_horticole'] = temporal_analysis['ratio_horticole'] * 100
            horticultural_analysis['temporal'] = temporal_analysis
        
        # Meilleurs et pires ratios - PAR ARRONDISSEMENT
        # S'assurer que 'nom_ev' existe
        if 'nom_ev' not in valid_ratio.columns:
            valid_ratio['nom_ev'] = [f'Espace_{i+1}' for i in range(len(valid_ratio))]
        
        # Meilleurs ratios (plus élevés) - par arrondissement
        if len(valid_ratio) >= 5 and 'arrondissement' in valid_ratio.columns:
            # Top ratios globaux
            top_ratio = valid_ratio.nlargest(5, 'ratio_horticole')[
                ['nom_ev', 'arrondissement', 'ratio_horticole', 'surface_totale_reelle']
            ].copy()
            top_ratio['ratio_horticole'] = top_ratio['ratio_horticole'] * 100
            horticultural_analysis['top_ratios'] = top_ratio
            
            # Top par arrondissement (meilleur de chaque arrondissement)
            top_by_arrond = valid_ratio.loc[valid_ratio.groupby('arrondissement')['ratio_horticole'].idxmax()].copy()
            top_by_arrond = top_by_arrond.nlargest(5, 'ratio_horticole')[
                ['nom_ev', 'arrondissement', 'ratio_horticole', 'surface_totale_reelle']
            ]
            top_by_arrond['ratio_horticole'] = top_by_arrond['ratio_horticole'] * 100
            horticultural_analysis['top_by_arrond'] = top_by_arrond
            
            # Pires ratios (plus faibles) - par arrondissement
            # Exclure les ratios nuls ou négatifs
            nonzero_ratio = valid_ratio[valid_ratio['ratio_horticole'] > 0]
            if len(nonzero_ratio) >= 5:
                low_ratio = nonzero_ratio.nsmallest(5, 'ratio_horticole')[
                    ['nom_ev', 'arrondissement', 'ratio_horticole', 'surface_totale_reelle']
                ].copy()
                low_ratio['ratio_horticole'] = low_ratio['ratio_horticole'] * 100
                horticultural_analysis['low_ratios'] = low_ratio
                
                # Pire par arrondissement
                low_by_arrond = nonzero_ratio.loc[nonzero_ratio.groupby('arrondissement')['ratio_horticole'].idxmin()].copy()
                low_by_arrond = low_by_arrond.nsmallest(5, 'ratio_horticole')[
                    ['nom_ev', 'arrondissement', 'ratio_horticole', 'surface_totale_reelle']
                ]
                low_by_arrond['ratio_horticole'] = low_by_arrond['ratio_horticole'] * 100
                horticultural_analysis['low_by_arrond'] = low_by_arrond
        
        return horticultural_analysis
    
    def counter_intuitive_insights(self):
        """Identification de corrélations contre-intuitives"""
        insights = []
        
        # 1. Paradoxe taille vs équipements
        if 'surface_totale_reelle' in self.df.columns and 'densite_espaces' in self.df.columns:
            # Nettoyer les données
            temp_df = self.df[['surface_totale_reelle', 'densite_espaces']].dropna()
            temp_df = temp_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(temp_df) > 2:
                corr_size_density = temp_df['surface_totale_reelle'].corr(temp_df['densite_espaces'])
                if corr_size_density < -0.3:
                    insights.append({
                        'type': 'Paradoxe Taille-Équipements',
                        'finding': f'Corrélation négative ({corr_size_density:.3f})',
                        'interpretation': 'Les GRANDS espaces sont MOINS équipés que les petits !',
                        'implication': 'Favoriser les squares de proximité vs grands parcs ?'
                    })
        
        # 2. Ancien = Mieux entretenu ?
        if 'age_espace' in self.df.columns and 'score_qualite' in self.df.columns:
            temp_df = self.df[['age_espace', 'score_qualite']].dropna()
            if len(temp_df) > 2:
                corr_age_quality = temp_df['age_espace'].corr(temp_df['score_qualite'])
                if corr_age_quality > 0.2:
                    insights.append({
                        'type': 'Effet Patrimoine',
                        'finding': f'Corrélation positive ({corr_age_quality:.3f})',
                        'interpretation': 'Les espaces ANCIENS ont un MEILLEUR score !',
                        'implication': 'Le patrimoine historique est mieux préservé'
                    })
        
        # 3. Loi de Pareto (20/80)
        if 'surface_totale_reelle' in self.df.columns and len(self.df) > 0:
            sorted_df = self.df.sort_values('surface_totale_reelle', ascending=False)
            n_top = max(1, int(len(sorted_df) * 0.2))
            top_20_pct = sorted_df.head(n_top)
            surface_top_20 = top_20_pct['surface_totale_reelle'].sum()
            surface_total = self.df['surface_totale_reelle'].sum()
            
            if surface_total > 0:
                pareto_ratio = (surface_top_20 / surface_total) * 100
                if pareto_ratio > 70:
                    insights.append({
                        'type': 'Concentration Extrême',
                        'finding': f'{pareto_ratio:.1f}% de la surface dans 20% des espaces',
                        'interpretation': 'Distribution très inégale des surfaces',
                        'implication': 'Quelques grands parcs vs multitude de petits espaces'
                    })
        
        # 4. Végétation vs Équipements
        if 'ratio_horticole' in self.df.columns and 'densite_espaces' in self.df.columns:
            temp_df = self.df[['ratio_horticole', 'densite_espaces']].dropna()
            temp_df = temp_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(temp_df) > 2:
                corr_green_equip = temp_df['ratio_horticole'].corr(temp_df['densite_espaces'])
                if corr_green_equip < -0.2:
                    insights.append({
                        'type': 'Trade-off Nature/Équipement',
                        'finding': f'Corrélation négative ({corr_green_equip:.3f})',
                        'interpretation': 'Plus d\'équipements = Moins de végétation !',
                        'implication': 'Arbitrage entre espace ludique et espace contemplatif'
                    })
        
        return insights
    
    def train_taille_models(self, target_col='surface_totale_reelle'):
        """
        Train separate LightGBM models for each taille_categorie
        """
        st.info("🔍 Entraînement des modèles LightGBM par catégorie de taille...")
        
        models = {}
        category_info = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Utiliser les catégories disponibles
        categories = self.df['taille_categorie'].unique()
        
        for i, taille in enumerate(categories):
            status_text.text(f"Entraînement du modèle pour: {taille}")
            progress = (i / len(categories))
            progress_bar.progress(progress)
            
            # Filter data for this taille
            taille_data = self.df[self.df['taille_categorie'] == taille].copy()
            
            if len(taille_data) < 10:
                st.warning(f"⚠️ Données insuffisantes pour {taille}: {len(taille_data)} échantillons")
                continue
            
            # Feature selection based on category
            if taille == 'Petit':
                feature_cols = ['arrondissement', 'annee_ouverture', 'ratio_horticole', 'densite_espaces']
            elif taille == 'Moyen':
                feature_cols = ['arrondissement', 'annee_ouverture', 'ratio_horticole', 
                              'densite_espaces', 'type_ev', 'categorie']
            else:  # Grand
                feature_cols = ['arrondissement', 'annee_ouverture', 'ratio_horticole',
                              'densite_espaces', 'type_ev', 'categorie', 'zone_paris', 'age_espace']
            
            # Only keep features that exist
            feature_cols = [f for f in feature_cols if f in taille_data.columns]
            
            if len(feature_cols) < 2:
                st.warning(f"⚠️ Pas assez de caractéristiques pour {taille}")
                continue
            
            # Prepare features and target
            X = taille_data[feature_cols].copy()
            y = taille_data[target_col].copy()
            
            # Encode categorical variables
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Convert arrondissement to category for LightGBM
            if 'arrondissement' in X.columns:
                X['arrondissement'] = X['arrondissement'].astype('category')
            
            # Time-based split
            X_sorted = X.copy()
            X_sorted['target'] = y.values
            
            if 'annee_ouverture' in X_sorted.columns:
                X_sorted = X_sorted.sort_values('annee_ouverture')
                cutoff_idx = int(len(X_sorted) * 0.8)
                X_train = X_sorted.iloc[:cutoff_idx][feature_cols]
                y_train = X_sorted.iloc[:cutoff_idx]['target']
                X_test = X_sorted.iloc[cutoff_idx:][feature_cols]
                y_test = X_sorted.iloc[cutoff_idx:]['target']
            else:
                # Random split if no time column
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Model parameters based on category
            if taille == 'Petit':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 15,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'min_data_in_leaf': 5,
                    'max_depth': 4,
                }
            elif taille == 'Moyen':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'min_data_in_leaf': 10,
                    'max_depth': 6,
                }
            else:  # Grand
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'min_data_in_leaf': 5,
                    'max_depth': 6,
                }
            
            # Train model
            categorical_cols = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
            test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_cols)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[test_data],
                num_boost_round=300,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Evaluate
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'test_r2': r2_score(y_test, y_pred_test),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            }
            
            # Store model and info
            models[taille] = {
                'model': model,
                'feature_cols': feature_cols,
                'metrics': metrics,
                'label_encoders': label_encoders,
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape
            }
            
            category_info[taille] = {
                'n_samples': len(taille_data),
                'mean': taille_data[target_col].mean(),
                'median': taille_data[target_col].median(),
                'min': taille_data[target_col].min(),
                'max': taille_data[target_col].max(),
                'std': taille_data[target_col].std()
            }
        
        progress_bar.progress(1.0)
        status_text.text("✅ Entraînement des modèles terminé !")
        
        return models, category_info
    
    def forecast_future_by_taille(self, models, category_info, forecast_years=5):
        """
        Forecast surface areas for future years by taille_categorie
        """
        st.info(f"📊 Génération des prévisions pour les {forecast_years} prochaines années...")
        
        # Get current year
        current_year = int(self.df['annee_ouverture'].max())
        
        all_forecasts = []
        
        # Progress bar for forecasting
        forecast_progress = st.progress(0)
        forecast_status = st.empty()
        
        # For each future year
        for year_ahead in range(1, forecast_years + 1):
            forecast_status.text(f"Prévision pour l'année {current_year + year_ahead}")
            forecast_progress.progress(year_ahead / forecast_years)
            
            future_year = current_year + year_ahead
            
            year_forecasts = []
            
            # Estimate NEW spaces that will be created
            for taille in models.keys():
                model_info = models[taille]
                taille_data = self.df[self.df['taille_categorie'] == taille]
                
                if len(taille_data) == 0:
                    continue
                
                # Estimate number of new spaces for this taille
                if 'annee_ouverture' in taille_data.columns:
                    recent_years = sorted(taille_data['annee_ouverture'].unique())[-5:]
                    recent_counts = []
                    for year in recent_years:
                        count = len(taille_data[taille_data['annee_ouverture'] == year])
                        recent_counts.append(count)
                    
                    if len(recent_counts) > 1:
                        avg_recent = np.mean(recent_counts)
                        new_spaces = max(1, int(avg_recent * 0.15))
                    else:
                        new_spaces = max(1, int(len(taille_data) * 0.05))
                else:
                    new_spaces = max(1, int(len(taille_data) * 0.05))
                
                # Limit to reasonable number
                new_spaces = min(new_spaces, 10)
                
                # For each new space, predict its surface
                for space_num in range(new_spaces):
                    # Determine district for new space
                    if 'arrondissement' in taille_data.columns:
                        district_counts = taille_data['arrondissement'].value_counts()
                        if len(district_counts) > 0:
                            district_weights = 1.0 / (district_counts + 1)
                            district_weights = district_weights / district_weights.sum()
                            
                            try:
                                district = np.random.choice(district_counts.index, p=district_weights.values)
                            except:
                                district = district_counts.index[0]
                        else:
                            district = np.random.randint(1, 21)
                    else:
                        district = np.random.randint(1, 21)
                    
                    # Get district characteristics for features
                    district_data = taille_data[taille_data['arrondissement'] == district] if 'arrondissement' in taille_data.columns else taille_data
                    
                    # Prepare feature values
                    feature_values = {}
                    
                    for col in model_info['feature_cols']:
                        if col == 'arrondissement':
                            feature_values[col] = district
                        elif col == 'annee_ouverture':
                            feature_values[col] = future_year
                        elif col in district_data.columns and len(district_data) > 0:
                            if pd.api.types.is_numeric_dtype(district_data[col]):
                                if col in ['ratio_horticole', 'densite_espaces']:
                                    feature_values[col] = district_data[col].median()
                                else:
                                    feature_values[col] = district_data[col].mean()
                            else:
                                mode_values = district_data[col].mode()
                                feature_values[col] = mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown'
                        else:
                            if col == 'ratio_horticole':
                                feature_values[col] = 0.8
                            elif col == 'densite_espaces':
                                feature_values[col] = 0.03
                            elif col == 'age_espace':
                                feature_values[col] = 0
                            else:
                                feature_values[col] = 0
                    
                    # Create prediction DataFrame
                    X_pred = pd.DataFrame([feature_values])
                    
                    # Prepare for prediction
                    if 'arrondissement' in X_pred.columns:
                        X_pred['arrondissement'] = X_pred['arrondissement'].astype('category')
                    
                    # Encode categorical features
                    for col, encoder in model_info.get('label_encoders', {}).items():
                        if col in X_pred.columns:
                            X_pred[col] = X_pred[col].astype(str)
                            if not X_pred[col].iloc[0] in encoder.classes_:
                                X_pred[col] = encoder.classes_[0]
                            else:
                                X_pred[col] = encoder.transform(X_pred[col])
                    
                    # Ensure all required columns exist
                    for col in model_info['feature_cols']:
                        if col not in X_pred.columns:
                            X_pred[col] = 0
                    
                    # Reorder columns
                    X_pred = X_pred[model_info['feature_cols']].copy()
                    
                    # Make prediction
                    try:
                        prediction = model_info['model'].predict(X_pred)[0]
                        
                        # Apply reasonable bounds
                        if taille == 'Petit':
                            prediction = max(10, min(prediction, 1000))
                        elif taille == 'Moyen':
                            prediction = max(1000, min(prediction, 10000))
                        else:  # Grand
                            prediction = max(10000, min(prediction, 500000))
                            
                    except Exception as e:
                        prediction = category_info[taille]['median']
                    
                    year_forecasts.append({
                        'arrondissement': district,
                        'taille_categorie': taille,
                        'annee_creation': future_year,
                        'annee_ouverture': future_year,
                        'predicted_surface': prediction,
                        'forecast_year': year_ahead,
                        'is_new_space': True,
                        'prediction_method': 'LightGBM Model'
                    })
            
            # Add year forecasts to overall
            if year_forecasts:
                year_df = pd.DataFrame(year_forecasts)
                all_forecasts.append(year_df)
        
        forecast_progress.progress(1.0)
        forecast_status.text("✅ Prévisions générées avec succès !")
        
        # Combine all forecasts
        if all_forecasts:
            final_forecasts = pd.concat(all_forecasts, ignore_index=True)
            
            # Create a clean forecast-only DataFrame
            forecast_only = final_forecasts[final_forecasts['annee_creation'] > current_year].copy()
            
            return final_forecasts, forecast_only
        
        return None, None

def load_data():
    """Chargement des données"""
    try:
        df = pd.read_csv("cleandataset.csv")
        st.sidebar.success(f"✅ Dataset chargé : {len(df)} lignes")
        return df
    except FileNotFoundError:
        st.error("❌ Fichier 'cleandataset.csv' introuvable")
        st.info("💡 Assurez-vous que le fichier est dans le même répertoire")
        return None
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        return None

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🌳 Analyse des Espaces Verts Parisiens</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Dashboard d\'analyse exploratoire et prédictive avec LightGBM</p>', unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    if df is None:
        return
    
    # Initialisation de l'analyseur
    try:
        analyzer = GreenSpaceAnalyzer(df)
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation : {e}")
        return
    
    # ==================== SIDEBAR ====================
    st.sidebar.title("🎛️ Contrôles du Dashboard")
    st.sidebar.markdown("---")
    
    # Filtres
    st.sidebar.subheader("📊 Filtres")
    
    # Filtre par année
    if 'annee_ouverture' in analyzer.df.columns:
        min_year = int(analyzer.df['annee_ouverture'].min())
        max_year = int(analyzer.df['annee_ouverture'].max())
        year_range = st.sidebar.slider(
            "Période d'ouverture",
            min_year, max_year, (min_year, max_year)
        )
        analyzer.df = analyzer.df[
            (analyzer.df['annee_ouverture'] >= year_range[0]) & 
            (analyzer.df['annee_ouverture'] <= year_range[1])
        ]
    
    # Filtre par arrondissement
    if 'arrondissement' in analyzer.df.columns:
        arrondissements = sorted(analyzer.df['arrondissement'].dropna().unique().tolist())
        selected_arrond = st.sidebar.multiselect(
            "Arrondissements",
            options=arrondissements,
            default=arrondissements
        )
        if selected_arrond:
            analyzer.df = analyzer.df[analyzer.df['arrondissement'].isin(selected_arrond)]
    
    # Filtre par taille (UNIQUEMENT Petit/Moyen/Grand)
    if 'taille_categorie' in analyzer.df.columns:
        tailles_valides = ['Petit', 'Moyen', 'Grand']
        selected_tailles = st.sidebar.multiselect(
            "Catégories de taille",
            options=tailles_valides,
            default=tailles_valides
        )
        if selected_tailles:
            analyzer.df = analyzer.df[analyzer.df['taille_categorie'].isin(selected_tailles)]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📈 **{len(analyzer.df)}** espaces verts affichés")
    
    # ==================== PRÉDICTIONS AVANCÉES ====================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Paramètres de Prédiction")
    
    forecast_years = st.sidebar.slider(
        "Années à prédire",
        min_value=1,
        max_value=10,
        value=5,
        help="Nombre d'années futures à prédire"
    )
    
    run_predictions = st.sidebar.button("🚀 Lancer les Prédictions LightGBM", type="primary")
    
    # ==================== MÉTRIQUES CLÉS ====================
    st.markdown('<h2 class="section-header">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
    
    metrics = analyzer.get_key_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Nombre d'Espaces",
            f"{metrics['total_espaces']:,}",
            help="Nombre total d'espaces verts"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Surface Totale",
            f"{metrics['surface_totale']:.1f} ha",
            help="Surface cumulée en hectares"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Densité Moyenne",
            f"{metrics['densite_moyenne']:.2f}",
            help="Équipements par hectare"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Score Qualité Moyen",
            f"{metrics['score_qualite_moyen']:.1f}/100",
            help="Score composite de qualité"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Surface Médiane",
            f"{metrics['surface_mediane']:.0f} m²",
            help="Surface médiane des espaces"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Ratio Horticole",
            f"{metrics['ratio_horticole_moyen']:.1f}%",
            help="% moyen de surface horticole"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col7:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Âge Moyen",
            f"{metrics['age_moyen']:.0f} ans",
            help="Âge moyen des espaces verts"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col8:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'annee_ouverture' in analyzer.df.columns:
            oldest_year = analyzer.df['annee_ouverture'].min()
            st.metric(
                "Plus Ancien",
                f"{int(oldest_year)}",
                help="Année du plus ancien espace"
            )
        else:
            st.metric("Plus Ancien", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== ANALYSE PAR ARRONDISSEMENT ====================
    st.markdown('<h2 class="section-header">🗺️ Analyse Géographique</h2>', unsafe_allow_html=True)
    
    arrond_stats = analyzer.analyze_by_arrondissement()
    
    if arrond_stats is not None and len(arrond_stats) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Surface totale par arrondissement
            if 'surface_totale_m2' in arrond_stats.columns:
                fig1 = px.bar(
                    arrond_stats.reset_index(),
                    x='arrondissement',
                    y='surface_totale_m2',
                    title='Surface Totale par Arrondissement',
                    labels={'surface_totale_m2': 'Surface (m²)', 'arrondissement': 'Arrondissement'},
                    color='surface_totale_m2',
                    color_continuous_scale='Greens'
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("⚠️ Données de surface non disponibles")
        
        with col2:
            # Score de qualité par arrondissement
            if 'score_qualite' in arrond_stats.columns:
                fig2 = px.bar(
                    arrond_stats.reset_index(),
                    x='arrondissement',
                    y='score_qualite',
                    title='Score de Qualité par Arrondissement',
                    labels={'score_qualite': 'Score Qualité (/100)', 'arrondissement': 'Arrondissement'},
                    color='score_qualite',
                    color_continuous_scale='RdYlGn'
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("⚠️ Scores de qualité non disponibles")
        
        # Tableau des statistiques
        st.subheader("📋 Statistiques Agrégées par Arrondissement")
        st.dataframe(
            arrond_stats.style.background_gradient(
                subset=['score_qualite'] if 'score_qualite' in arrond_stats.columns else [],
                cmap='RdYlGn'
            ).format({
                'surface_totale_m2': '{:,.0f}',
                'surface_moyenne_m2': '{:,.0f}',
                'densite_moyenne': '{:.4f}',
                'ratio_horticole_moyen': '{:.2%}',
                'score_qualite': '{:.1f}'
            }),
            height=400
        )
    else:
        st.warning("⚠️ Analyse par arrondissement non disponible")
    
    # ==================== ANALYSE TEMPORELLE ====================
    st.markdown('<h2 class="section-header">⏳ Évolution Temporelle</h2>', unsafe_allow_html=True)
    
    temporal_data = analyzer.temporal_evolution()
    
    if not temporal_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Nombre de créations par an
            if 'nombre_cumule' in temporal_data.columns:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=temporal_data['annee'],
                    y=temporal_data['nombre_cumule'],
                    mode='lines',
                    fill='tozeroy',
                    name='Cumul',
                    line=dict(color='#2E8B57', width=3)
                ))
                fig3.update_layout(
                    title='Croissance Cumulative des Espaces Verts',
                    xaxis_title='Année',
                    yaxis_title='Nombre Cumulé d\'Espaces',
                    template='plotly_white',
                    hovermode='x unified'
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("⚠️ Données temporelles non disponibles")
        
        with col2:
            # Évolution de la surface moyenne
            if 'surface_moyenne' in temporal_data.columns:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=temporal_data['annee'],
                    y=temporal_data['surface_moyenne'],
                    mode='lines+markers',
                    name='Surface moyenne',
                    line=dict(color='#FF6B6B', width=2),
                    marker=dict(size=5)
                ))
                fig4.update_layout(
                    title='Évolution de la Surface Moyenne',
                    xaxis_title='Année',
                    yaxis_title='Surface Moyenne (m²)',
                    template='plotly_white',
                    hovermode='x unified'
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("⚠️ Données de surface moyenne non disponibles")
    else:
        st.warning("⚠️ Analyse temporelle non disponible")
    
    # ==================== ANALYSE DU RATIO HORTICOLE ====================
    st.markdown('<h2 class="section-header">🌿 Analyse Détailée du Ratio Horticole</h2>', unsafe_allow_html=True)
    
    horticultural_analysis = analyzer.analyze_horticultural_ratio()
    
    if horticultural_analysis:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Distribution du ratio horticole
            if 'ratio_horticole' in analyzer.df.columns:
                fig_ratio_dist = px.histogram(
                    analyzer.df,
                    x='ratio_horticole',
                    nbins=50,
                    title='Distribution du Ratio Horticole',
                    labels={'ratio_horticole': 'Ratio Horticole', 'count': 'Fréquence'},
                    color_discrete_sequence=['#2E8B57']
                )
                fig_ratio_dist.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_ratio_dist, use_container_width=True)
            else:
                st.info("⚠️ Ratio horticole non disponible")
        
        with col2:
            # Ratio horticole par catégorie de taille
            if 'ratio_horticole' in analyzer.df.columns and 'taille_categorie' in analyzer.df.columns:
                fig_ratio_taille = px.box(
                    analyzer.df,
                    x='taille_categorie',
                    y='ratio_horticole',
                    title='Ratio Horticole par Taille',
                    labels={'ratio_horticole': 'Ratio Horticole', 'taille_categorie': 'Taille'},
                    color='taille_categorie'
                )
                fig_ratio_taille.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_ratio_taille, use_container_width=True)
            else:
                st.info("⚠️ Données nécessaires non disponibles")
        
        with col3:
            # Évolution temporelle du ratio horticole
            if 'temporal' in horticultural_analysis:
                fig_ratio_time = px.line(
                    horticultural_analysis['temporal'],
                    x='annee_ouverture',
                    y='ratio_horticole',
                    title='Évolution du Ratio Horticole Moyen',
                    labels={'annee_ouverture': 'Année', 'ratio_horticole': 'Ratio Horticole (%)'},
                    markers=True
                )
                fig_ratio_time.update_traces(line_color='#228B22')
                fig_ratio_time.update_layout(height=350)
                st.plotly_chart(fig_ratio_time, use_container_width=True)
            else:
                st.info("⚠️ Évolution temporelle non disponible")
        
        # ================================================
        # CORRECTION: Section Meilleurs et Pires Ratios Horticoles
        # ================================================
        st.subheader("🏆 Meilleurs et Pires Ratios Horticoles par Arrondissement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Meilleurs Ratios (Global)")
            if 'top_ratios' in horticultural_analysis and horticultural_analysis['top_ratios'] is not None:
                if not horticultural_analysis['top_ratios'].empty:
                    display_df = horticultural_analysis['top_ratios'].copy()
                    if 'nom_ev' in display_df.columns:
                        display_df['nom_ev'] = display_df['nom_ev'].apply(
                            lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                        )
                    
                    st.dataframe(
                        display_df.style.format({
                            'ratio_horticole': '{:.1f}%',
                            'surface_totale_reelle': '{:,.0f}'
                        }),
                        hide_index=True,
                        height=200
                    )
                else:
                    st.write("Aucune donnée disponible")
            else:
                st.write("Aucune donnée disponible")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Meilleur par arrondissement
            if 'top_by_arrond' in horticultural_analysis and horticultural_analysis['top_by_arrond'] is not None:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🏅 Meilleur par Arrondissement")
                display_df = horticultural_analysis['top_by_arrond'].copy()
                if 'nom_ev' in display_df.columns:
                    display_df['nom_ev'] = display_df['nom_ev'].apply(
                        lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                    )
                
                st.dataframe(
                    display_df.style.format({
                        'ratio_horticole': '{:.1f}%',
                        'surface_totale_reelle': '{:,.0f}'
                    }),
                    hide_index=True,
                    height=150
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Ratios Faibles (Global)")
            if 'low_ratios' in horticultural_analysis and horticultural_analysis['low_ratios'] is not None:
                if not horticultural_analysis['low_ratios'].empty:
                    display_df = horticultural_analysis['low_ratios'].copy()
                    if 'nom_ev' in display_df.columns:
                        display_df['nom_ev'] = display_df['nom_ev'].apply(
                            lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                        )
                    
                    st.dataframe(
                        display_df.style.format({
                            'ratio_horticole': '{:.1f}%',
                            'surface_totale_reelle': '{:,.0f}'
                        }),
                        hide_index=True,
                        height=200
                    )
                else:
                    st.write("Aucune donnée disponible")
            else:
                st.write("Aucune donnée disponible")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Pire par arrondissement
            if 'low_by_arrond' in horticultural_analysis and horticultural_analysis['low_by_arrond'] is not None:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("### ⚠️ Plus Faible par Arrondissement")
                display_df = horticultural_analysis['low_by_arrond'].copy()
                if 'nom_ev' in display_df.columns:
                    display_df['nom_ev'] = display_df['nom_ev'].apply(
                        lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                    )
                
                st.dataframe(
                    display_df.style.format({
                        'ratio_horticole': '{:.1f}%',
                        'surface_totale_reelle': '{:,.0f}'
                    }),
                    hide_index=True,
                    height=150
                )
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== ANALYSE DÉTAILLÉE DE L'ACCESSIBILITÉ ====================
    st.markdown('<h2 class="section-header">🚪 Analyse Détailée de l\'Accessibilité</h2>', unsafe_allow_html=True)
    
    accessibility_data = analyzer.analyze_accessibility()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Surface Totale",
            f"{accessibility_data['total_surface']/10000:.1f} ha",
            help="Surface totale de tous les espaces"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Surface Accessible",
            f"{accessibility_data['open_surface']/10000:.1f} ha",
            help="Surface des espaces ouverts au public"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Taux d'Accessibilité",
            f"{accessibility_data['accessibility_rate']:.1f}%",
            help="% de surface réellement accessible"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualisation de l'accessibilité
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart: Espaces ouverts vs fermés
        if accessibility_data['open_spaces'] + accessibility_data['closed_spaces'] > 0:
            access_pie_data = pd.DataFrame({
                'Statut': ['Ouverts', 'Fermés', 'Statut Inconnu'],
                'Nombre': [
                    accessibility_data['open_spaces'],
                    accessibility_data['closed_spaces'],
                    accessibility_data['unknown_status']
                ]
            })
            
            fig_access_pie = px.pie(
                access_pie_data,
                values='Nombre',
                names='Statut',
                title='Répartition: Espaces Ouverts vs Fermés',
                color='Statut',
                color_discrete_map={
                    'Ouverts': 'green',
                    'Fermés': 'red',
                    'Statut Inconnu': 'gray'
                }
            )
            st.plotly_chart(fig_access_pie, use_container_width=True)
        else:
            st.info("⚠️ Données d'accessibilité non disponibles")
    
    with col2:
        # Surface accessible vs fermée
        if accessibility_data['total_surface'] > 0:
            surface_fermee = accessibility_data['closed_surface'] / 10000
            surface_totale = accessibility_data['total_surface'] / 10000
            
            fig_access_bar = go.Figure()
            fig_access_bar.add_trace(go.Bar(
                x=['Accessible', 'Fermée'],
                y=[accessibility_data['open_surface']/10000, surface_fermee],
                marker_color=['green', 'red'],
                text=[f"{accessibility_data['open_surface']/10000:.1f} ha", 
                      f"{surface_fermee:.1f} ha"],
                textposition='auto'
            ))
            
            fig_access_bar.update_layout(
                title='Surface Accessible vs Fermée',
                yaxis_title='Surface (hectares)',
                showlegend=False
            )
            st.plotly_chart(fig_access_bar, use_container_width=True)
        else:
            st.info("⚠️ Données de surface non disponibles")
    
    # Insight sur surface perdue
    if accessibility_data['closed_surface'] > 0:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Surface Verte \"Perdue\"")
        lost_percentage = (accessibility_data['closed_surface'] / accessibility_data['total_surface'] * 100) if accessibility_data['total_surface'] > 0 else 0
        st.write(f"**{surface_fermee:.1f} hectares** d'espaces verts sont actuellement FERMÉS "
                f"au public, soit **{lost_percentage:.1f}%** de la surface totale !")
        st.write("\n**💡 Recommandation:** Audit des espaces fermés pour identifier ceux qui "
                "pourraient être réhabilités et rouverts au public.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== CLUSTERING ====================
    st.markdown('<h2 class="section-header">🎯 Segmentation des Espaces Verts</h2>', unsafe_allow_html=True)
    
    cluster_profiles = analyzer.clustering_analysis(n_clusters=4)
    
    if not cluster_profiles.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot 3D si assez de dimensions
            required_cols = ['surface_totale_reelle', 'densite_espaces', 'ratio_horticole', 'cluster']
            if all(col in analyzer.df.columns for col in required_cols):
                fig6 = px.scatter_3d(
                    analyzer.df,
                    x='surface_totale_reelle',
                    y='densite_espaces',
                    z='ratio_horticole',
                    color='cluster',
                    size='score_qualite' if 'score_qualite' in analyzer.df.columns else None,
                    hover_name='nom_ev' if 'nom_ev' in analyzer.df.columns else None,
                    title='Clustering 3D des Espaces Verts',
                    labels={
                        'surface_totale_reelle': 'Surface (m²)',
                        'densite_espaces': 'Densité Équipements',
                        'ratio_horticole': 'Ratio Horticole',
                        'cluster': 'Cluster'
                    },
                    color_continuous_scale='Viridis'
                )
                fig6.update_layout(height=500)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("⚠️ Dimensions insuffisantes pour le clustering 3D")
        
        with col2:
            st.subheader("Profils des Clusters")
            st.dataframe(
                cluster_profiles.style.background_gradient(cmap='YlGn').format({
                    'surface_totale_reelle': '{:,.0f}',
                    'densite_espaces': '{:.4f}',
                    'ratio_horticole': '{:.2%}',
                    'age_espace': '{:.0f}',
                    'count': '{:.0f}'
                }),
                height=300
            )
    else:
        st.warning("⚠️ Clustering non disponible (données insuffisantes)")
    
    # ==================== CORRÉLATIONS ====================
    st.markdown('<h2 class="section-header">🔗 Analyse des Corrélations</h2>', unsafe_allow_html=True)
    
    counter_insights = analyzer.counter_intuitive_insights()
    
    if len(counter_insights) > 0:
        for i, insight in enumerate(counter_insights, 1):
            st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
            st.markdown(f"### 🔍 Insight #{i}: {insight['type']}")
            st.write(f"**📊 Constat:** {insight['finding']}")
            st.write(f"**💡 Interprétation:** {insight['interpretation']}")
            st.write(f"**⚡ Implication:** {insight['implication']}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Aucune corrélation contre-intuitive détectée")
    
    # Matrice de corrélation
    corr_matrix = analyzer.correlation_analysis()
    
    if corr_matrix is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig7 = px.imshow(
                corr_matrix,
                labels=dict(color="Corrélation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Matrice de Corrélation",
                zmin=-1, zmax=1
            )
            fig7.update_layout(height=500)
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("⚠️ Matrice de corrélation non disponible")
    
    # ==================== ESPACES REMARQUABLES ====================
    st.markdown('<h2 class="section-header">⭐ Espaces Remarquables par Arrondissement</h2>', unsafe_allow_html=True)
    
    outliers = analyzer.identify_outliers()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.subheader("🏆 Top 5 - Meilleure Qualité")
        if outliers['top_qualite'] is not None and not outliers['top_qualite'].empty:
            display_df = outliers['top_qualite'].copy()
            if 'nom_ev' in display_df.columns:
                display_df['nom_ev'] = display_df['nom_ev'].apply(
                    lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                )
            
            st.dataframe(
                display_df.style.format({
                    'score_qualite': '{:.1f}',
                    'surface_totale_reelle': '{:,.0f}'
                }),
                hide_index=True,
                height=200
            )
        else:
            st.write("Aucune donnée disponible")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.subheader("🌳 Top 5 - Plus Grandes Surfaces")
        if outliers['top_surface'] is not None and not outliers['top_surface'].empty:
            display_df = outliers['top_surface'].copy()
            if 'nom_ev' in display_df.columns:
                display_df['nom_ev'] = display_df['nom_ev'].apply(
                    lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                )
            
            st.dataframe(
                display_df.style.format({
                    'score_qualite': '{:.1f}',
                    'surface_totale_reelle': '{:,.0f}'
                }),
                hide_index=True,
                height=200
            )
        else:
            st.write("Aucune donnée disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.subheader("⚠️ Bottom 5 - Qualité à Améliorer")
        if outliers['low_qualite'] is not None and not outliers['low_qualite'].empty:
            display_df = outliers['low_qualite'].copy()
            if 'nom_ev' in display_df.columns:
                display_df['nom_ev'] = display_df['nom_ev'].apply(
                    lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                )
            
            st.dataframe(
                display_df.style.format({
                    'score_qualite': '{:.1f}',
                    'age_espace': '{:.0f}'
                }),
                hide_index=True,
                height=200
            )
        else:
            st.write("Aucune donnée disponible")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🎯 Top 5 - Haute Densité d'Équipements")
        if outliers['top_densite'] is not None and not outliers['top_densite'].empty:
            display_df = outliers['top_densite'].copy()
            if 'nom_ev' in display_df.columns:
                display_df['nom_ev'] = display_df['nom_ev'].apply(
                    lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else str(x)
                )
            
            st.dataframe(
                display_df.style.format({
                    'score_qualite': '{:.1f}',
                    'densite_espaces': '{:.4f}'
                }),
                hide_index=True,
                height=200
            )
        else:
            st.write("Aucune donnée disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== PRÉDICTIONS AVANCÉES AVEC LIGHTGBM ====================
    st.markdown('<h2 class="section-header">🔮 Prédictions Avancées avec LightGBM</h2>', unsafe_allow_html=True)
    
    if run_predictions:
        with st.spinner("Entraînement des modèles LightGBM en cours..."):
            # Train models
            models, category_info = analyzer.train_taille_models()
            
            if models:
                # Generate forecasts
                forecasts, forecast_only = analyzer.forecast_future_by_taille(
                    models, category_info, forecast_years
                )
                
                if forecasts is not None and forecast_only is not None:
                    # Display model performance
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("📈 Performance des Modèles LightGBM")
                    
                    model_summary = []
                    for taille, model_info in models.items():
                        model_summary.append({
                            'Catégorie': taille,
                            'R² (Test)': f"{model_info['metrics']['test_r2']:.3f}",
                            'MAE (Test)': f"{model_info['metrics']['test_mae']:,.0f} m²",
                            'RMSE (Test)': f"{model_info['metrics']['test_rmse']:,.0f} m²",
                            'Échantillons': f"{model_info['X_train_shape'][0]}+{model_info['X_test_shape'][0]}"
                        })
                    
                    summary_df = pd.DataFrame(model_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display forecast summary
                    st.markdown('<h3 class="section-header">📊 Résumé des Prévisions</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        total_new = len(forecast_only)
                        st.metric(
                            "Nouveaux Espaces",
                            f"{total_new}",
                            help=f"Nouveaux espaces sur {forecast_years} ans"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        total_surface = forecast_only['predicted_surface'].sum()
                        st.metric(
                            "Surface Nouvelle",
                            f"{total_surface/10000:.1f} ha",
                            help="Surface totale des nouveaux espaces"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        avg_surface = forecast_only['predicted_surface'].mean()
                        st.metric(
                            "Surface Moyenne",
                            f"{avg_surface:,.0f} m²",
                            help="Surface moyenne des nouveaux espaces"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        existing_surface = analyzer.df['surface_totale_reelle'].sum()
                        growth_pct = (total_surface / existing_surface * 100) if existing_surface > 0 else 0
                        st.metric(
                            "Croissance",
                            f"{growth_pct:.1f}%",
                            help="Croissance de la surface verte"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    if not forecast_only.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Total surface by year
                            yearly_totals = forecast_only.groupby('annee_creation')['predicted_surface'].sum().reset_index()
                            
                            fig_pred1 = px.bar(
                                yearly_totals,
                                x='annee_creation',
                                y='predicted_surface',
                                title='Surface Totale des Nouveaux Espaces par Année',
                                labels={'annee_creation': 'Année', 'predicted_surface': 'Surface (m²)'},
                                color='predicted_surface',
                                color_continuous_scale='Greens'
                            )
                            fig_pred1.update_layout(showlegend=False)
                            st.plotly_chart(fig_pred1, use_container_width=True)
                        
                        with col2:
                            # Number of spaces by category
                            space_counts = forecast_only.groupby('taille_categorie').size().reset_index()
                            space_counts.columns = ['taille_categorie', 'nombre']
                            
                            fig_pred2 = px.pie(
                                space_counts,
                                values='nombre',
                                names='taille_categorie',
                                title='Répartition des Nouveaux Espaces par Catégorie',
                                color='taille_categorie',
                                color_discrete_map={'Petit': '#66c2a5', 'Moyen': '#fc8d62', 'Grand': '#8da0cb'}
                            )
                            st.plotly_chart(fig_pred2, use_container_width=True)
                        
                        # Detailed forecasts table
                        st.subheader("📋 Détail des Prévisions")
                        
                        # Group by year and category for display
                        forecast_display = forecast_only.groupby(['annee_creation', 'taille_categorie']).agg({
                            'predicted_surface': ['count', 'sum', 'mean']
                        }).round(0)
                        
                        forecast_display.columns = ['Nombre', 'Surface Totale', 'Surface Moyenne']
                        forecast_display = forecast_display.reset_index()
                        
                        st.dataframe(
                            forecast_display.style.format({
                                'Nombre': '{:.0f}',
                                'Surface Totale': '{:,.0f}',
                                'Surface Moyenne': '{:,.0f}'
                            }).background_gradient(subset=['Surface Totale'], cmap='Greens'),
                            use_container_width=True
                        )
                    
                    # Store forecasts in session state
                    st.session_state['forecasts'] = forecasts
                    st.session_state['forecast_only'] = forecast_only
                    st.session_state['models'] = models
                    st.session_state['category_info'] = category_info
                else:
                    st.error("❌ Échec de la génération des prévisions")
            else:
                st.error("❌ Échec de l'entraînement des modèles")
    
    # If forecasts already exist in session state, show them
    elif 'forecasts' in st.session_state:
        st.info("📊 Prévisions déjà disponibles. Cliquez sur 'Lancer les Prédictions LightGBM' pour recalculer.")
        
        # Display existing forecasts
        forecast_only = st.session_state.get('forecast_only')
        if forecast_only is not None and not forecast_only.empty:
            st.subheader("📊 Prévisions Existantes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Summary metrics
                total_new = len(forecast_only)
                total_surface = forecast_only['predicted_surface'].sum()
                avg_surface = forecast_only['predicted_surface'].mean()
                
                st.metric("Nouveaux Espaces", f"{total_new}")
                st.metric("Surface Nouvelle", f"{total_surface/10000:.1f} ha")
                st.metric("Surface Moyenne", f"{avg_surface:,.0f} m²")
            
            with col2:
                st.dataframe(
                    forecast_only[['annee_creation', 'taille_categorie', 'arrondissement', 'predicted_surface']]
                    .sort_values('annee_creation')
                    .head(10),
                    use_container_width=True
                )
    
    # ==================== RECOMMANDATIONS ====================
    st.markdown('<h2 class="section-header">🚀 Recommandations Stratégiques</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🎯 Justice Territoriale")
        arrond_stats = analyzer.analyze_by_arrondissement()
        if arrond_stats is not None and 'surface_totale_m2' in arrond_stats.columns:
            worst_arrond = arrond_stats.nsmallest(3, 'surface_totale_m2')
            st.write("**Arrondissements sous-dotés:**")
            for idx, row in worst_arrond.iterrows():
                st.write(f"• {int(idx)}e: {row['surface_totale_m2']:,.0f} m²")
            st.write("\n**💡 Recommandation:** Prioriser la création d'espaces verts dans ces zones")
        else:
            st.write("Analyse par arrondissement non disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.subheader("📈 Tendances Positives")
        horticultural_analysis = analyzer.analyze_horticultural_ratio()
        if 'temporal' in horticultural_analysis:
            temporal_data_ratio = horticultural_analysis['temporal']
            if len(temporal_data_ratio) > 5:
                recent_avg = temporal_data_ratio.tail(5)['ratio_horticole'].mean()
                old_avg = temporal_data_ratio.head(5)['ratio_horticole'].mean()
                
                if old_avg > 0:
                    evolution = ((recent_avg - old_avg) / old_avg) * 100
                    st.write(f"**Évolution ratio horticole:**")
                    st.write(f"• Récents: {recent_avg:.1f}%")
                    st.write(f"• Anciens: {old_avg:.1f}%")
                    st.write(f"• **Progression: {evolution:+.1f}%**")
        st.write("\n✅ Les espaces récents sont plus végétalisés")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.subheader("⚠️ Points d'Attention")
        
        # Espaces anciens à rénover
        if 'age_espace' in analyzer.df.columns and 'score_qualite' in analyzer.df.columns:
            old_spaces = analyzer.df[analyzer.df['age_espace'] > 50]
            low_quality_old = old_spaces[old_spaces['score_qualite'] < 40]
            
            st.write(f"**Espaces >50 ans:** {len(old_spaces)}")
            st.write(f"**Dont qualité faible:** {len(low_quality_old)}")
            st.write("\n**💡 Action:** Programme de rénovation ciblé")
        else:
            st.write("Données d'âge ou qualité non disponibles")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== EXPORT DES DONNÉES ====================
    st.markdown('<h2 class="section-header">💾 Export des Données</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = analyzer.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger Dataset Complet",
            data=csv_data,
            file_name="espaces_verts_analyse_complete.csv",
            mime="text/csv"
        )
    
    with col2:
        arrond_stats = analyzer.analyze_by_arrondissement()
        if arrond_stats is not None:
            csv_arrond = arrond_stats.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Statistiques par Arrondissement",
                data=csv_arrond,
                file_name="stats_arrondissements.csv",
                mime="text/csv"
            )
        else:
            st.info("Stats arrondissements non dispo")
    
    with col3:
        if 'forecast_only' in st.session_state:
            csv_forecast = st.session_state['forecast_only'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Prévisions LightGBM",
                data=csv_forecast,
                file_name="previsions_lightgbm.csv",
                mime="text/csv"
            )
        else:
            st.info("Exécutez d'abord les prédictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Dashboard d'Analyse des Espaces Verts Parisiens avec LightGBM</strong></p>
        <p>Source de données: Open Data Paris | Analyse réalisée avec Python, Streamlit, Plotly, LightGBM</p>
        <p>🌳 Pour un Paris plus vert et équitable</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()