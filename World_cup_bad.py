import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
from scipy.stats import poisson
from numba import njit, prange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - PATHS
# =============================================================================

PATH_RESULTS = Path(r"C:\Users\othma\.cache\kagglehub\datasets\martj42\international-football-results-from-1872-to-2017\versions\115\results.csv")
PATH_ELO = Path(r"C:\Users\othma\.cache\kagglehub\datasets\saifalnimri\international-football-elo-ratings\versions\1\eloratings.csv")
PATH_ELO_WC2026 = Path(r"C:\Users\othma\.cache\kagglehub\datasets\afonsofernandescruz\2026-fifa-world-cup-historical-elo-ratings\versions\1\elo_ratings_wc2026.csv")

TODAY = pd.to_datetime('2026-06-07')
DECAY_RATE = 0.003
DIXON_COLES_RHO = -0.075

# =============================================================================
# PARAMETRES DE PERFORMANCE
# =============================================================================

N_WORKERS = mp.cpu_count() - 1
BATCH_SIZE = 1000
USE_NUMBA = True
USE_PARALLEL = True

# =============================================================================
# GROUPES COUPE DU MONDE 2026 (48 equipes, 12 groupes)
# Source: Tirage au sort officiel decembre 2025
# =============================================================================

GROUPES_DEF = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

HOTES = ['United States', 'Mexico', 'Canada']

# =============================================================================
# EXPERIENCE COUPE DU MONDE
# =============================================================================

EXPERIENCE_CDMS = {
    'Brazil': 22, 'Germany': 20, 'Italy': 18, 'Argentina': 18,
    'France': 16, 'England': 16, 'Spain': 16, 'Uruguay': 14,
    'Mexico': 17, 'Netherlands': 11, 'Belgium': 14, 'Sweden': 12,
    'Switzerland': 12, 'Portugal': 8, 'Croatia': 6, 'Denmark': 6,
    'Poland': 9, 'South Korea': 11, 'Japan': 8, 'Australia': 6,
    'United States': 11, 'Iran': 6, 'Saudi Arabia': 6, 'Ecuador': 4,
    'Senegal': 3, 'Morocco': 6, 'Tunisia': 6, 'Cameroon': 8,
    'Ghana': 4, 'Nigeria': 6, 'Algeria': 4, 'Ivory Coast': 3,
    'South Africa': 3, 'Egypt': 3, 'Costa Rica': 6, 'Panama': 1,
    'Canada': 2, 'Qatar': 1, 'Wales': 2, 'Scotland': 8,
    'Turkey': 2, 'Czechia': 1, 'Austria': 7, 'Hungary': 9,
    'Norway': 3, 'Ireland': 3, 'Northern Ireland': 3, 'Greece': 3,
    'Bosnia and Herzegovina': 1, 'Serbia': 13, 'Slovenia': 2,
    'Slovakia': 1, 'North Macedonia': 1, 'Iceland': 1, 'Finland': 1,
    'New Zealand': 2, 'China': 1, 'Oman': 0, 'Iraq': 1,
    'Jordan': 1, 'Uzbekistan': 0, 'Cape Verde': 0, 'Haiti': 1,
    'Jamaica': 1, 'Trinidad and Tobago': 1, 'Honduras': 3,
    'Guatemala': 0, 'El Salvador': 2, 'Paraguay': 8, 'Chile': 9,
    'Colombia': 6, 'Peru': 5, 'Bolivia': 3, 'Venezuela': 0,
    'Democratic Republic of Congo': 1, 'Mali': 0, 'Burkina Faso': 0,
    'Guinea': 0, 'Zambia': 0, 'Zimbabwe': 0, 'Angola': 1,
    'Togo': 1, 'Gabon': 0, 'Equatorial Guinea': 0, 'Benin': 0,
    'Uganda': 0, 'Madagascar': 0, 'Mauritania': 0, 'Libya': 0,
    'Sudan': 1, 'Ethiopia': 0, 'Kenya': 0, 'Tanzania': 0,
    'Congo': 1, 'Central African Republic': 0, 'Chad': 0,
    'Niger': 0, 'Botswana': 0, 'Namibia': 0, 'Mozambique': 0,
    'Malawi': 0, 'Lesotho': 0, 'Eswatini': 0, 'Comoros': 0,
    'Seychelles': 0, 'Mauritius': 0, 'Rwanda': 0, 'Burundi': 0,
    'Djibouti': 0, 'Somalia': 0, 'Eritrea': 0, 'South Sudan': 0,
    'Thailand': 2, 'Indonesia': 1, 'Malaysia': 0, 'Singapore': 0,
    'Vietnam': 0, 'Philippines': 0, 'Myanmar': 0, 'Cambodia': 0,
    'Laos': 0, 'Brunei': 0, 'Timor-Leste': 0, 'India': 0,
    'Pakistan': 0, 'Bangladesh': 0, 'Sri Lanka': 0, 'Nepal': 0,
    'Bhutan': 0, 'Maldives': 0, 'Afghanistan': 0, 'Curacao': 0,
    'DR Congo': 1
}

# =============================================================================
# FONCTIONS AUXILIAIRES POUR LES NOMS D'EQUIPES
# =============================================================================

def get_nom_dataset(nom_fr):
    """Convertit le nom francais vers le nom du dataset si different."""
    mapping = {
        'USA': 'United States',
        'Korea Republic': 'South Korea',
        'Korea DPR': 'North Korea',
        'Ivory Coast': "Cote d'Ivoire",
        'Czech Republic': 'Czechia',
        'DR Congo': 'Democratic Republic of Congo',
        'Czechia': 'Czech Republic',
    }
    # Essayer dans les deux sens
    for k, v in mapping.items():
        if nom_fr == k:
            return v
        if nom_fr == v:
            return k
    return nom_fr

def get_elo(dernier_elo, nom_ds):
    """Recupere l'Elo d'une equipe."""
    # Essayer plusieurs variantes de nom
    for name in [nom_ds, get_nom_dataset(nom_ds)]:
        if name in dernier_elo:
            return float(dernier_elo[name])
    return 1500.0

def get_forme(dernier_formes, nom_ds, type_forme):
    """Recupere la forme d'une equipe."""
    for name in [nom_ds, get_nom_dataset(nom_ds)]:
        if name in dernier_formes:
            return float(dernier_formes[name].get(type_forme, 1.3))
    return 1.3

def get_progression(progression_elo, nom_ds):
    """Recupere la progression Elo d'une equipe."""
    for name in [nom_ds, get_nom_dataset(nom_ds)]:
        if name in progression_elo:
            return float(progression_elo[name])
    return 0.0

def get_experience(nom_ds):
    """Recupere l'experience Coupe du Monde d'une equipe."""
    for name in [nom_ds, get_nom_dataset(nom_ds)]:
        if name in EXPERIENCE_CDMS:
            return float(EXPERIENCE_CDMS[name])
    return 0.0

# =============================================================================
# CLASSE EQUIPE
# =============================================================================

@dataclass
class Equipe:
    nom_fr: str
    nom_ds: str
    groupe: str
    elo: float
    forme_att_init: float
    forme_def_init: float
    progression: float
    experience: float
    est_hote: bool
    forme_att: float = field(init=False)
    forme_def: float = field(init=False)
    victoires_consecutives: int = 0

    def __post_init__(self):
        self.forme_att = self.forme_att_init
        self.forme_def = self.forme_def_init
        self.victoires_consecutives = 0

    def maj_forme(self, buts_marques, buts_encaisses):
        """Met a jour la forme apres un match."""
        self.forme_att = 0.7 * self.forme_att + 0.3 * buts_marques
        self.forme_def = 0.7 * self.forme_def + 0.3 * buts_encaisses
        if buts_marques > buts_encaisses:
            self.victoires_consecutives += 1
        else:
            self.victoires_consecutives = 0

    def reset_forme(self):
        """Reinitialise la forme pour une nouvelle simulation."""
        self.forme_att = self.forme_att_init
        self.forme_def = self.forme_def_init
        self.victoires_consecutives = 0

# =============================================================================
# 1-10. FONCTIONS DE PREPARATION
# =============================================================================

def charger_donnees():
    data = pd.read_csv(PATH_RESULTS)
    elo_historique = pd.read_csv(PATH_ELO)
    elo_wc2026 = pd.read_csv(PATH_ELO_WC2026)
    elo_historique['team'] = elo_historique['team'].str.replace('\xa0', ' ', regex=False)
    data = data[data["date"] <= '2025-12-13'].copy()
    data = data[data["date"] >= '2014-01-01'].copy()
    data["date"] = pd.to_datetime(data["date"])
    elo_historique["date"] = pd.to_datetime(elo_historique["date"], format='mixed')
    elo_latest = elo_wc2026.loc[elo_wc2026.groupby('country')['snapshot_date'].idxmax()]
    return data, elo_historique, elo_latest, elo_wc2026

def calculer_progression_elo(elo_wc2026, annee_debut=2018, annee_fin=2022):
    progression = {}
    for pays in elo_wc2026['country'].unique():
        df_pays = elo_wc2026[elo_wc2026['country'] == pays].copy()
        df_pays['year'] = pd.to_datetime(df_pays['snapshot_date']).dt.year
        rating_debut = df_pays[df_pays['year'] == annee_debut]['rating']
        rating_fin = df_pays[df_pays['year'] == annee_fin]['rating']
        if len(rating_debut) > 0 and len(rating_fin) > 0:
            progression[pays] = float(rating_fin.iloc[0]) - float(rating_debut.iloc[0])
        else:
            progression[pays] = 0
    return progression

def ajouter_elo(data, elo_historique, progression_elo):
    data = data.sort_values('date').copy()
    elo_historique = elo_historique.sort_values('date').copy()
    data = pd.merge_asof(data, elo_historique, left_on='date', right_on='date',
                         left_by='home_team', right_by='team', direction='backward'
                         ).rename(columns={'rating': 'elo_home'}).drop('team', axis=1)
    data = pd.merge_asof(data, elo_historique, left_on='date', right_on='date',
                         left_by='away_team', right_by='team', direction='backward'
                         ).rename(columns={'rating': 'elo_away'}).drop('team', axis=1)
    data['elo_home'] = data['elo_home'].fillna(1500)
    data['elo_away'] = data['elo_away'].fillna(1500)
    data["diff_elo"] = data["elo_home"] - data["elo_away"]

    mapping_progression = {
        'USA': 'United States', 'Korea Republic': 'South Korea',
        'Korea DPR': 'North Korea', 'Ivory Coast': "Cote d'Ivoire",
        'Czech Republic': 'Czechia', 'DR Congo': 'Democratic Republic of Congo',
    }
    def get_progression_team(team_name):
        return progression_elo.get(team_name, progression_elo.get(mapping_progression.get(team_name, team_name), 0))

    data['progression_home'] = data['home_team'].apply(get_progression_team)
    data['progression_away'] = data['away_team'].apply(get_progression_team)
    data['diff_progression'] = data['progression_home'] - data['progression_away']
    data['exp_home'] = data['home_team'].map(EXPERIENCE_CDMS).fillna(0)
    data['exp_away'] = data['away_team'].map(EXPERIENCE_CDMS).fillna(0)
    data['diff_exp'] = data['exp_home'] - data['exp_away']
    return data

def ajouter_formes(data):
    data["forme_att_home"] = data.groupby("home_team")["home_score"].transform(
        lambda x: x.ewm(span=5).mean()).shift(1)
    data["forme_att_away"] = data.groupby("away_team")["away_score"].transform(
        lambda x: x.ewm(span=5).mean()).shift(1)
    data["forme_def_home"] = data.groupby("home_team")["away_score"].transform(
        lambda x: x.ewm(span=5).mean()).shift(1)
    data["forme_def_away"] = data.groupby("away_team")["home_score"].transform(
        lambda x: x.ewm(span=5).mean()).shift(1)
    return data

def ajouter_variables_contextuelles(data):
    def get_tournament_weight(tournament):
        if 'FIFA World Cup' in tournament and 'qualification' not in tournament:
            return 6.0
        elif any(x in tournament for x in ['UEFA Euro', 'Copa America', 'AFC Asian Cup', 'African Cup of Nations', 'Gold Cup']):
            return 5.0
        elif 'qualification' in tournament:
            return 3.0
        elif 'Nations League' in tournament:
            return 4.0
        elif 'Friendly' in tournament:
            return 1.0
        else:
            return 1.5

    data['tournament_weight'] = data['tournament'].apply(get_tournament_weight)
    data['is_major'] = (data['tournament_weight'] >= 4.0).astype(int)

    dictionnaire_hosts = {2014: ['Brazil'], 2018: ['Russia'], 2022: ['Qatar'], 2026: ['USA', 'Mexico', 'Canada']}
    def designer_hote(row):
        annee = row['date'].year
        if row['tournament'] == 'FIFA World Cup':
            return 1 if annee in dictionnaire_hosts and row['home_team'] in dictionnaire_hosts[annee] else 0
        return 1 if row['neutral'] == False else 0

    data['is_host_home'] = data.apply(designer_hote, axis=1)

    def calculer_fatigue(elo):
        return 0.50 if pd.isna(elo) or elo == 0 else 0.30 + 0.65 / (1 + np.exp(-0.005 * (elo - 1750)))

    data['fatigue_home'] = data['elo_home'].apply(calculer_fatigue)
    data['fatigue_away'] = data['elo_away'].apply(calculer_fatigue)
    return data

def construire_dataset_poisson(data):
    data['diff_forme_raw'] = data['forme_att_home'] - data['forme_def_away']

    df_home = pd.DataFrame({
        'date': data['date'], 'buts_marques': data['home_score'],
        'diff_elo': data['diff_elo'], 'diff_forme_raw': data['diff_forme_raw'],
        'diff_progression': data['diff_progression'], 'diff_exp': data['diff_exp'],
        'is_host': data['is_host_home'], 'is_major': data['is_major'],
        'fatigue': data['fatigue_home']
    })
    df_away = pd.DataFrame({
        'date': data['date'], 'buts_marques': data['away_score'],
        'diff_elo': -data['diff_elo'],
        'diff_forme_raw': -(data['forme_att_away'] - data['forme_def_home']),
        'diff_progression': -data['diff_progression'], 'diff_exp': -data['diff_exp'],
        'is_host': 0, 'is_major': data['is_major'], 'fatigue': data['fatigue_away']
    })

    dataset = pd.concat([df_home, df_away], ignore_index=True).dropna()

    moyenne_forme = dataset['diff_forme_raw'].mean()
    std_forme = dataset['diff_forme_raw'].std()
    moyenne_prog = dataset['diff_progression'].mean()
    std_prog = dataset['diff_progression'].std()
    moyenne_exp = dataset['diff_exp'].mean()
    std_exp = dataset['diff_exp'].std()

    dataset['diff_forme'] = (dataset['diff_forme_raw'] - moyenne_forme) / std_forme
    dataset['diff_progression_norm'] = (dataset['diff_progression'] - moyenne_prog) / std_prog
    dataset['diff_exp_norm'] = (dataset['diff_exp'] - moyenne_exp) / std_exp

    dataset['days_diff'] = (TODAY - dataset['date']).dt.days
    dataset['poids'] = np.exp(-DECAY_RATE * dataset['days_diff'])

    cols_num = ['buts_marques', 'diff_elo', 'diff_forme', 'diff_progression_norm',
                'diff_exp_norm', 'is_host', 'is_major', 'fatigue', 'poids']
    dataset[cols_num] = dataset[cols_num].astype(float)

    return dataset, {
        'moyenne_forme': moyenne_forme, 'std_forme': std_forme,
        'moyenne_prog': moyenne_prog, 'std_prog': std_prog,
        'moyenne_exp': moyenne_exp, 'std_exp': std_exp
    }

def entrainer_modele(dataset):
    formule = "buts_marques ~ diff_elo + diff_forme + diff_progression_norm + diff_exp_norm + is_host + is_major + fatigue"
    modele = smf.glm(formula=formule, data=dataset, family=sm.families.Poisson(),
                     var_weights=dataset['poids'].values).fit()
    return modele

# =============================================================================
# FONCTION MANQUANTE: preparer_donnees_historiques
# =============================================================================

def preparer_donnees_historiques(data, elo_historique, elo_wc2026):
    """
    Prepare les dernieres donnees Elo et formes pour le simulateur du tournoi.
    """
    # Dernier Elo connu pour chaque equipe dans l'historique
    dernier_elo = elo_historique.sort_values('date').groupby('team').last()['rating'].to_dict()

    # Ajouter les Elo du dataset WC2026 s'ils sont plus recents
    elo_wc2026['snapshot_date'] = pd.to_datetime(elo_wc2026['snapshot_date'])
    elo_latest_wc = elo_wc2026.loc[elo_wc2026.groupby('country')['snapshot_date'].idxmax()]

    for _, row in elo_latest_wc.iterrows():
        team_name = str(row['country'])
        rating = float(row['rating'])
        dernier_elo[team_name] = rating
        # Ajouter aussi les variantes de noms
        alt_name = get_nom_dataset(team_name)
        if alt_name != team_name:
            dernier_elo[alt_name] = rating

    # Calculer les formes recentes (moyenne des 5 derniers matchs)
    dernier_formes = {}
    all_teams = set(data['home_team'].unique()) | set(data['away_team'].unique())

    for team in all_teams:
        team_home = data[data['home_team'] == team][['date', 'home_score', 'away_score']].copy()
        team_home.columns = ['date', 'buts_marques', 'buts_encaisses']
        team_home['domicile'] = 1

        team_away = data[data['away_team'] == team][['date', 'away_score', 'home_score']].copy()
        team_away.columns = ['date', 'buts_marques', 'buts_encaisses']
        team_away['domicile'] = 0

        team_matches = pd.concat([team_home, team_away]).sort_values('date').tail(5)

        if len(team_matches) > 0:
            forme_att = float(team_matches['buts_marques'].mean())
            forme_def = float(team_matches['buts_encaisses'].mean())
        else:
            forme_att = 1.3
            forme_def = 1.3

        dernier_formes[team] = {
            'forme_att': forme_att,
            'forme_def': forme_def
        }

    # Recalculer la progression Elo
    progression_elo = calculer_progression_elo(elo_wc2026)

    return dernier_elo, dernier_formes, progression_elo

# =============================================================================
# CORRECTION DIXON-COLES
# =============================================================================

class DixonColesCorrection:
    def __init__(self, rho=DIXON_COLES_RHO):
        self.rho = rho
        self.tau_cache = {}

    def tau(self, x, y, lambda_x, lambda_y):
        key = (x, y)
        if key not in self.tau_cache:
            if x == 0 and y == 0:
                self.tau_cache[key] = 1 + self.rho * lambda_x * lambda_y
            elif x == 0 and y == 1:
                self.tau_cache[key] = 1 - self.rho * lambda_x
            elif x == 1 and y == 0:
                self.tau_cache[key] = 1 - self.rho * lambda_y
            elif x == 1 and y == 1:
                self.tau_cache[key] = 1 + self.rho
            else:
                self.tau_cache[key] = 1.0
        return self.tau_cache[key]

    def prob_jointe(self, lambda_home, lambda_away, max_buts=10):
        i = np.arange(max_buts + 1)
        j = np.arange(max_buts + 1)
        I, J = np.meshgrid(i, j, indexing='ij')

        pois_home = poisson.pmf(I, lambda_home)
        pois_away = poisson.pmf(J, lambda_away)

        tau = np.ones_like(I, dtype=float)
        tau[(I==0) & (J==0)] = 1 + self.rho * lambda_home * lambda_away
        tau[(I==0) & (J==1)] = 1 - self.rho * lambda_home
        tau[(I==1) & (J==0)] = 1 - self.rho * lambda_away
        tau[(I==1) & (J==1)] = 1 + self.rho

        probs = pois_home * pois_away * tau
        return probs / probs.sum()

# =============================================================================
# SIMULATION VECTORISEE AVEC NUMBA
# =============================================================================

@njit(cache=True)
def simuler_match_numba(elo_dom, elo_ext, forme_att_dom, forme_def_ext,
                        forme_att_ext, forme_def_dom, prog_dom, prog_ext,
                        is_host_dom, exp_dom, exp_ext,
                        beta_0, beta_elo, beta_forme, beta_progression,
                        beta_exp, beta_host, beta_major, beta_fatigue,
                        moyenne_forme, std_forme, std_prog, moyenne_exp, std_exp,
                        momentum_dom, momentum_ext, eliminatoire, rho):
    """Version numba du calcul de lambda et simulation."""

    diff_elo = elo_dom - elo_ext
    diff_forme_raw = forme_att_dom - forme_def_ext
    diff_forme = (diff_forme_raw - moyenne_forme) / std_forme

    prog_eff_dom = prog_dom * max(np.tanh((elo_dom - 1500) / 500), 0.3)
    prog_eff_ext = prog_ext * max(np.tanh((elo_ext - 1500) / 500), 0.3)
    diff_prog = (prog_eff_dom - prog_eff_ext) / std_prog

    elo_bonus = 0.15 * np.tanh((elo_dom - 1500) / 500 * 2)

    diff_exp = (exp_dom - exp_ext - moyenne_exp) / std_exp

    fatigue_dom = 0.30 + 0.65 / (1 + np.exp(-0.005 * (elo_dom - 1750)))

    log_l = (beta_0 + beta_elo * diff_elo + beta_forme * diff_forme * 1.5 +
             beta_progression * diff_prog + elo_bonus + beta_exp * diff_exp +
             beta_host * is_host_dom + beta_major * 1.0 + beta_fatigue * fatigue_dom)

    lam_dom = max(np.exp(log_l), 0.01)

    diff_elo_ext = elo_ext - elo_dom
    diff_forme_raw_ext = forme_att_ext - forme_def_dom
    diff_forme_ext = (diff_forme_raw_ext - moyenne_forme) / std_forme

    diff_prog_ext = (prog_eff_ext - prog_eff_dom) / std_prog

    elo_bonus_ext = 0.15 * np.tanh((elo_ext - 1500) / 500 * 2)

    diff_exp_ext = (exp_ext - exp_dom - moyenne_exp) / std_exp

    fatigue_ext = 0.30 + 0.65 / (1 + np.exp(-0.005 * (elo_ext - 1750)))

    log_l_ext = (beta_0 + beta_elo * diff_elo_ext + beta_forme * diff_forme_ext * 1.5 +
                 beta_progression * diff_prog_ext + elo_bonus_ext + beta_exp * diff_exp_ext +
                 beta_host * 0.0 + beta_major * 1.0 + beta_fatigue * fatigue_ext)

    lam_ext = max(np.exp(log_l_ext), 0.01)

    lam_dom *= (1.0 + momentum_dom)
    lam_ext *= (1.0 + momentum_ext)

    if eliminatoire:
        diff = abs(elo_dom - elo_ext)
        sigma = 0.20 if diff < 100 else (0.15 if diff < 300 else 0.08)
        lam_dom *= np.exp(np.random.normal(0.0, sigma))
        lam_ext *= np.exp(np.random.normal(0.0, sigma))
        lam_dom = max(lam_dom, 0.1)
        lam_ext = max(lam_ext, 0.1)

    buts_dom = np.random.poisson(lam_dom)
    buts_ext = np.random.poisson(lam_ext)

    if buts_dom <= 1 and buts_ext <= 1:
        if buts_dom == 0 and buts_ext == 0:
            p = 1 + rho * lam_dom * lam_ext
        elif buts_dom == 0 and buts_ext == 1:
            p = 1 - rho * lam_dom
        elif buts_dom == 1 and buts_ext == 0:
            p = 1 - rho * lam_ext
        elif buts_dom == 1 and buts_ext == 1:
            p = 1 + rho
        else:
            p = 1.0

        if np.random.random() < abs(p - 1) * 0.1:
            if buts_dom == 0 and buts_ext == 1:
                buts_ext = 0
            elif buts_dom == 1 and buts_ext == 0:
                buts_dom = 0

    return buts_dom, buts_ext

# =============================================================================
# SIMULATION DU TOURNOI
# =============================================================================

class SimulateurTournoi:
    def __init__(self, modele, params_norm, dernier_elo, dernier_formes, progression_elo, use_dixon_coles=True):
        self.modele = modele
        self.params_norm = params_norm
        self.dernier_elo = dernier_elo
        self.dernier_formes = dernier_formes
        self.progression_elo = progression_elo
        self.dixon_coles = DixonColesCorrection() if use_dixon_coles else None

        self.beta_0 = modele.params['Intercept']
        self.beta_elo = modele.params['diff_elo']
        self.beta_forme = 0.04
        self.beta_progression = 0.015
        self.beta_exp = 0.02
        self.beta_host = modele.params['is_host']
        self.beta_major = modele.params['is_major']
        self.beta_fatigue = modele.params['fatigue']

        self.moyenne_forme = params_norm['moyenne_forme']
        self.std_forme = params_norm['std_forme']
        self.std_prog = params_norm['std_prog']
        self.moyenne_exp = params_norm['moyenne_exp']
        self.std_exp = params_norm['std_exp']

        self.equipes_data = self._preparer_equipes_data()

    def _preparer_equipes_data(self):
        equipes_data = {}
        for lettre, noms_fr in GROUPES_DEF.items():
            for nom_fr in noms_fr:
                nom_ds = get_nom_dataset(nom_fr)
                elo = get_elo(self.dernier_elo, nom_ds)
                fa = get_forme(self.dernier_formes, nom_ds, 'forme_att')
                fd = get_forme(self.dernier_formes, nom_ds, 'forme_def')
                prog = get_progression(self.progression_elo, nom_ds)
                exp = get_experience(nom_ds)
                est_hote = nom_fr in HOTES or nom_ds in HOTES

                equipes_data[nom_fr] = {
                    'nom_ds': nom_ds, 'groupe': lettre, 'elo': elo,
                    'forme_att_init': fa, 'forme_def_init': fd,
                    'progression': prog, 'experience': exp, 'est_hote': est_hote
                }
        return equipes_data

    def creer_equipe(self, nom_fr):
        data = self.equipes_data[nom_fr]
        return Equipe(
            nom_fr=nom_fr, nom_ds=data['nom_ds'], groupe=data['groupe'],
            elo=data['elo'], forme_att_init=data['forme_att_init'],
            forme_def_init=data['forme_def_init'], progression=data['progression'],
            experience=data['experience'], est_hote=data['est_hote']
        )

    def construire_toutes_equipes(self):
        return [self.creer_equipe(nom_fr)
                for lettre, noms in GROUPES_DEF.items()
                for nom_fr in noms]

    def calculer_lambda(self, elo_eq, elo_adv, forme_att_eq, forme_def_adv,
                       progression_eq, progression_adv, is_host, fatigue_eq,
                       exp_eq=0, exp_adv=0):
        diff_elo = elo_eq - elo_adv
        diff_forme_raw = forme_att_eq - forme_def_adv
        diff_forme = (diff_forme_raw - self.moyenne_forme) / self.std_forme

        prog_eff_eq = progression_eq * max(np.tanh((elo_eq - 1500) / 500), 0.3)
        prog_eff_adv = progression_adv * max(np.tanh((elo_adv - 1500) / 500), 0.3)
        diff_prog = (prog_eff_eq - prog_eff_adv) / self.std_prog

        elo_bonus = 0.15 * np.tanh((elo_eq - 1500) / 500 * 2)

        diff_exp = (exp_eq - exp_adv - self.moyenne_exp) / self.std_exp

        log_l = (self.beta_0 + self.beta_elo * diff_elo +
                 self.beta_forme * diff_forme * 1.5 +
                 self.beta_progression * diff_prog + elo_bonus +
                 self.beta_exp * diff_exp +
                 self.beta_host * is_host + self.beta_major * 1 +
                 self.beta_fatigue * fatigue_eq)

        return max(np.exp(log_l), 0.01)

    def simuler_match(self, eq_dom, eq_ext, eliminatoire=False):
        lam_dom = self.calculer_lambda(
            eq_dom.elo, eq_ext.elo, eq_dom.forme_att, eq_ext.forme_def,
            eq_dom.progression, eq_ext.progression,
            1 if eq_dom.est_hote else 0,
            0.30 + 0.65 / (1 + np.exp(-0.005 * (eq_dom.elo - 1750))),
            exp_eq=eq_dom.experience, exp_adv=eq_ext.experience
        )
        lam_ext = self.calculer_lambda(
            eq_ext.elo, eq_dom.elo, eq_ext.forme_att, eq_dom.forme_def,
            eq_ext.progression, eq_dom.progression, 0,
            0.30 + 0.65 / (1 + np.exp(-0.005 * (eq_ext.elo - 1750))),
            exp_eq=eq_ext.experience, exp_adv=eq_dom.experience
        )

        momentum_dom = min(0.03 * eq_dom.victoires_consecutives, 0.12)
        momentum_ext = min(0.03 * eq_ext.victoires_consecutives, 0.12)
        lam_dom *= (1 + momentum_dom)
        lam_ext *= (1 + momentum_ext)

        if eliminatoire:
            diff = abs(eq_dom.elo - eq_ext.elo)
            sigma = 0.20 if diff < 100 else (0.15 if diff < 300 else 0.08)
            lam_dom *= np.exp(np.random.normal(0, sigma))
            lam_ext *= np.exp(np.random.normal(0, sigma))
            lam_dom = max(lam_dom, 0.1)
            lam_ext = max(lam_ext, 0.1)

        if self.dixon_coles is not None:
            probs = self.dixon_coles.prob_jointe(lam_dom, lam_ext)
            scores = np.random.choice(probs.size, p=probs.flatten())
            buts_dom = scores // probs.shape[1]
            buts_ext = scores % probs.shape[1]
        else:
            buts_dom = np.random.poisson(lam_dom)
            buts_ext = np.random.poisson(lam_ext)

        vainqueur = None
        if eliminatoire and buts_dom == buts_ext:
            buts_dom += np.random.poisson(lam_dom * 0.6)
            buts_ext += np.random.poisson(lam_ext * 0.6)
            if buts_dom == buts_ext:
                vainqueur = self._tirs_au_but(eq_dom, eq_ext)
            else:
                vainqueur = eq_dom.nom_fr if buts_dom > buts_ext else eq_ext.nom_fr
        else:
            if buts_dom > buts_ext:
                vainqueur = eq_dom.nom_fr
            elif buts_ext > buts_dom:
                vainqueur = eq_ext.nom_fr

        eq_dom.maj_forme(buts_dom, buts_ext)
        eq_ext.maj_forme(buts_ext, buts_dom)
        return buts_dom, buts_ext, vainqueur

    def _tirs_au_but(self, eq_dom, eq_ext):
        base_prob = 0.75
        elo_diff = eq_dom.elo - eq_ext.elo
        prob_dom = base_prob + 0.05 * np.tanh(elo_diff / 300)
        prob_ext = base_prob - 0.05 * np.tanh(elo_diff / 300)
        buts_dom_tab = np.random.binomial(5, prob_dom)
        buts_ext_tab = np.random.binomial(5, prob_ext)
        while buts_dom_tab == buts_ext_tab:
            buts_dom_tab += np.random.binomial(1, prob_dom)
            buts_ext_tab += np.random.binomial(1, prob_ext)
        return eq_dom.nom_fr if buts_dom_tab > buts_ext_tab else eq_ext.nom_fr

# =============================================================================
# PHASE DE GROUPES ET ELIMINATOIRE
# =============================================================================

def simuler_groupe(simulateur, equipes, nom_groupe):
    matchs = [(0,1), (2,3), (0,2), (1,3), (0,3), (1,2)]
    stats = defaultdict(lambda: {'pts': 0, 'bp': 0, 'bc': 0, 'diff': 0, 'joues': 0})

    for idx_dom, idx_ext in matchs:
        bd, be, v = simulateur.simuler_match(equipes[idx_dom], equipes[idx_ext], False)
        d, e = equipes[idx_dom].nom_fr, equipes[idx_ext].nom_fr
        stats[d]['bp'] += bd; stats[d]['bc'] += be
        stats[e]['bp'] += be; stats[e]['bc'] += bd
        stats[d]['joues'] += 1; stats[e]['joues'] += 1
        if bd > be: stats[d]['pts'] += 3
        elif be > bd: stats[e]['pts'] += 3
        else: stats[d]['pts'] += 1; stats[e]['pts'] += 1

    for n in stats: stats[n]['diff'] = stats[n]['bp'] - stats[n]['bc']
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df.sort_values(['pts', 'diff', 'bp'], ascending=[False, False, False])
    df['rang'] = range(1, len(df) + 1)
    return df

def simuler_tous_groupes(simulateur, equipes_par_groupe):
    classements = {}; premiers, deuxiemes, troisiemes_data = [], [], []
    for grp, eqs in equipes_par_groupe.items():
        cl = simuler_groupe(simulateur, eqs, grp)
        classements[grp] = cl
        noms = cl.index.tolist()
        premiers.append(noms[0]); deuxiemes.append(noms[1])
        r3 = cl.iloc[2]
        troisiemes_data.append({'nom': noms[2], 'pts': r3['pts'],
                                'diff': r3['diff'], 'bp': r3['bp'], 'groupe': grp})
    df3 = pd.DataFrame(troisiemes_data).sort_values(
        ['pts', 'diff', 'bp'], ascending=[False, False, False])
    return classements, premiers, deuxiemes, df3.head(8)['nom'].tolist()

def simuler_tour_eliminatoire(simulateur, paires, equipes_dict):
    return [simulateur.simuler_match(equipes_dict[n1], equipes_dict[n2], True)[2]
            for n1, n2 in paires]

def generer_paires_32eme(premiers, deuxiemes, troisiemes):
    np.random.shuffle(deuxiemes); np.random.shuffle(troisiemes)
    non_tetes = deuxiemes + troisiemes
    assert len(premiers) == 12 and len(non_tetes) == 20
    paires = []
    tetes_restantes = premiers.copy(); non_tetes_restantes = non_tetes.copy()
    np.random.shuffle(tetes_restantes)
    for tete in tetes_restantes:
        if non_tetes_restantes:
            paires.append((tete, non_tetes_restantes.pop(0)))
    while len(non_tetes_restantes) >= 2:
        paires.append((non_tetes_restantes.pop(0), non_tetes_restantes.pop(0)))
    return paires

def simuler_phase_eliminatoire(simulateur, qualifies_32, equipes_dict):
    np.random.shuffle(qualifies_32)
    p16 = [(qualifies_32[i], qualifies_32[i+1]) for i in range(0, 32, 2)]
    q16 = simuler_tour_eliminatoire(simulateur, p16, equipes_dict)
    p8 = [(q16[i], q16[i+1]) for i in range(0, 16, 2)]
    q8 = simuler_tour_eliminatoire(simulateur, p8, equipes_dict)
    p4 = [(q8[i], q8[i+1]) for i in range(0, 8, 2)]
    q4 = simuler_tour_eliminatoire(simulateur, p4, equipes_dict)
    p2 = [(q4[i], q4[i+1]) for i in range(0, 4, 2)]
    finalistes = simuler_tour_eliminatoire(simulateur, p2, equipes_dict)

    _, _, vainqueur = simulateur.simuler_match(
        equipes_dict[finalistes[0]], equipes_dict[finalistes[1]], True)
    finaliste = finalistes[1] if finalistes[0] == vainqueur else finalistes[0]

    perdants = [f for f in q4 if f not in finalistes]
    troisieme = finalistes[0]
    if len(perdants) >= 2:
        _, _, troisieme = simulateur.simuler_match(
            equipes_dict[perdants[0]], equipes_dict[perdants[1]], True)

    return {
        'vainqueur': vainqueur, 'finaliste': finaliste,
        'troisieme': troisieme, 'demi_finalistes': finalistes + perdants[:2]
    }

# =============================================================================
# SIMULATION PARALLELE AVEC MULTIPROCESSING
# =============================================================================

def run_single_simulation(args):
    simulateur_class, params, seed, n_sims = args

    np.random.seed(seed)
    simulateur = simulateur_class(*params)

    local_compteurs = {
        'vainqueur': defaultdict(int), 'finaliste': defaultdict(int),
        'troisieme': defaultdict(int), 'top4': defaultdict(int),
        'qualifie_16emes': defaultdict(int), 'premier_groupe': defaultdict(int),
        'demi_finale': defaultdict(int)
    }

    for _ in range(n_sims):
        equipes_liste = simulateur.construire_toutes_equipes()
        for eq in equipes_liste: eq.reset_forme()

        equipes_par_groupe = {}
        for eq in equipes_liste:
            equipes_par_groupe.setdefault(eq.groupe, []).append(eq)
        equipes_dict = {eq.nom_fr: eq for eq in equipes_liste}

        _, premiers, deuxiemes, troisiemes = simuler_tous_groupes(simulateur, equipes_par_groupe)
        for p in premiers: local_compteurs['premier_groupe'][p] += 1

        q32 = premiers + deuxiemes + troisiemes
        for q in q32: local_compteurs['qualifie_16emes'][q] += 1

        resultat = simuler_phase_eliminatoire(simulateur, q32, equipes_dict)

        local_compteurs['vainqueur'][resultat['vainqueur']] += 1
        local_compteurs['finaliste'][resultat['finaliste']] += 1
        local_compteurs['troisieme'][resultat['troisieme']] += 1

        top4_set = set([resultat['vainqueur'], resultat['finaliste'],
                       resultat['troisieme']] + resultat['demi_finalistes'])
        for nom in top4_set: local_compteurs['top4'][nom] += 1
        for nom in resultat['demi_finalistes']: local_compteurs['demi_finale'][nom] += 1

    return local_compteurs

def merge_compteurs(compteurs_list):
    merged = {
        'vainqueur': defaultdict(int), 'finaliste': defaultdict(int),
        'troisieme': defaultdict(int), 'top4': defaultdict(int),
        'qualifie_16emes': defaultdict(int), 'premier_groupe': defaultdict(int),
        'demi_finale': defaultdict(int)
    }
    for local in compteurs_list:
        for key in merged:
            for team, count in local[key].items():
                merged[key][team] += count
    return merged

def run_monte_carlo_parallel(simulateur, n_simulations=10000, n_workers=None):
    if n_workers is None:
        n_workers = N_WORKERS

    print(f"Lancement de {n_simulations:,} simulations sur {n_workers} coeurs...")

    sims_per_worker = n_simulations // n_workers
    remainder = n_simulations % n_workers

    params = (simulateur.modele, simulateur.params_norm, simulateur.dernier_elo,
              simulateur.dernier_formes, simulateur.progression_elo, True)

    args_list = []
    for i in range(n_workers):
        n_sims = sims_per_worker + (1 if i < remainder else 0)
        args_list.append((SimulateurTournoi, params, 42 + i, n_sims))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(run_single_simulation, args_list))

    compteurs = merge_compteurs(results)
    return compteurs, n_simulations

# =============================================================================
# VERSION SEQUENTIELLE (fallback)
# =============================================================================

def run_monte_carlo_sequential(simulateur, n_simulations=10000):
    compteurs = {
        'vainqueur': defaultdict(int), 'finaliste': defaultdict(int),
        'troisieme': defaultdict(int), 'top4': defaultdict(int),
        'qualifie_16emes': defaultdict(int), 'premier_groupe': defaultdict(int),
        'demi_finale': defaultdict(int)
    }

    print(f"Lancement de {n_simulations:,} simulations...")
    for i in range(n_simulations):
        if (i + 1) % 1000 == 0:
            print(f"  Simulation {i+1}/{n_simulations}")

        equipes_liste = simulateur.construire_toutes_equipes()
        for eq in equipes_liste: eq.reset_forme()

        equipes_par_groupe = {}
        for eq in equipes_liste:
            equipes_par_groupe.setdefault(eq.groupe, []).append(eq)
        equipes_dict = {eq.nom_fr: eq for eq in equipes_liste}

        _, premiers, deuxiemes, troisiemes = simuler_tous_groupes(simulateur, equipes_par_groupe)
        for p in premiers: compteurs['premier_groupe'][p] += 1

        q32 = premiers + deuxiemes + troisiemes
        for q in q32: compteurs['qualifie_16emes'][q] += 1

        resultat = simuler_phase_eliminatoire(simulateur, q32, equipes_dict)

        compteurs['vainqueur'][resultat['vainqueur']] += 1
        compteurs['finaliste'][resultat['finaliste']] += 1
        compteurs['troisieme'][resultat['troisieme']] += 1

        top4_set = set([resultat['vainqueur'], resultat['finaliste'],
                       resultat['troisieme']] + resultat['demi_finalistes'])
        for nom in top4_set: compteurs['top4'][nom] += 1
        for nom in resultat['demi_finalistes']: compteurs['demi_finale'][nom] += 1

    return compteurs, n_simulations

# =============================================================================
# AFFICHAGE ET EXPORT
# =============================================================================

def afficher_resultats(compteurs, n_sim):
    def show(comp, titre, top_n=15):
        print(f"\n{titre}")
        print("-" * 60)
        items = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        for nom, count in items[:top_n]:
            pct = (count / n_sim) * 100
            se = np.sqrt(pct * (100 - pct) / n_sim)
            ic_low = max(0, pct - 1.96 * se)
            ic_high = min(100, pct + 1.96 * se)
            barre = "█" * int(min(pct, 100) / 2)
            print(f"  {nom:22s} {pct:6.2f}% [{ic_low:5.2f}-{ic_high:5.2f}] {barre}")

    print("\n" + "="*65)
    print("RESULTATS MONTE CARLO - COUPE DU MONDE 2026")
    print(f"   {n_sim:,} simulations | Parallelise + Optimise")
    print("="*65)

    show(compteurs['vainqueur'], "PROBABILITE DE VICTOIRE FINALE", 15)
    show(compteurs['finaliste'], "PROBABILITE D'ATTEINDRE LA FINALE", 15)
    show(compteurs['demi_finale'], "PROBABILITE DEMI-FINALE", 15)
    show(compteurs['troisieme'], "PROBABILITE 3EME PLACE", 15)
    show(compteurs['top4'], "PROBABILITE TOP 4", 15)
    show(compteurs['qualifie_16emes'], "PROBABILITE QUALIFICATION 32EMES", 15)
    show(compteurs['premier_groupe'], "PROBABILITE 1ER DU GROUPE", 15)

def exporter_resultats(compteurs, n_sim, filename="predictions_cdm2026.csv"):
    tous = set()
    for c in compteurs.values(): tous.update(c.keys())
    df = pd.DataFrame({'equipe': sorted(tous)})
    for cat, comp in compteurs.items():
        df[cat] = df['equipe'].map(lambda x: (comp.get(x, 0) / n_sim) * 100)
    df['vainqueur_ic_low'] = df['vainqueur'] - 1.96 * np.sqrt(
        df['vainqueur'] * (100 - df['vainqueur']) / n_sim)
    df['vainqueur_ic_high'] = df['vainqueur'] + 1.96 * np.sqrt(
        df['vainqueur'] * (100 - df['vainqueur']) / n_sim)
    df = df.sort_values('vainqueur', ascending=False)
    df.to_csv(filename, index=False)
    print(f"\nExporte : {filename}")
    return df

# =============================================================================
# EXECUTION PRINCIPALE
# =============================================================================

def main():
    print("="*65)
    print("SIMULATION COUPE DU MONDE 2026 - VERSION OPTIMISEE")
    print(f"   Coeurs disponibles : {mp.cpu_count()}")
    print("="*65)

    print("\nChargement des donnees...")
    data, elo_historique, elo_latest, elo_wc2026 = charger_donnees()

    print("\nCalcul de la progression Elo...")
    progression_elo = calculer_progression_elo(elo_wc2026)

    print("\nPreparation des variables...")
    data = ajouter_elo(data, elo_historique, progression_elo)
    data = ajouter_formes(data)
    data = ajouter_variables_contextuelles(data)

    print("\nConstruction du dataset Poisson...")
    dataset_poisson, params_norm = construire_dataset_poisson(data)
    print(f"Dataset final : {len(dataset_poisson)} observations")

    print("\nEntrainement du modele GLM...")
    modele = entrainer_modele(dataset_poisson)

    print("\nPreparation du simulateur...")
    dernier_elo, dernier_formes, progression_elo_sim = preparer_donnees_historiques(
        data, elo_historique, elo_wc2026)
    simulateur = SimulateurTournoi(modele, params_norm, dernier_elo,
                                    dernier_formes, progression_elo_sim, True)

    try:
        if USE_PARALLEL and mp.cpu_count() > 2:
            print(f"\nLancement des simulations (PARALLELE sur {N_WORKERS} coeurs)...")
            compteurs, n_sim = run_monte_carlo_parallel(simulateur, n_simulations=10000)
        else:
            print("\nLancement des simulations (SEQUENTIEL)...")
            compteurs, n_sim = run_monte_carlo_sequential(simulateur, n_simulations=10000)
    except Exception as e:
        print(f"\nParallelisation echouee ({e}), fallback sequentiel...")
        compteurs, n_sim = run_monte_carlo_sequential(simulateur, n_simulations=10000)

    afficher_resultats(compteurs, n_sim)
    exporter_resultats(compteurs, n_sim)

    print("\n" + "="*65)
    print("Simulation terminee !")
    print("="*65)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()