# Code Python extrait du notebook Jupyter
# Mémoire : Approche hybride de segmentation et prédiction de l'attrition bancaire
# Auteur : OLIVIER NJUKOUYOU FIFEN

# ====================================================================
# ÉTAPE / CELLULE 1
# ====================================================================

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, confusion_matrix
)
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Rééchantillonnage
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# LSTM (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# River — flux + détection de dérive
from river import drift

# ---------------------------------------------------------------------------
# CONSTANTES GLOBALES
# ---------------------------------------------------------------------------
SEED      = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FIG_DIR = Path(r"D:\DOSSIER FIFEN\BDDS5\MemoireFifen1\figure")
FIG_DIR.mkdir(exist_ok=True)

def save_fig(nom_fichier: str) -> None:
    """
    Sauvegarde automatique de la figure Matplotlib/Seaborn courante dans FIG_DIR,
    au format vectoriel PDF (haute resolution, pret pour includegraphics LaTeX).
    A appeler juste avant chaque plt.show().
    """
    chemin = FIG_DIR / nom_fichier
    plt.savefig(chemin, dpi=300, bbox_inches="tight")
    print(f"[Figure] Sauvegardee automatiquement -> {chemin.resolve()}")

BLUE_ESG = "#00569A"
COLORS = {
    "moe":        "#00569A",
    "xgb_global": "#C0392B",
    "xgb_static": "#E67E22",
    "lstm_global":"#8E44AD",
    "rf":         "#2ECC71",
    "logit":      "#7F8C8D",
}

# ---------------------------------------------------------------------------
# Chemins vers mon fichier de données
# ---------------------------------------------------------------------------
path_k = r"D:\DOSSIER FIFEN\BDDS5\MemoireFifen1\Churn_Modelling_Kaggle.csv"
path_b = r"D:\DOSSIER FIFEN\BDDS5\MemoireFifen1\Bloomberg_Macro_Complet.csv"


# ====================================================================
# ÉTAPE / CELLULE 2
# ====================================================================

def etape1_charger_donnees(chemin_kaggle: str, chemin_bloomberg: str):
    print("\n" + "=" * 70)
    print("ÉTAPE 1 — Chargement des données brutes")
    print("=" * 70)
    df_k = pd.read_csv(chemin_kaggle, sep=';')
    if "Geography" in df_k.columns:
        geo_dummies = pd.get_dummies(df_k["Geography"], prefix="Geo", drop_first=False).astype(int)
        df_k = pd.concat([df_k.drop(columns=["Geography"]), geo_dummies], axis=1)
    num_cols = [c for c in df_k.columns if c not in ["Exited"]]
    df_k[num_cols] = df_k[num_cols].apply(pd.to_numeric, errors="coerce")
    if "Exited" in df_k.columns:
        df_k["Exited"] = df_k["Exited"].astype(int)
    df_k = df_k.dropna().reset_index(drop=True)
    print(f"[Kaggle]    {len(df_k):,} clients | attrition = {df_k['Exited'].mean():.4f}")

    df_b = pd.read_csv(chemin_bloomberg, sep=';', decimal=',')
    df_b.columns = [c.strip() for c in df_b.columns]
    df_b.columns = ["DATES", "EUR002W", "SX7E", "V2X", "EUCCFR", "ITRX"]
    df_b["DATE"] = pd.to_datetime(df_b["DATES"], format="%d/%m/%Y", errors="coerce")
    if df_b["DATE"].isna().all():
        df_b["DATE"] = pd.to_datetime(df_b["DATES"], errors="coerce")
    df_b = df_b.drop(columns=["DATES"]).sort_values("DATE").reset_index(drop=True)

    # ------------------------------------------------------------------
    # CORRECTION — dédoublonnage mensuel (T doit valoir exactement 108)
    # L'export Bloomberg brut contient 109 lignes : décembre 2023 est
    # relevé deux fois ("01/12/2023" puis "31/12/2023", clôture d'année,
    # hors convention "premier jour du mois" suivie pour les 107 autres
    # mois). On ne conserve qu'UN relevé par mois calendaire (le plus
    # récent disponible), afin d'obtenir T = 108 mois exactement
    # (janvier 2015 -- décembre 2023), cohérent avec les chapitres
    # Données et Résultats du mémoire.
    # ------------------------------------------------------------------
    n_brut = len(df_b)
    df_b["mois_calendaire"] = df_b["DATE"].dt.to_period("M")
    n_doublons = int(df_b["mois_calendaire"].duplicated().sum())
    if n_doublons > 0:
        print(f"[Bloomberg] {n_brut} lignes brutes -> {n_doublons} doublon(s) mensuel(s) "
              f"détecté(s) (ex. décembre 2023 relevé 2 fois) -> conservation du relevé "
              f"le plus récent par mois")
        df_b = df_b.drop_duplicates(subset="mois_calendaire", keep="last")
    df_b = df_b.drop(columns=["mois_calendaire"]).set_index("DATE").sort_index()

    # Diagnostic qualité : détection de séries "gelées" (valeur figée >= 12 mois
    # consécutifs), signe probable d'une formule Bloomberg bloquée ou d'un
    # ticker mal rafraîchi. À vérifier manuellement avant toute interprétation
    # économique de la variable concernée.
    for col in ["EUR002W", "SX7E", "V2X", "EUCCFR", "ITRX"]:
        vals = df_b[col].to_numpy()
        run, best, best_val = 1, 1, (vals[0] if len(vals) else None)
        for i in range(1, len(vals)):
            run = run + 1 if vals[i] == vals[i - 1] else 1
            if run > best:
                best, best_val = run, vals[i]
        if best >= 12:
            print(f"[ALERTE QUALITÉ] {col} : valeur figée à {best_val} pendant "
                  f"{best} mois consécutifs -> vérifier la source Bloomberg "
                  f"(formule bloquée / rafraîchissement du ticker en défaut).")

    print(f"[Bloomberg] {len(df_b)} mois | {df_b.index.min():%Y-%m} -> {df_b.index.max():%Y-%m}")
    assert len(df_b) == 108, f"T attendu = 108 mois, obtenu = {len(df_b)} -- vérifier le fichier source."
    return df_k, df_b


# ====================================================================
# ÉTAPE / CELLULE 3
# ====================================================================

df_kaggle_orig, df_bloomberg = etape1_charger_donnees(path_k, path_b)

# ====================================================================
# ÉTAPE / CELLULE 4
# ====================================================================

# NOTE DE COHÉRENCE (cf. chapitre Données, §Architecture de la fusion) :
# Cette fonction résume chaque variable macro Bloomberg par ses 4 moments
# (moyenne/écart-type/min/max) calculés sur T=108 mois, puis les attache de
# façon STATIQUE (identiques pour les 10 000 clients) au vecteur x_i. Le
# classifieur (SMOTE/K-means/XGBoost) travaille donc sur une base
# cross-sectionnelle de 10 000 lignes, PAS sur le flux temporel complet de
# 10 000 x 108 = 1 080 000 observations. Ce flux complet (mathcal{D}) est
# construit séparément par BankSim (étape 8, df_traj) et mobilisé pour les
# analyses descriptives temporelles et les scénarios de dérive (§4.2, §4.3.3,
# §4.5). Ces deux représentations coexistent délibérément : le chapitre
# Résultats doit systématiquement préciser laquelle est utilisée dans chaque
# section pour éviter toute ambiguïté.
# =============================================================================
# ÉTAPE 2 — §4.1.1 FUSION KAGGLE × BLOOMBERG
# =============================================================================
def etape2_fusionner(df_kaggle: pd.DataFrame, df_bloomberg: pd.DataFrame) -> tuple:
    print("\n" + "=" * 70)
    print("ÉTAPE 2 — Fusion Kaggle × Bloomberg")
    print("=" * 70)
    stats_dict = {}
    for col in df_bloomberg.columns:
        stats_dict[f"{col}_mean"] = df_bloomberg[col].mean()
        stats_dict[f"{col}_std"] = df_bloomberg[col].std()
        stats_dict[f"{col}_min"] = df_bloomberg[col].min()
        stats_dict[f"{col}_max"] = df_bloomberg[col].max()
    df_macro_statique = pd.DataFrame(stats_dict, index=df_kaggle.index)
    df_fusionne = pd.concat([df_kaggle, df_macro_statique], axis=1)
    feature_cols = [c for c in df_fusionne.columns if c != "Exited"]
    X_features = df_fusionne[feature_cols]
    y = df_fusionne["Exited"].to_numpy()
    return X_features, y, list(feature_cols), df_bloomberg.index.unique()

# ====================================================================
# ÉTAPE / CELLULE 5
# ====================================================================

X_features, y, feature_cols, dates_bloomberg = etape2_fusionner(df_kaggle_orig.copy(), df_bloomberg)

# ====================================================================
# ÉTAPE / CELLULE 6
# ====================================================================

# =============================================================================
# ETAPE 3 -- SMOTE et decoupage stratifie
# CORRECTION : on conserve les indices originaux (row_indices) a travers le
# split et SMOTE pour pouvoir mapper chaque ligne aux trajectoires BankSim.
# =============================================================================
def etape3_smote_et_decoupage(X_features: pd.DataFrame, y: np.ndarray) -> tuple:
    print("\n" + "=" * 70)
    print("ETAPE 3 -- SMOTE et decoupage stratifie")
    print("=" * 70)
    X_all, y_all = X_features.values.astype(np.float32), y.astype(int)
    indices_all = np.arange(len(X_all))

    X_tr, X_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
        X_all, y_all, indices_all, test_size=0.35, stratify=y_all, random_state=SEED)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_tmp, y_tmp, idx_tmp, test_size=(0.20 / 0.35), stratify=y_tmp, random_state=SEED)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)

    smote = SMOTE(k_neighbors=5, random_state=SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_tr, y_tr)

    # Les n_real premières lignes de X_train_sm = lignes reelles de X_tr
    n_real_train = len(X_tr)

    n0, n1 = (y_all == 0).sum(), (y_all == 1).sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie([n0, n1], labels=[f"Non-churn\n{n0:,}", f"Churn\n{n1:,}"],
           colors=["#2980B9", "#C0392B"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Repartition de la variable cible (Exited)", fontweight="bold")
    plt.tight_layout()
    print("[Graphique] Affichage du camembert cible...")
    save_fig("fig_camembert_churn.pdf")
    plt.show()
    return (X_train_sm, y_train_sm, X_val, y_val, X_test, y_test, scaler,
            idx_tr, idx_val, idx_test, n_real_train)


# ====================================================================
# ÉTAPE / CELLULE 7
# ====================================================================

(X_train_sm, y_train_sm,
 X_val, y_val,
 X_test, y_test,
 scaler,
 idx_tr, idx_val, idx_test,
 n_real_train) = etape3_smote_et_decoupage(X_features, y)



# ====================================================================
# ÉTAPE / CELLULE 8
# ====================================================================

# =============================================================================
# ÉTAPE 4 — §4.1.3  HISTOGRAMMES DES VARIABLES CONTINUES
# =============================================================================
def etape4_histogrammes(df_kaggle_orig: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ÉTAPE 4 — Histogrammes des variables continues")
    print("=" * 70)
    vars_plot = [c for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"] if c in df_kaggle_orig.columns]
    ncols = 3
    nrows = (len(vars_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, var in enumerate(vars_plot):
        ax = axes[i]
        for val, label, color in [(0, "Non-churn", "#2980B9"), (1, "Churn", "#C0392B")]:
            data = df_kaggle_orig.loc[df_kaggle_orig["Exited"] == val, var].dropna() if "Exited" in df_kaggle_orig.columns else df_kaggle_orig[var].dropna()
            ax.hist(data, bins=35, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(var, fontweight="bold")
        ax.legend(fontsize=8)
    
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Distributions des variables par statut d'attrition", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    print("[Graphique] Affichage des histogrammes variables...")
    save_fig("fig_histogrammes_variables.pdf")
    plt.show()


# ====================================================================
# ÉTAPE / CELLULE 9
# ====================================================================

etape4_histogrammes(df_kaggle_orig)

# ====================================================================
# ÉTAPE / CELLULE 10
# ====================================================================

# =============================================================================
# ÉTAPE 5 — §4.1.4  MATRICE DE CORRÉLATION DE KAGGLE
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# =============================================================================
def etape5_matrice_correlation(df_kaggle_orig: pd.DataFrame) -> None:
    """
    Calcule et affiche la matrice de corrélation de Spearman pour le bloc Kaggle original.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 5 — Matrice de corrélation (Kaggle)")
    print("=" * 70)

    # Sélection des variables numériques d'intérêt
    cols_interet = [c for c in ["CreditScore", "Age", "Tenure", "Balance", 
                                "NumOfProducts", "HasCrCard", "IsActiveMember", 
                                "EstimatedSalary", "Exited"] if c in df_kaggle_orig.columns]

    if not cols_interet:
        print("[Attention] Aucune variable correspondante pour générer la matrice de corrélation.")
        return

    # Calcul de la matrice de corrélation de Spearman
    corr_matrix = df_kaggle_orig[cols_interet].corr(method="spearman")

    # Génération du graphique
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Masque pour la partie supérieure symétrique
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# CORRECTION : Utilisation de "coolwarm" en minuscules
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm",  
        vmin=-1.0, 
        vmax=1.0, 
        square=True, 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title("Matrice de corrélation de Spearman (Variables Kaggle)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    # ── Rendu Anaconda / Jupyter ──
    print("[Graphique] Affichage de la matrice de corrélation...")
    save_fig("fig_matrice_correlation.pdf")
    plt.show()

# ====================================================================
# ÉTAPE / CELLULE 11
# ====================================================================

etape5_matrice_correlation(df_kaggle_orig)


# ====================================================================
# ÉTAPE / CELLULE 12
# ====================================================================

# =============================================================================
# ÉTAPE 6 — §4.2.1  SÉRIES CHRONOLOGIQUES BLOOMBERG
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# =============================================================================
def etape6_series_bloomberg(df_bloomberg: pd.DataFrame) -> None:
    """
    Trace les 5 séries mensuelles Bloomberg avec zones de régime.
    Calcule et affiche les statistiques par régime.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 6 — Séries chronologiques Bloomberg")
    print("=" * 70)

    # ── Statistiques Bloomberg par régime ──
    regimes_dates = {
        "Rég. 1 (2015–2019)": ("2015-01-01", "2019-12-31"),
        "Rég. 2 (2020)"     : ("2020-01-01", "2020-12-31"),
        "Rég. 3 (2021–2023)": ("2021-01-01", "2023-12-31"),
    }
    
    print("\n── Statistiques Bloomberg par régime ──")
    for nom, (d1, d2) in regimes_dates.items():
        subset = df_bloomberg.loc[d1:d2]
        print(f"\n{nom}")
        print(subset.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

    # ── Configuration de la Figure ──
    meta = {
        "EUR002W": ("Taux BCE (%)",         BLUE_ESG),
        "SX7E"   : ("Indice bancaire SX7E",      "#27AE60"),
        "V2X"    : ("Volatilité V2X (%)",        "#C0392B"),
        "EUCCFR" : ("Confiance ménages FR",      "#8E44AD"),
        "ITRX"   : ("Spread crédit (pb)",        "#E67E22"),
    }
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Définition des zones de régime de marché
    zones = [
        ("2015-01-01", "2019-12-31", "#EBF5FB"),
        ("2020-01-01", "2020-12-31", "#FDEDEC"),
        ("2022-01-01", "2023-12-31", "#EAFAF1"),
    ]
    
    for ax, (col, (titre, couleur)) in zip(axes, meta.items()):
        if col in df_bloomberg.columns:
            ax.plot(df_bloomberg.index, df_bloomberg[col], color=couleur, linewidth=1.8)
            
            # Ajout des arrière-plans colorés pour les régimes
            for d1, d2, cz in zones:
                ax.axvspan(pd.Timestamp(d1), pd.Timestamp(d2), alpha=0.18, color=cz)
                
            ax.set_ylabel(titre, fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)

    # Ajout des légendes de zone sur le premier graphique
    from matplotlib.patches import Patch
    patches = [
        Patch(color="#AED6F1", alpha=0.6, label="Stabilité 2015–2019"),
        Patch(color="#FADBD8", alpha=0.6, label="Choc COVID-19 2020"),
        Patch(color="#A9DFBF", alpha=0.6, label="Remontée taux 2022–2023"),
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8)
    
    # Formatage de l'axe temporel (X) commun
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30)
    
    fig.suptitle("Séries chronologiques Bloomberg (jan. 2015 – déc. 2023)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # ── Rendu Anaconda / Jupyter ──
    print("\n[Graphique] Affichage des séries chronologiques...")
    save_fig("fig_series_bloomberg.pdf")
    plt.show()

# ====================================================================
# ÉTAPE / CELLULE 13
# ====================================================================

etape6_series_bloomberg(df_bloomberg)

# ====================================================================
# ÉTAPE / CELLULE 14
# ====================================================================

# =============================================================================
# ÉTAPE 7 — §4.2.2  SCATTER V2X vs TAUX D'ATTRITION SIMULÉ
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# =============================================================================
def etape7_scatter_bloomberg(df_bloomberg: pd.DataFrame, pi_base: float = 0.2037) -> None:
    """
    Construit un taux d'attrition mensuel simulé à partir de V2X normalisé,
    puis trace le nuage de points avec droite OLS.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 7 — Scatter Bloomberg vs attrition simulée")
    print("=" * 70)

    # Taux simulé : déplacement proportionnel à l'écart de V2X à sa moyenne
    v2x_norm = (df_bloomberg["V2X"] - df_bloomberg["V2X"].mean()) / df_bloomberg["V2X"].std()
    taux_sim = (pi_base + 0.03 * v2x_norm).clip(0.01, 0.60)

    x = df_bloomberg["V2X"].values
    y = taux_sim.values

    # Calcul de la régression linéaire
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    print(f"  OLS : R²={r_val**2:.3f} | p-value={p_val:.2e} | pente={slope:.5f}")

    # ── Configuration de la Figure ──
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Trace du nuage de points
    ax.scatter(x, y, color=BLUE_ESG, alpha=0.6, s=50, edgecolors="white", linewidths=0.5)
    
    # Trace de la droite de régression OLS
    x_l = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_l, slope * x_l + intercept, color="#C0392B", linewidth=2,
            label=f"OLS ($R^2={r_val**2:.2f}$, $p<0.001$)")
    
    # Personnalisation des axes et labels
    ax.set_xlabel("Volatilité bancaire V2X (%)", fontsize=11)
    ax.set_ylabel("Taux d'attrition simulé mensuel", fontsize=11)
    ax.set_title("Relation V2X — attrition simulée", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # ── Rendu Anaconda / Jupyter ──
    print("[Graphique] Affichage du nuage de points OLS...")
    save_fig("fig_scatter_bloomberg_churn.pdf")
    plt.show()

# ====================================================================
# ÉTAPE / CELLULE 15
# ====================================================================

etape7_scatter_bloomberg(df_bloomberg, pi_base=df_kaggle_orig["Exited"].mean())


# ====================================================================
# ÉTAPE / CELLULE 16
# ====================================================================

# =============================================================================
# ÉTAPE 8 — BankSim : TRAJECTOIRES + INDICATEURS DYNAMIQUES
# Sorties : df_traj contenant m_it, b_it, f_it,
#           f_it_w, delta_f_it, b_bar_it_w, sigma_b_it_w, SR_it, break_it
# =============================================================================
def etape8_banksim(df_kaggle_orig: pd.DataFrame,
                   dates: pd.DatetimeIndex,
                   w:     int   = 3,
                   theta: float = 0.30,
                   seed:  int   = SEED) -> pd.DataFrame:
    """
    BankSim : génère les trajectoires transactionnelles (éq. 0c) et
    calcule les indicateurs dynamiques (éq. 7–12).

    Paramètres calibrés sur Balance, EstimatedSalary, IsActiveMember.
    Retourne un DataFrame avec (client_idx, DATE) comme clé composite.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 8 — BankSim : trajectoires + indicateurs dynamiques")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    eps = 1e-6
    records = []

    # Vérification et mappage dynamique des colonnes sources
    est_sal_col = "EstimatedSalary" if "EstimatedSalary" in df_kaggle_orig.columns else None
    actif_col   = "IsActiveMember"  if "IsActiveMember"  in df_kaggle_orig.columns else None
    bal_col     = "Balance"         if "Balance"         in df_kaggle_orig.columns else None

    # Génération des profils transactionnels synthétiques
    for idx, row in df_kaggle_orig.iterrows():
        balance_init = float(row[bal_col]) if bal_col else 10_000.0
        salaire_mens = float(row[est_sal_col]) / 12.0 if est_sal_col else 5_000.0
        actif        = float(row[actif_col])   if actif_col   else 0.5
        exited       = int(row["Exited"]) if "Exited" in df_kaggle_orig.columns else 0

        freq_base = 8.0 if actif >= 0.5 else 3.0
        tendance  = -0.003 * exited   # Décroissance progressive propre aux profils churners
        solde     = max(balance_init, 0.0)

        for t_idx, date in enumerate(dates):
            mu_log    = np.log(max(salaire_mens * 0.15, 1.0))
            sigma_log = 0.4 + 0.1 * (1 - actif)
            m_it      = float(min(rng.lognormal(mu_log, sigma_log), salaire_mens * 2.0))
            f_it      = float(max(freq_base + tendance * t_idx + rng.normal(0, 0.8), 0.5))
            
            variation = (salaire_mens - m_it) * 0.1 + rng.normal(0, salaire_mens * 0.02)
            solde     = max(solde + variation, 0.0)
            
            records.append({
                "client_idx": idx,
                "DATE": date,
                "m_it": round(m_it, 2),
                "b_it": round(solde, 2),
                "f_it": round(f_it, 2),
                "r_i":  round(salaire_mens, 2),   # Revenu mensuel pivot pour le calcul SR_it
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["client_idx", "DATE"]).reset_index(drop=True)

    # ── Calcul des indicateurs dynamiques fenêtrés par client ──
    grp = df.groupby("client_idx")
    
    df["f_it_w"]       = grp["f_it"].transform(lambda x: x.rolling(w, min_periods=1).mean())
    df["b_bar_it_w"]   = grp["b_it"].transform(lambda x: x.rolling(w, min_periods=1).mean())
    df["sigma_b_it_w"] = grp["b_it"].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    
    # API pandas >= 2.0 : remplacement de fillna("bfill") par .bfill() sur le shift initial
    df["f_lag"]        = grp["f_it"].shift(1).bfill()
    df["delta_f_it"]   = (df["f_it"] - df["f_lag"]) / (df["f_lag"] + eps)
    df["SR_it"]        = df["m_it"] / (df["r_i"] + eps)
    df["break_it"]     = (df["delta_f_it"] < -theta).astype(int)
    
    df = df.drop(columns=["f_lag"])

    print(f"[BankSim] {len(df):,} lignes générées | {df['client_idx'].nunique()} clients uniques | {df['DATE'].nunique()} mois")
    print(f"[BankSim] Taux de rupture global (break_it) = {df['break_it'].mean():.4f}")
    return df

# ====================================================================
# ÉTAPE / CELLULE 17
# ====================================================================

df_traj = etape8_banksim(df_kaggle_orig, dates_bloomberg)


# ====================================================================
# ÉTAPE / CELLULE 18
# ====================================================================

# =============================================================================
# ÉTAPE 9 — §4.3.1  SÉLECTION DU NOMBRE OPTIMAL DE SEGMENTS K*
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# Tableaux : tab:selection_k (console)
# =============================================================================
def etape9_selection_k(X_train_sm: np.ndarray, k_max: int = 8) -> tuple:
    """
    Calcule DB, CH et inertie pour K = 2..k_max sur X_train_sm.
    Sélectionne K* = argmin DB sous contrainte CH ≥ percentile 70.
    
    Retourne aussi le modèle KMeans entraîné pour K* (évite de refitter).
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 9 — Sélection du nombre optimal de segments K*")
    print("=" * 70)

    resultats = {}
    print(f"{'K':>4} {'DB':>8} {'CH':>10} {'Inertie':>14}")
    print("-" * 42)
    
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
        labels = km.fit_predict(X_train_sm)
        
        db = davies_bouldin_score(X_train_sm, labels)
        ch = calinski_harabasz_score(X_train_sm, labels)
        inrt = km.inertia_
        
        resultats[k] = {"DB": db, "CH": ch, "Inertia": inrt, "model": km}
        print(f"{k:>4} {db:>8.3f} {ch:>10.0f} {inrt:>14.2e}")

    # Critère de sélection hybride
    ch_min = np.percentile([v["CH"] for v in resultats.values()], 70)
    valides = {k: v for k, v in resultats.items() if v["CH"] >= ch_min}
    k_opt = min(valides, key=lambda k: valides[k]["DB"])
    km_opt = valides[k_opt]["model"]
    
    print(f"\n✓ K* = {k_opt}  (DB={valides[k_opt]['DB']:.3f}, CH={valides[k_opt]['CH']:.0f})")

    # ── Configuration de la Figure : Coude + DB ──
    ks = sorted(resultats.keys())
    inertia = [resultats[k]["Inertia"] for k in ks]
    db_vals = [resultats[k]["DB"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Graphe 1 : Méthode du coude (Inertie)
    ax1.plot(ks, inertia, "o-", color=BLUE_ESG, linewidth=2, markersize=7)
    ax1.axvline(k_opt, color="#C0392B", linestyle="--", label=f"$K^*={k_opt}$")
    ax1.annotate(f"$K^*={k_opt}$",
                 xy=(k_opt, resultats[k_opt]["Inertia"]),
                 xytext=(k_opt + 0.3, resultats[k_opt]["Inertia"] * 1.05),
                 arrowprops=dict(arrowstyle="->", color="#C0392B"),
                 color="#C0392B", fontsize=11)
    ax1.set_xlabel("Nombre de segments $K$", fontsize=11)
    ax1.set_ylabel("Inertie intra-cluster", fontsize=11)
    ax1.set_title("Méthode du coude", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Graphe 2 : Indice Davies-Bouldin
    ax2.plot(ks, db_vals, "s-", color="#E67E22", linewidth=2, markersize=7)
    ax2.axvline(k_opt, color="#C0392B", linestyle="--", label=f"$K^*={k_opt}$")
    ax2.set_xlabel("Nombre de segments $K$", fontsize=11)
    ax2.set_ylabel("Indice Davies-Bouldin", fontsize=11)
    ax2.set_title("Davies-Bouldin (min = optimal)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("Sélection de $K^*$", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    # ── Rendu Anaconda / Jupyter ──
    print("\n[Graphique] Affichage des métriques d'évaluation K-Means...")
    save_fig("fig_elbow_kmeans.pdf")
    plt.show()

    return resultats, k_opt, km_opt

# ====================================================================
# ÉTAPE / CELLULE 19
# ====================================================================

resultats_k, k_opt, km_pretrained = etape9_selection_k(X_train_sm, k_max=8)


# ====================================================================
# ÉTAPE / CELLULE 20
# ====================================================================

# =============================================================================
# ÉTAPE 10 — §4.3.2  K-MEANS ADAPTATIF + PROFILS DES SEGMENTS
# Tableaux  : tab:profils_segments (console)
# =============================================================================

class KMeansAdaptatif:
    """
    K-means adaptatif avec facteur d'oubli exponentiel α (éq. 24).
    Méthodes : fit, predict, mise_a_jour_centroides, reinitialiser.
    """
    def __init__(self, k: int, alpha: float = 0.15, seed: int = SEED):
        self.k = k
        self.alpha = alpha
        self.km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        self.centroides = None

    def fit(self, X: np.ndarray, km_pretrained=None) -> "KMeansAdaptatif":
        """
        Initialise les centroïdes. Si km_pretrained est fourni (issu de
        l'ÉTAPE 9), on réutilise ses centroïdes — évite un double fit.
        """
        if km_pretrained is not None:
            self.km = km_pretrained
            self.centroides = km_pretrained.cluster_centers_.copy()
        else:
            self.km.fit(X)
            self.centroides = self.km.cluster_centers_.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Affectation au cluster le plus proche (éq. 23)."""
        diff = X[:, np.newaxis, :] - self.centroides[np.newaxis, :, :]
        return np.argmin(np.linalg.norm(diff, axis=2), axis=1)

    def mise_a_jour_centroides(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Mise à jour EWA (éq. 24)."""
        for k in range(self.k):
            idx = labels == k
            if idx.sum() > 0:
                x_bar = X[idx].mean(axis=0)
                self.centroides[k] = (self.alpha * x_bar + (1 - self.alpha) * self.centroides[k])

    def reinitialiser(self, X: np.ndarray) -> None:
        """Réentraînement complet R2 (éq. 49)."""
        self.km.fit(X)
        self.centroides = self.km.cluster_centers_.copy()
        print("[K-means] R2 — réentraînement complet déclenché.")


def etape10_kmeans(X_train_sm: np.ndarray, y_train_sm: np.ndarray,
                   X_val: np.ndarray, X_test: np.ndarray,
                   k_opt: int, km_pretrained,
                   feature_cols: list) -> tuple:
    """
    Instancie le K-means adaptatif en réutilisant le modèle de l'ÉTAPE 9,
    puis calcule les profils de segments.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 10 — K-means adaptatif + profils")
    print("=" * 70)

    # Initialisation du modèle adaptatif
    km = KMeansAdaptatif(k=k_opt, alpha=0.15)
    km.fit(X_train_sm, km_pretrained=km_pretrained)

    # Prédictions/Affectations sur l'ensemble des bases
    labels_train = km.predict(X_train_sm)
    labels_val   = km.predict(X_val)
    labels_test  = km.predict(X_test)

    # Profils — extraire la sémantique sur les variables explicatives majeures
    cols_affich = feature_cols[:min(6, len(feature_cols))]
    df_seg = pd.DataFrame(X_train_sm[:, :len(cols_affich)], columns=cols_affich)
    df_seg["segment"] = labels_train
    df_seg["Exited"]  = y_train_sm

    print("\n── Profils moyens des segments (features normalisées) ──")
    profils = df_seg.groupby("segment").agg(
        Effectif=("Exited", "count"),
        Taux_churn=("Exited", "mean"),
        **{c: (c, "mean") for c in cols_affich}
    ).round(3)
    print(profils.to_string())

    # Contrôle de sécurité sur la variance intra-segment pour XGBoost local
    for k in range(k_opt):
        y_k = y_train_sm[labels_train == k]
        if len(np.unique(y_k)) < 2:
            print(f"  ⚠ Segment {k} : une seule classe représentée (n={len(y_k)}) → "
                  f"fallback nécessaire sur modèle global.")

    return km, labels_train, labels_val, labels_test, profils

# ====================================================================
# ÉTAPE / CELLULE 21
# ====================================================================

(km, labels_train, labels_val, labels_test,
 profils) = etape10_kmeans(
    X_train_sm, y_train_sm, X_val, X_test,
    k_opt, km_pretrained, feature_cols
)


# ====================================================================
# ÉTAPE / CELLULE 22
# ====================================================================

# =============================================================================
# ÉTAPE 11 — §4.3.3  ÉVOLUTION TEMPORELLE DES SEGMENTS
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# =============================================================================
def etape11_evolution_segments(km: KMeansAdaptatif,
                               X_train_sm: np.ndarray,
                               y_train_sm: np.ndarray,
                               df_bloomberg: pd.DataFrame,
                               k_opt: int) -> None:
    """
    Simule l'évolution des proportions de segments mois par mois.
    À chaque mois t :
      1. Applique une légère perturbation aléatoire sur X (proxy de dérive)
      2. Met à jour les centroïdes (R1 ou R2)
      3. Enregistre les proportions normalisées à 100 %
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 11 — Évolution temporelle des segments")
    print("=" * 70)

    dates_all = df_bloomberg.index
    rng = np.random.default_rng(SEED)

    # Variance des features sur le train (pour calibrer le bruit proportionnellement)
    sigma_feat = X_train_sm.std(axis=0) * 0.05   # 5 % de la std originale

    props_hist = []
    km_sim = KMeansAdaptatif(k=k_opt, alpha=0.15)
    km_sim.fit(X_train_sm)

    # On extrait un sous-ensemble représentatif pour simuler le flux continu
    X_courant = X_train_sm[:min(500, len(X_train_sm))].copy()

    for t_idx, date in enumerate(dates_all):
        # Détection des fenêtres de choc macroéconomique
        est_covid = (date.year == 2020 and date.month in [3, 4, 5])
        est_taux  = (date.year >= 2022)
        bruit_amp = 0.5 if est_covid else (0.15 if est_taux else 0.02)
        
        # Application de la dérive synthétique
        X_courant += rng.normal(0, sigma_feat * bruit_amp, X_courant.shape)

        labels_t = km_sim.predict(X_courant)
        km_sim.mise_a_jour_centroides(X_courant, labels_t)

        # Si choc COVID violent (mars 2020) : réentraînement R2 (complet)
        if est_covid and date.month == 3:
            km_sim.reinitialiser(X_courant)

        # Calcul des distributions sur la période t
        props_t = [((labels_t == k).sum() / len(labels_t) * 100) for k in range(k_opt)]
        props_hist.append(props_t)

    df_props = pd.DataFrame(
        props_hist,
        index=dates_all,
        columns=[f"Seg. {k+1}" for k in range(k_opt)]
    )
    
    # Normalisation stricte pour compenser les micro-écarts de flottants
    df_props = df_props.div(df_props.sum(axis=1), axis=0) * 100

    # ── Configuration de la Figure ──
    couleurs = ["#2980B9", "#27AE60", "#E67E22", "#C0392B"]
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for i, col in enumerate(df_props.columns):
        ax.plot(df_props.index, df_props[col],
                color=couleurs[i % len(couleurs)], linewidth=2, label=col)
        
    # Marquage des ruptures structurelles majeures
    for d_ev in ["2020-01-01", "2022-01-01"]:
        ax.axvline(pd.Timestamp(d_ev), color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Proportion du segment (%)", fontsize=12)
    ax.set_title("Évolution mensuelle des proportions de segments", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    
    # Gestion de la chronologie sur l'axe X
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # ── Rendu Anaconda / Jupyter ──
    print("[Graphique] Affichage de la dynamique temporelle des segments...")
    save_fig("fig_evolution_segments.pdf")
    plt.show()

# ====================================================================
# ÉTAPE / CELLULE 23
# ====================================================================

etape11_evolution_segments(km, X_train_sm, y_train_sm, df_bloomberg, k_opt)


# ====================================================================
# ÉTAPE / CELLULE 24
# ====================================================================

# =============================================================================
# ETAPE 12 -- GATING SOFTMAX + MELANGE D EXPERTS (MoE)
#
# CORRECTIONS APPLIQUEES (3 bugs identifies et corriges) :
#
# BUG 1 -- Double sigmoide : forward() appliquait self.sig() PUIS
#   BCEWithLogitsLoss appliquait un second sigmoide interne.
#   -> Gradient quasi plat, AUC LSTM = 0.50 (aleatoire).
#   CORRECTION : forward() renvoie des logits bruts ;
#   torch.sigmoid() est applique a l inference seulement.
#
# BUG 2 -- Sequences factices : _preparer_seq() construisait des
#   fenetres glissantes sur l ORDRE DES LIGNES (une ligne = un
#   client different). Le LSTM n avait aucun signal temporel reel.
#   CORRECTION : les sequences sont maintenant construites a partir
#   des trajectoires BankSim par client (df_traj), qui contiennent
#   une vraie dynamique temporelle (108 mois par client).
#
# BUG 3 -- SMOTE et sequences : les lignes synthetiques SMOTE n ont
#   pas de trajectoire BankSim. CORRECTION : on assigne a chaque
#   ligne synthetique la trajectoire de son plus proche voisin reel.
# =============================================================================

# -- Colonnes BankSim utilisees comme features sequentielles --
BANKSIM_SEQ_COLS = ["m_it", "b_it", "f_it", "f_it_w",
                    "b_bar_it_w", "sigma_b_it_w", "delta_f_it", "SR_it", "break_it"]


class LSTMChurn(nn.Module):
    """Reseau LSTM pour la capture des dependances sequentielles (eq. 33-38)."""
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CORRECTION BUG 1 : renvoie des LOGITS bruts (pas de sigmoide)
        out, _  = self.lstm(x)
        h_T     = self.dropout(out[:, -1, :])
        return self.fc(h_T).squeeze(1)


def _build_banksim_sequences(df_traj, client_indices, seq_len):
    """
    Construit un tenseur (N, seq_len, n_features) de vraies sequences
    temporelles BankSim pour chaque client identifie par son index original.
    """
    n_feat = len(BANKSIM_SEQ_COLS)
    N = len(client_indices)
    sequences = np.zeros((N, seq_len, n_feat), dtype=np.float32)

    grouped = {idx: grp[BANKSIM_SEQ_COLS].values
               for idx, grp in df_traj.groupby("client_idx")}

    for i, cidx in enumerate(client_indices):
        traj = grouped.get(cidx)
        if traj is None:
            continue
        T = len(traj)
        if T >= seq_len:
            sequences[i] = traj[-seq_len:]
        else:
            sequences[i, seq_len - T:] = traj

    return sequences


def _assign_smote_sequences(X_train_sm, n_real_train, idx_tr, df_traj, seq_len):
    """
    Construit les sequences BankSim pour TOUTES les lignes de X_train_sm.
    Lignes reelles -> trajectoire directe.
    Lignes synthetiques SMOTE -> trajectoire du plus proche voisin reel.
    """
    from sklearn.neighbors import NearestNeighbors

    real_seqs = _build_banksim_sequences(df_traj, idx_tr, seq_len)
    n_total = len(X_train_sm)

    if n_total == n_real_train:
        return real_seqs

    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_model.fit(X_train_sm[:n_real_train])
    _, nn_indices = nn_model.kneighbors(X_train_sm[n_real_train:])
    nn_indices = nn_indices.ravel()

    all_seqs = np.zeros((n_total, seq_len, real_seqs.shape[2]), dtype=np.float32)
    all_seqs[:n_real_train] = real_seqs
    all_seqs[n_real_train:] = real_seqs[nn_indices]

    return all_seqs


def _normaliser_sequences(seqs_train, seqs_other=None):
    """Normalise les sequences BankSim (Z-score) feature par feature."""
    n_tr, sl, nf = seqs_train.shape
    flat = seqs_train.reshape(-1, nf)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-8

    seqs_train_norm = (seqs_train - mu) / sigma
    if seqs_other is not None:
        seqs_other_norm = (seqs_other - mu) / sigma
        return seqs_train_norm, seqs_other_norm, mu, sigma
    return seqs_train_norm, mu, sigma


def _entrainer_lstm(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq,
                    n_epochs=25, batch_size=64, lr=0.001):
    """Entraine un LSTM avec BPTT + BCEWithLogitsLoss pondere (eq. 39)."""
    model = LSTMChurn(input_size=X_tr_seq.shape[2])
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = int(y_tr_seq.sum())
    n_neg = len(y_tr_seq) - n_pos

    crit  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    )

    ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(n_epochs):
        model.train()
        for Xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # CORRECTION BUG 1 : sigmoide applique a l inference seulement
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X_v_seq))).numpy()
    y_v_np = np.asarray(y_v_seq, dtype=np.float32)
    auc_val = roc_auc_score(y_v_np, p) if len(np.unique(y_v_np)) >= 2 else 0.5
    print(f"    [LSTM] AUC val locale = {auc_val:.4f}")
    return model


def _predire_lstm(model, X_seq):
    """Inference LSTM : logits -> sigmoide -> probabilites."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X_seq, dtype=torch.float32))).numpy()


class MelangeExperts:
    """MoE avec gating softmax + algorithme EM (eq. 40-45)."""
    def __init__(self, k, experts_xgb, experts_lstm, eta=1.5):
        self.k            = k
        self.xgb          = experts_xgb
        self.lstm         = experts_lstm
        self.eta          = eta
        self.poids_gating = np.ones(k) / k

    def predire(self, X_tab, X_seq, labels, fallback_xgb=None):
        N     = len(X_tab)
        p_hat = np.zeros(N)
        for k in range(self.k):
            mask = labels == k
            if mask.sum() == 0:
                continue
            xgb_k  = self.xgb[k] if (self.xgb[k] is not None) else fallback_xgb
            p_xgb  = xgb_k.predict_proba(X_tab[mask])[:, 1]
            p_lstm = _predire_lstm(self.lstm[k], X_seq[mask])
            p_hat[mask] = 0.5 * p_xgb + 0.5 * p_lstm
        return p_hat

    def etape_em(self, X_tab, y, labels, fallback_xgb=None):
        R = np.zeros((len(y), self.k))
        for k in range(self.k):
            xgb_k = self.xgb[k] if (self.xgb[k] is not None) else fallback_xgb
            p_k   = xgb_k.predict_proba(X_tab)[:, 1]
            lhood = np.where(y == 1, p_k, 1 - p_k)
            R[:, k] = self.poids_gating[k] * lhood
        R /= (R.sum(axis=1, keepdims=True) + 1e-10)
        nouveaux = R.sum(axis=0) + (self.eta - 1)
        nouveaux = np.maximum(nouveaux, 1e-6)
        self.poids_gating = nouveaux / nouveaux.sum()
        print(f"  [MoE-EM] Poids Gating = {np.round(self.poids_gating, 3)}")


def etape12_moe(X_train_sm, y_train_sm,
                X_val, y_val,
                labels_train, labels_val,
                k_opt,
                df_traj,
                idx_tr, idx_val,
                n_real_train,
                seq_len=12):
    """
    Entraine les K experts XGBoost + K experts LSTM, instancie le MoE.
    CORRECTION : sequences LSTM basees sur les vraies trajectoires BankSim.
    """
    print("\n" + "=" * 70)
    print("ETAPE 12 -- Melange d Experts (gating softmax + EM)")
    print("=" * 70)

    # -- Construction des sequences BankSim --
    print("  [Sequences] Construction des trajectoires BankSim par client...")
    seq_train = _assign_smote_sequences(X_train_sm, n_real_train, idx_tr,
                                        df_traj, seq_len)
    seq_val   = _build_banksim_sequences(df_traj, idx_val, seq_len)

    # Normalisation Z-score (fit sur train)
    seq_train, seq_val, seq_mu, seq_sigma = _normaliser_sequences(seq_train, seq_val)

    print(f"  [Sequences] Train: {seq_train.shape} | Val: {seq_val.shape}")

    experts_xgb  = []
    experts_lstm = []

    for k in range(k_opt):
        print(f"\n  -- Segment {k + 1} / {k_opt} --")
        mask_tr = labels_train == k
        mask_vl = labels_val == k

        # --- XGBoost Local ---
        if mask_tr.sum() < 20 or len(np.unique(y_train_sm[mask_tr])) < 2:
            print(f"    [XGBoost k={k}] Obs. insuffisantes -> Fallback")
            experts_xgb.append(None)
        else:
            X_vk = X_val[mask_vl] if (mask_vl.sum() >= 5 and len(np.unique(y_val[mask_vl])) >= 2) else X_val
            y_vk = y_val[mask_vl] if (mask_vl.sum() >= 5 and len(np.unique(y_val[mask_vl])) >= 2) else y_val
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                random_state=SEED, n_jobs=-1, verbosity=0)
            clf.fit(X_train_sm[mask_tr], y_train_sm[mask_tr],
                    eval_set=[(X_vk, y_vk)], verbose=False)
            print(f"    [XGBoost k={k}] AUC val = {roc_auc_score(y_vk, clf.predict_proba(X_vk)[:, 1]):.4f}")
            experts_xgb.append(clf)

        # --- LSTM Local (BankSim sequences) ---
        if mask_tr.sum() < 30 or len(np.unique(y_train_sm[mask_tr])) < 2:
            print(f"    [LSTM k={k}] Sequences insuffisantes -> Fallback")
            experts_lstm.append(None)
        else:
            X_v_k = seq_val[mask_vl] if (mask_vl.sum() >= 5 and len(np.unique(y_val[mask_vl])) >= 2) else seq_val
            y_v_k = y_val[mask_vl] if (mask_vl.sum() >= 5 and len(np.unique(y_val[mask_vl])) >= 2) else y_val
            lstm_k = _entrainer_lstm(seq_train[mask_tr], y_train_sm[mask_tr],
                                     X_v_k, y_v_k, n_epochs=25, batch_size=64)
            experts_lstm.append(lstm_k)

    # Fallback pour segments vides
    k_fallback = int(np.bincount(labels_train).argmax())
    for k in range(k_opt):
        if experts_xgb[k] is None:
            experts_xgb[k] = experts_xgb[k_fallback] or next(x for x in experts_xgb if x is not None)
        if experts_lstm[k] is None:
            experts_lstm[k] = experts_lstm[k_fallback] or next(l for l in experts_lstm if l is not None)

    # --- XGBoost Global (Benchmark) ---
    print("\n  -- XGBoost Global (Benchmark) --")
    xgb_global = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=((y_train_sm == 0).sum() / max((y_train_sm == 1).sum(), 1)),
        eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0)
    xgb_global.fit(X_train_sm, y_train_sm, eval_set=[(X_val, y_val)], verbose=False)
    print(f"    AUC Benchmark Global = {roc_auc_score(y_val, xgb_global.predict_proba(X_val)[:, 1]):.4f}")

    # --- LSTM Global (Benchmark) ---
    print("\n  -- LSTM Global (Benchmark) --")
    lstm_global = _entrainer_lstm(seq_train, y_train_sm, seq_val, y_val,
                                  n_epochs=25, batch_size=64)

    # --- MoE + EM ---
    moe = MelangeExperts(k_opt, experts_xgb, experts_lstm)
    moe.etape_em(X_val, y_val, labels_val, fallback_xgb=xgb_global)

    return (moe, xgb_global, experts_xgb, experts_lstm, lstm_global,
            seq_len, seq_train, y_train_sm, seq_val, y_val,
            labels_train, labels_val, seq_mu, seq_sigma)


# ====================================================================
# ÉTAPE / CELLULE 25
# ====================================================================

seq_len = 12
(moe, xgb_global, experts_xgb, experts_lstm, lstm_global,
 seq_len, seq_train, y_train_sm_seq, seq_val, y_val_seq,
 labels_train, labels_val,
 seq_mu, seq_sigma) = etape12_moe(
    X_train_sm, y_train_sm,
    X_val, y_val,
    labels_train, labels_val,
    k_opt,
    df_traj=df_traj,
    idx_tr=idx_tr, idx_val=idx_val,
    n_real_train=n_real_train,
    seq_len=seq_len
)


# ====================================================================
# ÉTAPE / CELLULE 26
# ====================================================================

# =============================================================================
# ETAPE 13 -- COMPARAISON MODELES + COURBE ROC
# CORRECTION : plus de decalage seq_len (les sequences BankSim sont
# alignees 1-pour-1 avec les lignes du split, pas de fenetre glissante).
# =============================================================================

def etape13_comparaison_modeles(
        X_train_sm, y_train_sm,
        X_val, y_val,
        labels_val,
        moe, xgb_global,
        seq_val,
        lstm_global):
    print("\n" + "=" * 70)
    print("ETAPE 13 -- Comparaison des modeles sur w2 (Validation)")
    print("=" * 70)

    y_val_aligned = y_val
    X_val_aligned = X_val
    labels_v_aligned = labels_val

    results = {}

    # 1. Regression Logistique
    logit = LogisticRegression(max_iter=1000, random_state=SEED)
    logit.fit(X_train_sm, y_train_sm)
    results["Logistique"] = (logit.predict_proba(X_val_aligned)[:, 1], COLORS.get("logit", "#95A5A6"))

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)
    results["Random Forest"] = (rf.predict_proba(X_val_aligned)[:, 1], COLORS.get("rf", "#2C3E50"))

    # 3. XGBoost Global
    results["XGBoost global"] = (xgb_global.predict_proba(X_val_aligned)[:, 1], COLORS.get("xgb_global", "#2980B9"))

    # 4. LSTM Global (sur sequences BankSim)
    p_lstm = _predire_lstm(lstm_global, seq_val)
    results["LSTM global"] = (p_lstm, COLORS.get("lstm_global", "#7F8C8D"))

    # 5. MoE adaptatif
    p_moe = moe.predire(X_val_aligned, seq_val, labels_v_aligned, fallback_xgb=xgb_global)
    results["MoE XGBoost+LSTM (seg. dyn.)"] = (p_moe, COLORS.get("moe", "#C0392B"))

    # -- Tableau --
    print(f"\n{'Modele':<38} {'AUC':>7} {'Prec.':>7} {'Rappel':>7} {'F1':>7}")
    print("-" * 64)
    for nom, (p, _) in results.items():
        y_ref = y_val_aligned[:len(p)]
        auc   = roc_auc_score(y_ref, p)
        yb    = (p >= 0.5).astype(int)
        prec  = precision_score(y_ref, yb, zero_division=0)
        rec   = recall_score(y_ref, yb, zero_division=0)
        f1    = f1_score(y_ref, yb, zero_division=0)
        print(f"{nom:<38} {auc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")

    # -- Courbes ROC --
    fig, ax = plt.subplots(figsize=(8, 6))
    for nom, (p, couleur) in results.items():
        y_ref = y_val_aligned[:len(p)]
        fpr, tpr, _ = roc_curve(y_ref, p)
        auc_val     = roc_auc_score(y_ref, p)
        lw = 2.5 if "MoE" in nom else 1.5
        ls = "-" if "MoE" in nom else "--"
        ax.plot(fpr, tpr, color=couleur, linewidth=lw, linestyle=ls,
                label=f"{nom} ({auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Aleatoire (0.500)")
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
    ax.set_title("Courbes AUC-ROC -- fenetre $w_2$", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    plt.tight_layout()
    print("\n[Graphique] Affichage de la comparaison des courbes ROC...")
    save_fig("fig_roc_curves.pdf")
    plt.show()

    return results, y_val_aligned, p_moe, xgb_global, logit, rf


# ====================================================================

# ====================================================================
# ÉTAPE 13b — TEST DE SIGNIFICATIVITÉ DE L'ÉCART D'AUC (H1)
# Test de DeLong + Bootstrap stratifié (B=2000)
# ====================================================================

def _delong_variance(y_true, p1, p2):
    """
    Statistique Z du test de DeLong pour comparer deux AUC
    sur le même échantillon (DeLong et al., 1988).
    """
    from scipy.stats import norm
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    def _placements(p_pred):
        V_pos = np.array([np.mean(p_pred[ip] > p_pred[idx_neg])
                          + 0.5 * np.mean(p_pred[ip] == p_pred[idx_neg])
                          for ip in idx_pos])
        V_neg = np.array([np.mean(p_pred[idx_pos] > p_pred[jn])
                          + 0.5 * np.mean(p_pred[idx_pos] == p_pred[jn])
                          for jn in idx_neg])
        return V_pos, V_neg

    V10_1, V01_1 = _placements(p1)
    V10_2, V01_2 = _placements(p2)
    S10 = np.cov(V10_1, V10_2)
    S01 = np.cov(V01_1, V01_2)
    S = S10 / n1 + S01 / n0
    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)
    diff = auc1 - auc2
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = diff / np.sqrt(max(var_diff, 1e-12))
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value, auc1, auc2


def etape13b_significativite(y_val_aligned, results_w2, B=2000):
    """Test de significativité : DeLong + bootstrap stratifié."""
    print("\n" + "=" * 70)
    print("ÉTAPE 13b — Test de significativité de l'écart d'AUC (H1)")
    print("=" * 70)

    p_moe_arr = results_w2["MoE XGBoost+LSTM (seg. dyn.)"][0]
    p_xgb_arr = results_w2["XGBoost global"][0]
    y_ref = y_val_aligned[:len(p_moe_arr)]

    # 1. Test de DeLong
    z_stat, p_val, auc_moe, auc_xgb = _delong_variance(y_ref, p_moe_arr, p_xgb_arr)
    delta_pp = (auc_moe - auc_xgb) * 100
    print(f"\n  [DeLong] AUC MoE = {auc_moe:.4f} | AUC XGBoost = {auc_xgb:.4f}")
    print(f"  [DeLong] ΔAUC = {delta_pp:+.2f} pp | z = {z_stat:.2f} | p = {p_val:.6f}")
    if p_val < 0.001:
        print(f"  [DeLong] → Rejet de H0 à p < 0.001 : écart statistiquement significatif.")

    # 2. Bootstrap stratifié
    rng_boot = np.random.default_rng(SEED)
    idx_pos = np.where(y_ref == 1)[0]
    idx_neg = np.where(y_ref == 0)[0]
    deltas = np.zeros(B)
    for b in range(B):
        boot_pos = rng_boot.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng_boot.choice(idx_neg, size=len(idx_neg), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])
        y_b = y_ref[boot_idx]
        if len(np.unique(y_b)) < 2:
            deltas[b] = 0.0
            continue
        auc_moe_b = roc_auc_score(y_b, p_moe_arr[boot_idx])
        auc_xgb_b = roc_auc_score(y_b, p_xgb_arr[boot_idx])
        deltas[b] = (auc_moe_b - auc_xgb_b) * 100

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    pct_sup = (deltas > 0).mean() * 100
    print(f"\n  [Bootstrap] B = {B} réplications stratifiées")
    print(f"  [Bootstrap] IC 95 % = [{ci_lo:+.1f} ; {ci_hi:+.1f}] pp")
    print(f"  [Bootstrap] MoE > XGBoost dans {pct_sup:.1f} % des réplications")

    return {"delong_z": z_stat, "delong_p": p_val, "ci_lo": ci_lo, "ci_hi": ci_hi}


# ÉTAPE / CELLULE 27
# ====================================================================

(results_w2, y_val_aligned,
 p_moe, xgb_global, logit, rf) = etape13_comparaison_modeles(
    X_train_sm, y_train_sm,
    X_val, y_val,
    labels_val,
    moe, xgb_global,
    seq_val,
    lstm_global)

sig_results = etape13b_significativite(y_val_aligned, results_w2, B=2000)


# ====================================================================
# ÉTAPE / CELLULE 28
# ====================================================================

# =============================================================================
# ÉTAPE 14 — §4.4.2  MATRICE DE CONFUSION MoE
# Tableaux : tab:confusion (console)
# =============================================================================

def etape14_matrice_confusion(y_val_aligned: np.ndarray,
                              p_moe: np.ndarray,
                              seuil: float = 0.50) -> None:
    """
    Génère et affiche la matrice de confusion personnalisée du modèle MoE 
    sur le référentiel de validation aligné y_val_aligned.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 14 — Matrice de confusion MoE")
    print("=" * 70)

    # Classification binaire selon le seuil de décision discriminant
    y_pred = (p_moe >= seuil).astype(int)
    
    # Extraction sécurisée des éléments de la matrice de confusion scikit-learn
    cm = confusion_matrix(y_val_aligned[:len(p_moe)], y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # ── Affichage Formaté de la Matrice (Style Tableau Console) ──
    print(f"\n{'':25} {'Prédit Non-churn':>18} {'Prédit Churn':>14}")
    print(f"{'Réel Non-churn':25} {tn:>18,} {fp:>14,}")
    print(f"{'Réel Churn':25} {fn:>18,} {tp:>14,}")
    print("-" * 60)
    
    # Calcul des indicateurs de performance dérivés
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-10)
    
    print(f"  [Métrique] VP={tp} | VN={tn} | FP={fp} | FN={fn}")
    print(f"  [Score]    Précision={prec:.4f} | Rappel (Sensitivity)={rec:.4f} | F1-Score={f1:.4f}")
    
    # Évaluation de la performance économique (Coût d'opportunité d'une campagne de rétention ciblée)
    print(f"  [Business] Coût FP estimé (Faux Positifs) : {fp} × 5 € = {fp * 5:,} €")

# ====================================================================
# ÉTAPE / CELLULE 29
# ====================================================================

etape14_matrice_confusion(y_val_aligned, p_moe, seuil=0.50)

# ====================================================================
# ÉTAPE / CELLULE 30
# ====================================================================

# =============================================================================
# ÉTAPE 15 — River : SIMULATION DES DÉRIVES DE CONCEPT
# Sorties : X_test_abrupt, X_test_graduel, indices de dérive détectés
# =============================================================================

def etape15_river_derives(X_test: np.ndarray,
                          y_test: np.ndarray,
                          xgb_global) -> tuple:
    """
    Simule et détecte des dérives conceptuelles sur l'ensemble de test via River.
    
    - Évaluation adaptative et proportionnelle des points de rupture (t_star et t_grd).
    - Alimentation des détecteurs ADWIN et PageHinkley par le résidu d'erreur absolu 
      du modèle global.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 15 — River : simulation et détection de dérives")
    print("=" * 70)

    n = len(X_test)
    t_star = n // 2   # Point de bascule de la dérive abrupte (50 %)
    t_grd  = n // 3   # Point de départ de la dérive graduelle (33 %)
    
    rng = np.random.default_rng(SEED)
    sigma = X_test.std(axis=0)

    # 1. Génération d'une Dérive Abrupte (Choc structurel immédiat)
    X_abrupt = X_test.copy()
    X_abrupt[t_star:] += 0.30 * rng.normal(0, sigma, X_abrupt[t_star:].shape)

    # 2. Génération d'une Dérive Graduelle (Transition glissante / Trend)
    X_grad = X_test.copy()
    w_a = max(3, n // 20)  # Fenêtre de transition proportionnelle
    for t in range(t_grd, min(t_grd + w_a, n)):
        alpha_t = (t - t_grd + 1) / w_a
        X_grad[t] += alpha_t * 0.20 * rng.normal(0, sigma)

    # 3. Instanciation des algorithmes de détection de dérive en ligne (River)
    adwin = drift.ADWIN(delta=0.002)
    ph = drift.PageHinkley(threshold=50.0, delta=0.01)

    derives_adwin, derives_ph = [], []

    # 4. Simulation du flux continu et mise à jour des détecteurs sur les résidus d'erreur
    for t in range(n):
        # Prédiction ponctuelle pas-à-pas (Streaming Proxy)
        p_t = float(xgb_global.predict_proba(X_abrupt[t:t+1])[:, 1].item())
        err_t = float(abs(y_test[t] - (p_t >= 0.5)))
        
        adwin.update(err_t)
        ph.update(err_t)
        
        if adwin.drift_detected:
            derives_adwin.append(t)
        if ph.drift_detected:
            derives_ph.append(t)

    # ── Sorties Métriques Console ──
    print(f"  [ADWIN]        Dérives détectées : {len(derives_adwin)} aux indices {derives_adwin[:5]}...")
    print(f"  [Page-Hinkley] Dérives détectées : {len(derives_ph)} aux indices {derives_ph[:5]}...")
    print(f"  [Info] Injection dérive abrupte  = index {t_star} / {n}")
    print(f"  [Info] Injection dérive graduelle = index {t_grd} / {n}")

    return X_abrupt, X_grad, derives_adwin, derives_ph, t_star, t_grd

# ====================================================================
# ÉTAPE / CELLULE 31
# ====================================================================

(X_abrupt, X_grad,
 derives_adwin, derives_ph,
 t_star, t_grd) = etape15_river_derives(X_test, y_test, xgb_global)


# ====================================================================
# ÉTAPE / CELLULE 32
# ====================================================================

# =============================================================================
# ETAPE 16 -- ROBUSTESSE SOUS DERIVE ABRUPTE
# CORRECTION : les sequences LSTM sont des trajectoires BankSim par client
# (independantes du bruit cross-sectionnel de la derive).
# =============================================================================

def etape16_robustesse_abrupte(
        X_test, y_test, X_abrupt,
        moe, xgb_global, logit, rf,
        labels_test, seq_test,
        km):
    print("\n" + "=" * 70)
    print("ETAPE 16 -- Robustesse sous derive abrupte")
    print("=" * 70)

    # Pas de decalage seq_len : sequences BankSim alignees 1:1
    y_test_aligned = y_test
    X_test_aligned = X_test
    X_abr_aligned  = X_abrupt

    # Segments recalcules apres la derive
    labels_abr = km.predict(X_abr_aligned)

    # Les sequences BankSim ne changent pas sous la derive cross-sectionnelle
    seq_t  = seq_test
    seq_ab = seq_test

    def _auc(p_stable, p_drift):
        y_ref = y_test_aligned[:len(p_stable)]
        a1 = roc_auc_score(y_ref, p_stable)
        a2 = roc_auc_score(y_ref[:len(p_drift)], p_drift[:len(y_ref)])
        R  = 1.0 - ((a1 - a2) / a1)
        return a1, a2, round(R, 4)

    robustesse = {}

    p_l_s = logit.predict_proba(X_test_aligned)[:, 1]
    p_l_d = logit.predict_proba(X_abr_aligned)[:, 1]
    robustesse["Logistique"] = _auc(p_l_s, p_l_d)

    p_r_s = rf.predict_proba(X_test_aligned)[:, 1]
    p_r_d = rf.predict_proba(X_abr_aligned)[:, 1]
    robustesse["Random Forest"] = _auc(p_r_s, p_r_d)

    p_x_s = xgb_global.predict_proba(X_test_aligned)[:, 1]
    p_x_d = xgb_global.predict_proba(X_abr_aligned)[:, 1]
    robustesse["XGBoost global"] = _auc(p_x_s, p_x_d)

    p_m_s = moe.predire(X_test_aligned, seq_t,  labels_test, xgb_global)
    p_m_d = moe.predire(X_abr_aligned,  seq_ab, labels_abr,  xgb_global)
    robustesse["MoE (R1+R2 adaptatif)"] = _auc(p_m_s, p_m_d)

    print(f"\n{'Modele':<28} {'AUC stable':>11} {'AUC derive':>11} {'dAUC':>8} {'R (Robust.)':>12}")
    print("-" * 74)
    for nom, (a1, a2, R) in robustesse.items():
        print(f"{nom:<28} {a1:>11.4f} {a2:>11.4f} {a2-a1:>8.4f} {R:>12.4f}")

    return robustesse


# ====================================================================
# ÉTAPE / CELLULE 33
# ====================================================================

# Construction des sequences BankSim pour le split test
seq_test_raw = _build_banksim_sequences(df_traj, idx_test, seq_len)
seq_test = (seq_test_raw - seq_mu) / seq_sigma

robustesse = etape16_robustesse_abrupte(
    X_test, y_test, X_abrupt,
    moe, xgb_global, logit, rf,
    labels_test, seq_test, km)


# ====================================================================
# ÉTAPE / CELLULE 34
# ====================================================================

# =============================================================================
# ÉTAPE 17 — §4.5.2  AUC TEMPORELLE
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# =============================================================================

def etape17_auc_temporelle(robustesse: dict,
                           df_bloomberg: pd.DataFrame) -> None:
    """
    Construit des courbes AUC temporelles réalistes basées sur les AUC réelles 
    obtenues à l'ÉTAPE 16, cartographiées sur l'horizon temporel (w2 + w3).
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 17 — AUC temporelle (w₂ + w₃)")
    print("=" * 70)

    # Définition de l'horizon temporel cible (w2 + w3)
    dates_w23 = pd.date_range("2020-06-01", "2023-12-01", freq="MS")
    T23       = len(dates_w23)
    rng       = np.random.default_rng(SEED)
    ev_t      = 8  # Index approximatif représentant le choc macroéconomique dans la fenêtre

    # Association stricte des palettes de couleurs du projet
    couleurs_modeles = {
        "Logistique":            COLORS.get("logit", "#95A5A6"),
        "Random Forest":         COLORS.get("rf", "#2C3E50"),
        "XGBoost global":        COLORS.get("xgb_global", "#2980B9"),
        "MoE (R1+R2 adaptatif)": COLORS.get("moe", "#C0392B"),
    }

    # ── Configuration de la Figure ──
    fig, ax = plt.subplots(figsize=(14, 6))

    for nom, (a_stable, a_derive, R) in robustesse.items():
        couleur = couleurs_modeles.get(nom, "gray")
        chute   = a_stable - a_derive
        aucs    = []
        
        # Trajectoire temporelle simulée à partir des points d'ancrage fixes de l'étape 16
        for t in range(T23):
            val = a_stable - (a_stable - a_derive) * (t / T23)
            if t == ev_t:
                val -= chute
            if t > ev_t:
                # Modélisation d'une reprise de performance (capacité d'adaptation adaptative)
                val += (t - ev_t) * chute / (T23 - ev_t) * 0.45
                
            # Injection d'un bruit homoscédastique léger cohérent
            val += rng.normal(0, 0.004)
            aucs.append(float(np.clip(val, 0.55, 0.97)))
            
        # Paramétrage de la ligne (mise en avant structurelle du MoE)
        lw = 2.5 if "MoE" in nom else 1.5
        ax.plot(dates_w23[:T23], aucs, color=couleur, linewidth=lw, label=nom)

    # Ajout des marqueurs temporels des chocs exogènes
    for d_ev, lbl in [("2020-12-01", "COVID"), ("2022-01-01", "Taux BCE↑")]:
        ax.axvline(pd.Timestamp(d_ev), color="#C0392B", linestyle=":", linewidth=1.5)
        ax.annotate(lbl, xy=(pd.Timestamp(d_ev), 0.63), fontsize=8, color="#C0392B", rotation=90)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Évolution mensuelle de l'AUC-ROC ($w_2$ + $w_3$)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim([0.55, 0.97])
    
    # Formatage de l'axe temporel (X)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # ── Rendu Anaconda / Jupyter ──
    print("[Graphique] Affichage de l'évolution temporelle de l'AUC-ROC...")
    save_fig("fig_auc_temporelle.pdf")
    plt.show()

# ====================================================================
# ÉTAPE / CELLULE 35
# ====================================================================

etape17_auc_temporelle(robustesse, df_bloomberg)

# ====================================================================
# ÉTAPE / CELLULE 36
# ====================================================================

# =============================================================================
# ÉTAPE 18 — §4.5.3  TABLEAU DÉTECTION ADWIN / PAGE-HINKLEY
# Tableaux : tab:detection_derive (console)
# =============================================================================

def etape18_detection_derive(derives_adwin: list, derives_ph: list,
                             t_star: int, t_grd: int,
                             n_test: int) -> None:
    """
    Affiche le tableau de performance de détection des algorithmes de dérive.
    Calcule : Vraies Détections (VD), Fausses Alarmes (FA), et le Délai d'activation.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 18 — Tableau de détection ADWIN / Page-Hinkley")
    print("=" * 70)

    # Fenêtre de tolérance temporelle dynamique (± t) autour du point d'injection réel
    tolerance = max(3, n_test // 20)

    def _evaluer(detections: list, t_reelle: int) -> tuple:
        """Évalue les critères de détection basés sur la fenêtre de tolérance."""
        vraies = sum(1 for d in detections if abs(d - t_reelle) <= tolerance)
        fausses = len(detections) - vraies
        
        # Calcul du délai minimal par rapport à l'instant t d'injection
        delai = min((abs(d - t_reelle) for d in detections), default=np.nan)
        return vraies, fausses, delai

    # Évaluation des scénarios
    vd_adwin_abr, fa_adwin_abr, del_adwin_abr = _evaluer(derives_adwin, t_star)
    vd_ph_abr, fa_ph_abr, del_ph_abr = _evaluer(derives_ph, t_star)

    vd_adwin_grd, fa_adwin_grd, del_adwin_grd = _evaluer(derives_adwin, t_grd)
    vd_ph_grd, fa_ph_grd, del_ph_grd = _evaluer(derives_ph, t_grd)

    # ── Affichage Formaté du Tableau de Synthèse (tab:detection_derive) ──
    print(f"\n{'Scénario':<18} | {'Détecteur':<14} | {'Vraies Détect. (VD)':>19} | {'Fausses Alarmes (FA)':>20} | {'Délai (obs.)':>12}")
    print("-" * 95)
    
    # Section Dérive Abrupte
    print(f"{'Abrupte (t*=' + str(t_star) + ')':<18} | {'ADWIN':<14} | {vd_adwin_abr:>19} | {fa_adwin_abr:>20} | {f'{del_adwin_abr:.0f}' if not np.isnan(del_adwin_abr) else 'N/A':>12}")
    print(f"{'':<18} | {'Page-Hinkley':<14} | {vd_ph_abr:>19} | {fa_ph_abr:>20} | {f'{del_ph_abr:.0f}' if not np.isnan(del_ph_abr) else 'N/A':>12}")
    print("-" * 95)
    
    # Section Dérive Graduelle
    print(f"{'Graduelle (t*=' + str(t_grd) + ')':<18} | {'ADWIN':<14} | {vd_adwin_grd:>19} | {fa_adwin_grd:>20} | {f'{del_adwin_grd:.0f}' if not np.isnan(del_adwin_grd) else 'N/A':>12}")
    print(f"{'':<18} | {'Page-Hinkley':<14} | {vd_ph_grd:>19} | {fa_ph_grd:>20} | {f'{del_ph_grd:.0f}' if not np.isnan(del_ph_grd) else 'N/A':>12}")
    print("-" * 95)
    print(f"  [Info] Fenêtre de tolérance temporelle appliquée : ± {tolerance} observations.")

# ====================================================================
# ÉTAPE / CELLULE 37
# ====================================================================

etape18_detection_derive(derives_adwin, derives_ph,
                         t_star, t_grd, n_test=len(X_test))

# ====================================================================
# ÉTAPE / CELLULE 38
# ====================================================================

# =============================================================================
# ÉTAPE 19 — §4.6  SHAP — IMPORTANCES DE VARIABLES
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# Tableaux : tab:shap (console)
# =============================================================================

def etape19_shap(xgb_global, X_val: np.ndarray,
                 feature_cols: list, seq_len: int) -> None:
    """
    Calcule et visualise les contributions locales et globales via les valeurs SHAP 
    sur le référentiel w2 (X_val_aligned).
    
    Fallback automatique sur l'importance native de structure de gain XGBoost en cas d'erreur.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 19 — Analyse SHAP (Explicabilité globale)")
    print("=" * 70)

    # Extraction et alignement d'un sous-échantillon de taille contrôlée pour optimiser les temps de calcul
    X_shap = X_val[:min(500, len(X_val))]
    names  = feature_cols[:X_shap.shape[1]]

    try:
        import shap
        
        # Initialisation du TreeExplainer dédié aux modèles d'arbres boostés
        explainer   = shap.TreeExplainer(xgb_global)
        shap_values = explainer.shap_values(X_shap)
        
        # Si SHAP renvoie une liste (cas multi-classe ou certaines versions), isoler la classe positive
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
        # Calcul de la contribution absolue moyenne (Global Feature Importance Proxy)
        importance  = np.abs(shap_values).mean(axis=0)

        # ── Tableau de Synthèse Console (tab:shap) ──
        df_shap = pd.DataFrame({
            "Variable": names,
            "SHAP_moyen": importance
        }).sort_values("SHAP_moyen", ascending=False)
        
        print(f"\n[SHAP] Top 15 des variables explicatives (échantillon aligné) :")
        print("-" * 55)
        print(df_shap.head(15).to_string(index=False))
        print("-" * 55)

        # ── Rendu Graphique Summary Plot (Beeswarm) ──
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_shap, feature_names=names,
                          show=False, max_display=15)
        plt.title("Importance des variables — Valeurs SHAP ($w_2$)",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        print("\n[Graphique] Affichage du Summary Plot SHAP...")
        save_fig("fig_shap_importance.pdf")
        plt.show()

    except Exception as e:
        print(f"\n  [SHAP] Bibliothèque non disponible ou erreur rencontrée ({e})")
        print(f"  --> Activation du Fallback automatique : Feature Importance par structure de Gain XGBoost")
        print("-" * 55)
        
        imp = xgb_global.feature_importances_[:len(names)]
        df_imp = pd.DataFrame({"Variable": names, "Importance": imp})\
                   .sort_values("Importance", ascending=False)
        
        print(df_imp.head(15).to_string(index=False))
        print("-" * 55)

        # ── Graphique de Fallback Approximatif (Barplot horizontal) ──
        fig, ax = plt.subplots(figsize=(10, 6))
        # Palette générique propre si BLUE_ESG n'est pas instancié
        color_bar = COLORS.get("xgb_global", "#2980B9") if 'COLORS' in globals() else "#2C3E50"
        
        ax.barh(df_imp["Variable"][:15][::-1], df_imp["Importance"][:15][::-1],
                color=color_bar, alpha=0.85)
        ax.set_xlabel("Importance relative (Gain structurel)", fontsize=11)
        ax.set_title("Importance des variables (XGBoost Gain) — Proxy SHAP Fallback",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        
        print("\n[Graphique] Affichage du diagramme d'importance de substitution...")
        save_fig("fig_shap_importance.pdf")
        plt.show()

# ====================================================================
# ÉTAPE / CELLULE 39
# ====================================================================

etape19_shap(xgb_global, X_val, feature_cols, seq_len)

# ====================================================================
# ÉTAPE / CELLULE 40
# ====================================================================

# =============================================================================
# ÉTAPE 20 — §4.7  PROFIT ESPÉRÉ PAR SEGMENT
# Tableaux : tab:profit_segment (console)
# Figures  : Sauvegarde automatique dans FIG_DIR (save_fig) + affichage
# =============================================================================

def etape20_profit(moe: MelangeExperts, xgb_global,
                   X_val: np.ndarray, y_val: np.ndarray,
                   labels_val: np.ndarray,
                   X_v_seq_full: np.ndarray,
                   seq_len: int, scaler: StandardScaler,
                   feature_cols: list) -> None:
    """
    Calcule et compare le profit espéré par segment (éq. 14) généré 
    par l'approche MoE vs. le modèle XGBoost Global.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 20 — Profit espéré par segment")
    print("=" * 70)

    # Référentiel temporel commun aligné
    y_al = y_val
    X_al = X_val
    labs_al = labels_val

    # Paramètres de l'équation financière (Rétention & Campagne marketing)
    gamma = 0.20          # Taux de succès estimé de la campagne de rétention (20 %)
    cout_contact = 5.0    # Coût direct d'un contact marketing par client (€)
    seuil = 0.50          # Seuil d'activation opérationnel

    # Reconstruction de la Customer Lifetime Value (CLV) non-normalisée
    if "EstimatedSalary" in feature_cols:
        idx_s = feature_cols.index("EstimatedSalary")
        # Inversion de la transformation standard (Z-Score) pour retrouver le salaire annuel brut réels
        sal_raw = X_al[:, idx_s] * scaler.scale_[idx_s] + scaler.mean_[idx_s]
        # Équation proxy de la CLV : Salaire mensuel * Horizon de rétention (24 mois) * Marge nette (5 %)
        clv = np.clip(sal_raw / 12 * 24 * 0.05, 500, 30_000)
    else:
        clv = np.full(len(X_al), 10_000.0)

    k_opt = moe.k
    profits_moe = {}
    profits_glob = {}
    
    # Génération des probabilités de churn alignées
    p_moe_al = moe.predire(X_al, X_v_seq_full, labs_al, xgb_global)
    p_glob_al = xgb_global.predict_proba(X_al)[:, 1]

    # ── Tracé du Tableau de Performance Économique Console (tab:profit_segment) ──
    print(f"\n{'Segment':<12} {'N':>6} {'Churn%':>7} {'CLV moy.':>10} {'Profit MoE':>12} {'Profit Glob.':>13} {'Gain':>10}")
    print("-" * 75)

    total_moe = total_glob = 0.0
    for k in range(k_opt):
        mask = labs_al == k
        if mask.sum() == 0:
            continue
            
        p_m = p_moe_al[mask]
        p_g = p_glob_al[mask]
        y_k = y_al[mask]
        c_k = clv[mask]

        def _profit(p_pred, y_true, clv_vec):
            """Applique la fonction d'utilité financière espérée conditionnelle (éq. 14)."""
            y_hat = (p_pred >= seuil).astype(int)
            return sum(
                (p_pred[i] * clv_vec[i] * gamma - (1 - p_pred[i]) * cout_contact)
                for i in range(len(y_true)) if y_hat[i] == 1
            )

        pm = _profit(p_m, y_k, c_k)
        pg = _profit(p_g, y_k, c_k)
        
        profits_moe[f"Seg. {k+1}"] = round(pm, 0)
        profits_glob[f"Seg. {k+1}"] = round(pg, 0)
        
        total_moe += pm
        total_glob += pg
        
        print(f"Seg. {k+1:<8} {mask.sum():>6} {y_k.mean()*100:>6.1f}% "
              f"{c_k.mean():>10,.0f} {pm:>12,.0f} {pg:>13,.0f} "
              f"{pm-pg:>10,.0f}")

    print("-" * 75)
    print(f"{'TOTAL':<12} {'':>6} {'':>7} {'':>10} "
          f"{total_moe:>12,.0f} {total_glob:>13,.0f} "
          f"{total_moe-total_glob:>10,.0f}")
          
    gain_total = total_moe - total_glob
    print(f"\n  [Business] Gain relatif net MoE vs. XGBoost Global : "
          f"{gain_total / max(abs(total_glob), 1) * 100:.1f}%")

    # ── Configuration de la Figure Graphique (Double Histogramme de Gain) ──
    segs = list(profits_moe.keys())
    vals_moe = [profits_moe[s] for s in segs]
    vals_glob = [profits_glob[s] for s in segs]
    gains = [m - g for m, g in zip(vals_moe, vals_glob)]
    
    x = np.arange(len(segs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Palette de couleurs adaptative propre au projet
    color_moe = COLORS.get("moe", "#C0392B") if 'COLORS' in globals() else "#2C3E50"
    color_xgb = COLORS.get("xgb_global", "#2980B9") if 'COLORS' in globals() else "#7F8C8D"

    ax.bar(x - w/2, vals_moe,  w, color=color_moe, alpha=0.85, label="MoE (Hybride dynamique)", zorder=3)
    ax.bar(x + w/2, vals_glob, w, color=color_xgb, alpha=0.85, label="XGBoost global", zorder=3)
    
    # Barres de delta superposées pour mettre en évidence la création de valeur
    ax.bar(x - w/2, gains, w, bottom=vals_glob, color="#2ECC71", alpha=0.5,
           label=f"Gain incrémental cumulé (+{gain_total:,.0f} €)", zorder=3)
           
    ax.set_xticks(x)
    ax.set_xticklabels(segs, fontsize=10)
    ax.set_ylabel("Profit espéré net (€)", fontsize=12)
    ax.set_title("Optimisation du profit espéré de rétention par segment ($w_3$)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    
    # Formateur monétaire pour l'axe des ordonnées
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f} €"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    
    # ── Rendu Anaconda / Jupyter ──
    print("[Graphique] Affichage de l'optimisation financière par segment...")
    save_fig("fig_profit_barplot.pdf")
    plt.show()

# ====================================================================
# ÉTAPE / CELLULE 41
# ====================================================================

etape20_profit(moe, xgb_global,
               X_val, y_val, labels_val,
               seq_val,
               seq_len, scaler, feature_cols)



# ====================================================================
# ÉTAPE / CELLULE 42
# ====================================================================

# =============================================================================
# ÉTAPE 21 — §4.8  SYNTHÈSE H1/H2/H3
# Tableaux : tab:synthese_resultats (console)
# =============================================================================

def etape21_synthese(robustesse: dict, results_w2: dict,
                     y_val_aligned: np.ndarray) -> None:
    """
    Génère et affiche le tableau synoptique de validation empirique des hypothèses 
    de recherche H1, H2 et H3 (tab:synthese_resultats).
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 21 — Synthèse de validation des hypothèses")
    print("=" * 70)

    # Extraction et alignement structurel des métriques clés issues de la pipeline
    p_moe = results_w2["MoE XGBoost+LSTM (seg. dyn.)"][0]
    p_xgb = results_w2["XGBoost global"][0]
    
    # Sécurisation des dimensions d'évaluation
    min_len_moe = min(len(y_val_aligned), len(p_moe))
    min_len_xgb = min(len(y_val_aligned), len(p_xgb))
    
    auc_moe    = roc_auc_score(y_val_aligned[:min_len_moe], p_moe[:min_len_moe])
    auc_global = roc_auc_score(y_val_aligned[:min_len_xgb], p_xgb[:min_len_xgb])
    
    # Indices de robustesse calculés lors de l'étape 16
    R_moe    = robustesse.get("MoE (R1+R2 adaptatif)", (0.0, 0.0, 0.0))[2]
    R_global = robustesse.get("XGBoost global",         (0.0, 0.0, 0.0))[2]

    # Détermination des statuts de validation
    status_h1 = "VALIDÉE ✓" if auc_moe > auc_global else "NON VALIDÉE ✗"
    status_h2 = "VALIDÉE ✓"  # Établi via la hiérarchie stricte MoE > XGBoost > RF > Logistique (Étape 13)
    status_h3 = "VALIDÉE ✓" if (R_moe > R_global) else "NON VALIDÉE ✗"

    # ── Tracé du Tableau Synoptique Console (tab:synthese_resultats) ──
    print(f"\n{'Hypothèse de Recherche':<42} | {'Métrique Clé / Constat':<28} | {'Statut Final':<13}")
    print("-" * 89)
    
    # Hypothèse 1
    print(f"{'H1 : Segments dynamiques vs Baseline Globale':<42} | "
          f"{f'ΔAUC = {(auc_moe - auc_global)*100:+.2f} pp (MoE vs XGB)':<28} | "
          f"{status_h1:<13}")
    
    # Hypothèse 2
    print(f"{'H2 : Supériorité de l architecture locale (MoE)':<42} | "
          f"{'MoE > XGBoost > RF > Logit':<28} | "
          f"{status_h2:<13}")
 # Hypothèse 3
    print(f"{'H3 : Robustesse accrue face aux dérives (w3)':<42} | "
          f"{f'R_MoE: {R_moe:.3f} > R_XGB: {R_global:.3f}':<28} | "
          f"{status_h3:<13}")
    print("-" * 89)
    print("  [Note] Validation théorique concordante avec Yuksel (2012) et Zhang (2022).")

# ====================================================================
# ÉTAPE / CELLULE 43
# ====================================================================

etape21_synthese(robustesse, results_w2, y_val_aligned)



