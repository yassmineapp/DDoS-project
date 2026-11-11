# -*- coding: utf-8 -*-
"""
Sentinelle DDoS - application Streamlit de détection d'attaques DDoS.

Pipeline : chargement CSV -> prétraitement -> découpage 70/15/15 ->
entraînement (Régression logistique, SVM, Random Forest, XGBoost) ->
métriques (Accuracy, Précision, Recall, F1) -> prédiction.
Aucun résultat n'est écrit en dur : tout est calculé à partir du CSV chargé.
"""
import base64
import hashlib
import hmac
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

BASE = Path(__file__).parent

# ----------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Sentinelle DDoS", page_icon=Image.open(BASE / "logo.png"),
                   layout="wide")

# ----------------------------------------------------------------------------
# 2. IDENTIFIANTS (mot de passe stocké sous forme de hash PBKDF2, jamais en clair)
#    Pour le changer : voir le fichier LISEZMOI.txt
# ----------------------------------------------------------------------------
IDENTIFIANT = "yassmine"
SEL = "ef72b25fd958c9b9089dbd79bf50f19e"
HASH_MDP = "a95e9e683b303bc5514dc2f37d6e750cd5d9706cddd03ec3333e1b10b1757773"


def verifier_identifiants(utilisateur: str, mot_de_passe: str) -> bool:
    """Compare l'identifiant et le hash du mot de passe en temps constant."""
    h = hashlib.pbkdf2_hmac("sha256", mot_de_passe.encode(), bytes.fromhex(SEL), 200_000).hex()
    ok_user = hmac.compare_digest(utilisateur.strip().lower().encode(), IDENTIFIANT.encode())
    ok_pwd = hmac.compare_digest(h.encode(), HASH_MDP.encode())
    return ok_user and ok_pwd


# ----------------------------------------------------------------------------
# 3. STYLE (polices + cartes + verdict)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@500&family=Public+Sans:wght@400;500;600&display=swap');
html, body, .stApp { font-family: 'Public Sans', 'Segoe UI', sans-serif; }
/* Fond : halo bleu + quadrillage discret, comme un écran de contrôle */
.stApp {
  background:
    radial-gradient(ellipse 70% 45% at 15% 0%, rgba(45,212,191,.10), transparent 70%),
    radial-gradient(ellipse 60% 40% at 100% 10%, rgba(255,107,61,.07), transparent 70%),
    linear-gradient(rgba(150,180,220,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(150,180,220,.035) 1px, transparent 1px),
    #070B14;
  background-size: auto, auto, 44px 44px, 44px 44px, auto;
}
h1, h2, h3 { font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important; letter-spacing: -0.01em; }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] {
  border-color: rgba(45,212,191,.20) !important; background: rgba(14,22,38,.72);
  box-shadow: 0 0 28px rgba(45,212,191,.04); border-radius: 10px; }
.stTabs [data-baseweb="tab"] { font-family:'Space Grotesk',sans-serif; font-weight:700; padding:10px 18px; }
.kicker { font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.22em;
          color:#2DD4BF; text-transform:uppercase; }
.entete { display:flex; align-items:center; gap:18px; margin-bottom:10px; }
.entete img { width:68px; height:68px; filter: drop-shadow(0 0 14px rgba(45,212,191,.55)); }
.entete h1 { margin:2px 0 0 0; font-size:2.1rem; }
.entete p { margin:0; color:#8FA3BD; }
.verdict { border-radius:12px; padding:28px 32px; color:#fff; margin:12px 0; }
.verdict.normal { background:linear-gradient(135deg,#0F766E,#0B3B3A); box-shadow:0 0 36px rgba(45,212,191,.25); }
.verdict.attaque { background:linear-gradient(135deg,#C2410C,#6B230C); box-shadow:0 0 36px rgba(255,107,61,.30); }
.verdict .titre { font-family:'Space Grotesk',sans-serif; font-size:2.4rem; font-weight:700; line-height:1.1; }
.verdict .sous { opacity:.9; margin-top:6px; }
.barre { background:rgba(255,255,255,.25); border-radius:99px; height:12px; margin-top:16px; }
.barre > div { background:#fff; height:12px; border-radius:99px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def logo_base64() -> str:
    """Logo SVG encodé pour l'insérer dans du HTML."""
    return base64.b64encode((BASE / "logo.svg").read_bytes()).decode()


def entete(titre: str, sous_titre: str):
    st.markdown(
        f'<div class="entete"><img src="data:image/svg+xml;base64,{logo_base64()}">'
        f'<div><div class="kicker">Observatoire · Flux réseau</div>'
        f'<h1>{titre}</h1><p>{sous_titre}</p></div></div>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 4. CONNEXION
# ----------------------------------------------------------------------------
def page_connexion():
    """Affiche le formulaire de connexion (5 essais maximum par session)."""
    st.session_state.setdefault("essais", 0)
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.write("")
        entete("Sentinelle DDoS", "Détection d'attaques réseau par machine learning")
        st.write("")
        if st.session_state.essais >= 5:
            st.error("Trop de tentatives. Rechargez la page pour réessayer.")
            return
        with st.container(border=True):
            with st.form("connexion", border=False):
                utilisateur = st.text_input("Identifiant")
                mot_de_passe = st.text_input("Mot de passe", type="password")
                envoyer = st.form_submit_button("Se connecter", type="primary",
                                                width="stretch")
        if envoyer:
            if verifier_identifiants(utilisateur, mot_de_passe):
                st.session_state.connecte = True
                st.rerun()
            else:
                st.session_state.essais += 1
                st.error(f"Identifiant ou mot de passe incorrect "
                         f"({st.session_state.essais}/5).")


# ----------------------------------------------------------------------------
# 5. DONNÉES
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_csv(contenu):
    """Charge dataset.csv (contenu=None) ou le fichier envoyé par l'utilisateur."""
    if contenu is None:
        return pd.read_csv(BASE / "dataset.csv")
    return pd.read_csv(io.BytesIO(contenu))


def detecter_cible(df: pd.DataFrame) -> str:
    """Cherche la colonne d'étiquette (0 = normal, 1 = attaque)."""
    for nom in ["label", "is_anomaly", "is_attack", "attack", "target", "class"]:
        if nom in df.columns:
            return nom
    return df.columns[-1]


def normaliser_cible(serie: pd.Series):
    """Retourne la cible en 0/1, ou None si impossible."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(int) if set(serie.unique()) <= {0, 1} else None
    texte = serie.astype(str).str.lower().str.strip()
    correspondance = {"normal": 0, "benign": 0, "0": 0,
                      "ddos": 1, "attack": 1, "attaque": 1, "anomaly": 1, "1": 1}
    resultat = texte.map(correspondance)
    return resultat.astype("Int64").astype(int) if resultat.notna().all() else None


# Colonnes d'identification exclues (comme dans le TP : elles favorisent les raccourcis)
COLONNES_ID = ["timestamp", "src_ip", "dst_ip", "src_port"]
# Colonnes numériques à traiter comme des catégories si elles existent
COLONNES_CATEGORIELLES = ["protocol", "dst_port"]


def preparer(df: pd.DataFrame, cible: str, supprimer_doublons: bool, mode_nan: str):
    """Nettoyage + construction de X (brut), y. Retourne aussi un journal."""
    journal = []
    df = df.dropna(subset=[cible]).copy()
    if supprimer_doublons:
        avant = len(df)
        df = df.drop_duplicates().copy()
        journal.append(f"Doublons supprimés : {avant - len(df)}")
    nb_nan = int(df.isna().sum().sum())
    if mode_nan == "Supprimer les lignes":
        df = df.dropna()
        journal.append(f"Valeurs manquantes : {nb_nan} (lignes supprimées)")
    else:
        df = df.fillna(0)
        journal.append(f"Valeurs manquantes : {nb_nan} (remplacées par 0)")
    y = normaliser_cible(df[cible])
    X_brut = df.drop(columns=[cible] + [c for c in COLONNES_ID if c in df.columns])
    return X_brut, y, journal


def encoder(X_brut: pd.DataFrame) -> pd.DataFrame:
    """One-hot des colonnes catégorielles (texte + protocol/dst_port)."""
    cats = [c for c in X_brut.columns
            if X_brut[c].dtype == object or c in COLONNES_CATEGORIELLES]
    return pd.get_dummies(X_brut, columns=cats, dtype=int)


# ----------------------------------------------------------------------------
# 6. MODÈLES ET ENTRAÎNEMENT
# ----------------------------------------------------------------------------
# True = le modèle a besoin de variables standardisées
MODELES = {
    "Régression logistique": (lambda: LogisticRegression(max_iter=1000, random_state=42), True),
    "SVM": (lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=42), True),
    "Random Forest": (lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2), False),
    "XGBoost": (lambda: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                      eval_metric="logloss", random_state=42, n_jobs=2), False),
}


def entrainer(X_brut, y, noms):
    """Découpe 70/15/15, standardise (fit sur train), entraîne et évalue sur validation."""
    X = encoder(X_brut)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                random_state=42, stratify=y_tmp)
    scaler = StandardScaler().fit(X_tr)  # fit uniquement sur train (pas de fuite de données)
    paquet = {"colonnes": list(X.columns), "scaler": scaler, "modeles": {}, "lignes": []}
    for nom in noms:
        fabrique, besoin_echelle = MODELES[nom]
        entree_tr = scaler.transform(X_tr) if besoin_echelle else X_tr
        entree_val = scaler.transform(X_val) if besoin_echelle else X_val
        modele = fabrique().fit(entree_tr, y_tr)
        pred = modele.predict(entree_val)
        paquet["modeles"][nom] = modele
        paquet["lignes"].append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_val, pred),
            "Précision": precision_score(y_val, pred, zero_division=0),
            "Recall": recall_score(y_val, pred, zero_division=0),
            "F1-score": f1_score(y_val, pred, zero_division=0)})
    paquet["tableau"] = pd.DataFrame(paquet["lignes"]).sort_values("F1-score", ascending=False)
    paquet["meilleur"] = paquet["tableau"].iloc[0]["Modèle"]
    paquet["X_test"], paquet["y_test"] = X_te, y_te
    paquet["tailles"] = (len(X_tr), len(X_val), len(X_te))
    paquet["positifs_test"] = int(y_te.sum())
    return paquet


def entree_modele(paquet, nom, X):
    """Applique la standardisation seulement si le modèle l'exige."""
    return paquet["scaler"].transform(X) if MODELES[nom][1] else X


def specification_formulaire(X_brut: pd.DataFrame) -> dict:
    """Décrit chaque variable brute pour construire le formulaire de prédiction."""
    spec = {}
    for c in X_brut.columns:
        if X_brut[c].dtype == object or c in COLONNES_CATEGORIELLES:
            spec[c] = {"type": "cat", "valeurs": sorted(X_brut[c].unique().tolist())}
        else:
            est_entier = pd.api.types.is_integer_dtype(X_brut[c])
            spec[c] = {"type": "int" if est_entier else "float",
                       "min": X_brut[c].min(), "max": X_brut[c].max(),
                       "mediane": X_brut[c].median()}
    return spec


def predire(paquet, nom, X_brut_nouveau: pd.DataFrame):
    """Encode comme à l'entraînement puis retourne (classes, probabilités d'attaque)."""
    X = encoder(X_brut_nouveau).reindex(columns=paquet["colonnes"], fill_value=0)
    modele = paquet["modeles"][nom]
    entree = entree_modele(paquet, nom, X)
    return modele.predict(entree), modele.predict_proba(entree)[:, 1]


# ----------------------------------------------------------------------------
# 7. GRAPHIQUES
# ----------------------------------------------------------------------------
# Couleurs communes : turquoise lumineux = normal, orange = attaque
ECHELLE = alt.Scale(domain=["Normal", "Attaque"], range=["#2DD4BF", "#FF6B3D"])


def graphique_confusion(y_vrai, y_pred):
    """Matrice de confusion sous forme de carte de chaleur."""
    cm = confusion_matrix(y_vrai, y_pred, labels=[0, 1])
    noms = ["Normal", "Attaque"]
    d = pd.DataFrame([{"Réel": noms[i], "Prédit": noms[j], "Flux": int(cm[i, j])}
                      for i in range(2) for j in range(2)])
    base = alt.Chart(d).encode(x=alt.X("Prédit:N", sort=noms, axis=alt.Axis(labelAngle=0)),
                               y=alt.Y("Réel:N", sort=noms))
    carres = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Flux:Q", scale=alt.Scale(range=["#0E1626", "#2DD4BF"]), legend=None))
    texte = base.mark_text(fontSize=22, fontWeight="bold").encode(
        text="Flux:Q",
        color=alt.condition(alt.datum.Flux > cm.max() / 2, alt.value("#06201D"), alt.value("#E6EDF7")))
    return (carres + texte).properties(height=240)


def jauge(titre, valeur, couleur):
    """Jauge circulaire (« boussole ») : la valeur (0 à 1) est lue en pourcentage au centre."""
    d = pd.DataFrame({"part": ["valeur", "reste"], "v": [valeur, 1 - valeur], "ordre": [0, 1]})
    arc = alt.Chart(d).mark_arc(innerRadius=46, outerRadius=62, cornerRadius=3).encode(
        theta=alt.Theta("v:Q", stack=True), order=alt.Order("ordre:Q"),
        color=alt.Color("part:N", legend=None,
                        scale=alt.Scale(domain=["valeur", "reste"], range=[couleur, "#1B2842"])))
    centre = alt.Chart(pd.DataFrame({"t": [f"{valeur * 100:.0f} %"]})).mark_text(
        fontSize=22, fontWeight="bold", color="#E6EDF7").encode(text="t:N")
    return (arc + centre).properties(
        width=150, height=150,
        title=alt.TitleParams(titre, anchor="middle", color="#9FB3C8", fontSize=13))

def regle_alerte(df, y, variable, seuil, tranches=4):
    """Découpe la variable en tranches (quantiles) et signale celles dont le taux d'attaque dépasse le seuil."""
    d = pd.DataFrame({"v": df.loc[y.index, variable].to_numpy(), "y": y.to_numpy()})
    try:
        d["tranche"] = pd.qcut(d["v"], tranches, duplicates="drop")
    except ValueError:
        return None
    t = d.groupby("tranche", observed=True).agg(Flux=("y", "size"), Attaques=("y", "sum")).reset_index()
    t["Taux d'attaque (%)"] = t["Attaques"] / t["Flux"] * 100
    t["Intervalle"] = t["tranche"].astype(str)
    t["Statut"] = np.where(t["Taux d'attaque (%)"] > seuil, "ALERTE", "OK")
    return t[["Intervalle", "Flux", "Attaques", "Taux d'attaque (%)", "Statut"]]


def avis_donnees(df, y, cible, variable, table, seuil):
    """Rédige automatiquement un résumé : tout est calculé sur les données, rien n'est inventé."""
    phrases = [f"Le jeu contient **{len(y)} flux**, dont **{int(y.sum())} attaques** "
               f"({y.mean() * 100:.1f} %)."]
    numeriques = [c for c in df.select_dtypes("number").columns if c != cible]
    if numeriques:
        d = df.loc[y.index, numeriques]
        m0, m1 = d[y == 0].mean(), d[y == 1].mean()
        ecart = ((m1 - m0).abs() / d.std().replace(0, np.nan)).dropna()
        if len(ecart):
            top = ecart.idxmax()
            phrases.append(f"La variable qui sépare le mieux les deux classes est **{top}** : "
                           f"moyenne {m0[top]:.2f} pour le trafic normal contre {m1[top]:.2f} pour les attaques.")
    if table is not None:
        alertes = table[table["Statut"] == "ALERTE"]
        if len(alertes):
            col = "Taux d'attaque (%)"
            liste = ", ".join(f"{r['Intervalle']} ({r[col]:.1f} %)" for _, r in alertes.iterrows())
            phrases.append(f"Règle d'alerte ({seuil} %) : sur **{variable}**, "
                           f"{len(alertes)} tranche(s) dépassent le seuil : {liste}.")
        else:
            phrases.append(f"Règle d'alerte ({seuil} %) : aucune tranche de **{variable}** ne dépasse le seuil.")
    manquantes, doublons = int(df.isna().sum().sum()), int(df.duplicated().sum())
    phrases.append(f"Qualité des données : {manquantes} valeur(s) manquante(s) et {doublons} doublon(s).")
    return phrases


def avis_modele(paquet, nom, y_te, pred):
    """Résumé automatique du test final, calculé à partir de la matrice de confusion."""
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    phrases = [f"Sur le test ({len(y_te)} flux, dont {tp + fn} attaques), **{nom}** détecte "
               f"**{tp}** attaque(s), en manque **{fn}** (faux négatifs) et déclenche "
               f"**{fp}** fausse(s) alerte(s) (faux positifs)."]
    f1 = paquet["tableau"]["F1-score"]
    if f1.max() == f1.min():
        phrases.append("Tous les modèles ont le même F1 en validation : le jeu de données est "
                       "peut-être trop simple ou trop petit pour les départager.")
    else:
        phrases.append(f"En validation, l'écart de F1 entre le meilleur et le dernier modèle est de "
                       f"{f1.max() - f1.min():.3f}.")
    if paquet["positifs_test"] < 50:
        phrases.append(f"Attention : seulement {paquet['positifs_test']} attaques dans le test, "
                       "les métriques sont peu stables.")
    return phrases



# ----------------------------------------------------------------------------
# 8. ONGLETS
# ----------------------------------------------------------------------------
def onglet_donnees(df, cible):
    y = normaliser_cible(df[cible].dropna())
    if y is None:
        st.error(f"La colonne « {cible} » doit contenir 0/1 (ou normal/ddos).")
        return
    numeriques = [c for c in df.select_dtypes("number").columns if c != cible]
    # Mini-courbe : taux d'attaque par tranche de lignes consécutives (calculé sur les données)
    taux = [float(t.mean() * 100) for t in np.array_split(y.to_numpy(), 12)] if len(y) >= 12 else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flux analysés", f"{len(df):,}".replace(",", " "), border=True)
    c2.metric("Taux d'attaque", f"{y.mean() * 100:.1f} %", border=True,
              chart_data=taux, chart_type="area",
              help="Mini-courbe : taux d'attaque par tranche de lignes consécutives.")
    c3.metric("Valeurs manquantes", int(df.isna().sum().sum()), border=True)
    c4.metric("Doublons", int(df.duplicated().sum()), border=True)

    g1, g2 = st.columns([2, 1])
    # --- Champ de flux : nuage de points interactif (filtrage croisé par sélection) ---
    with g1.container(border=True):
        st.markdown("**Champ de flux** · dessinez un rectangle pour isoler une zone")
        if len(numeriques) < 2:
            st.info("Il faut au moins deux variables numériques.")
        else:
            a, b = st.columns(2)
            vx = a.selectbox("Axe X", numeriques, index=0)
            vy = b.selectbox("Axe Y", numeriques, index=1)
            d = pd.DataFrame({"X": df.loc[y.index, vx].to_numpy(),
                              "Y": df.loc[y.index, vy].to_numpy(),
                              "Classe": y.map({0: "Normal", 1: "Attaque"}).to_numpy()})
            d_aff = d.sample(min(len(d), 5000), random_state=0)  # limite d'affichage d'Altair
            zone = alt.selection_interval(name="zone")
            nuage = alt.Chart(d_aff).mark_circle(size=55).encode(
                x=alt.X("X:Q", title=vx, scale=alt.Scale(zero=False)),
                y=alt.Y("Y:Q", title=vy, scale=alt.Scale(zero=False)),
                color=alt.Color("Classe:N", scale=ECHELLE, legend=alt.Legend(orient="top", title=None)),
                opacity=alt.condition(zone, alt.value(0.9), alt.value(0.18)),
                tooltip=["Classe", alt.Tooltip("X:Q", title=vx), alt.Tooltip("Y:Q", title=vy)]
            ).add_params(zone).properties(height=330)
            evenement = st.altair_chart(nuage, on_select="rerun", key=f"champ_{vx}_{vy}",
                                        width="stretch")
            if len(d) > len(d_aff):
                st.caption(f"Affichage limité à {len(d_aff)} points ; les compteurs ci-dessous "
                           "portent sur tous les flux.")
            # Récupération du rectangle sélectionné et calcul sur les flux concernés
            zone_sel = evenement.selection.get("zone", {}) if evenement else {}
            if zone_sel.get("X") and zone_sel.get("Y"):
                x0, x1 = sorted(zone_sel["X"])
                y0, y1 = sorted(zone_sel["Y"])
                sous = d[d["X"].between(x0, x1) & d["Y"].between(y0, y1)]
                s1, s2, s3 = st.columns(3)
                s1.metric("Flux dans la zone", len(sous), border=True)
                s2.metric("Attaques dans la zone", int((sous["Classe"] == "Attaque").sum()), border=True)
                s3.metric("Taux d'attaque (zone)",
                          f"{(sous['Classe'] == 'Attaque').mean() * 100:.1f} %" if len(sous) else "—",
                          border=True)
    with g2.container(border=True):
        st.markdown("**Répartition des classes**")
        comptes = pd.DataFrame({"Classe": ["Normal", "Attaque"],
                                "Flux": [int((y == 0).sum()), int((y == 1).sum())]})
        donut = alt.Chart(comptes).mark_arc(innerRadius=62).encode(
            theta="Flux:Q", tooltip=["Classe", "Flux"],
            color=alt.Color("Classe:N", scale=ECHELLE, legend=alt.Legend(orient="bottom", title=None)))
        st.altair_chart(donut, width="stretch")

    # --- Règle d'alerte (inspirée de la règle « taux > 20 % ») + avis automatique ---
    r1, r2 = st.columns([3, 2])
    table, variable, seuil = None, None, 20
    with r1.container(border=True):
        st.markdown("**Règle d'alerte** · taux d'attaque par tranche d'une variable")
        if numeriques:
            k1, k2 = st.columns([2, 1])
            variable = k1.selectbox("Variable surveillée", numeriques, key="var_regle")
            seuil = k2.slider("Seuil d'alerte (%)", 1, 100, 20)
            table = regle_alerte(df, y, variable, seuil)
            if table is None:
                st.info("Cette variable ne peut pas être découpée en tranches.")
            else:
                st.dataframe(table, hide_index=True, width="stretch", column_config={
                    "Taux d'attaque (%)": st.column_config.ProgressColumn(
                        "Taux d'attaque (%)", min_value=0, max_value=100, format="%.1f")})
        else:
            st.info("Aucune variable numérique disponible.")
    with r2.container(border=True):
        st.markdown("**Avis automatique**")
        for phrase in avis_donnees(df, y, cible, variable, table, seuil):
            st.markdown("• " + phrase)

    with st.expander("Aperçu du tableau et statistiques"):
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        st.dataframe(df.describe().T, width="stretch")


def onglet_entrainement(df, cible):
    st.subheader("Prétraitement et entraînement")
    if normaliser_cible(df[cible].dropna()) is None:
        st.error("La cible n'est pas exploitable (voir l'onglet Données).")
        return
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            doublons = st.checkbox("Supprimer les doublons (avant le découpage)", value=True)
            mode_nan = st.radio("Valeurs manquantes", ["Remplacer par 0", "Supprimer les lignes"],
                                horizontal=True)
        with c2:
            choix = st.multiselect("Algorithmes à comparer", list(MODELES), default=list(MODELES))
    st.caption("Découpage stratifié : 70 % entraînement, 15 % validation, 15 % test.")
    if st.button("Lancer l'entraînement", type="primary", disabled=not choix):
        with st.spinner("Entraînement en cours…"):
            X_brut, y, journal = preparer(df, cible, doublons, mode_nan)
            st.session_state.paquet = entrainer(X_brut, y, choix)
            st.session_state.spec = specification_formulaire(X_brut)
            st.session_state.journal = journal
        st.success("Entraînement terminé. Consultez l'onglet Résultats.")
    for ligne in st.session_state.get("journal", []):
        st.write("•", ligne)


def onglet_resultats():
    paquet = st.session_state.get("paquet")
    if paquet is None:
        st.info("Lancez d'abord l'entraînement dans l'onglet précédent.")
        return
    n_tr, n_val, n_te = paquet["tailles"]
    nom = paquet["meilleur"]
    y_te = paquet["y_test"]
    pred = paquet["modeles"][nom].predict(entree_modele(paquet, nom, paquet["X_test"]))

    # --- Boussole de détection : 4 jauges circulaires sur le test final ---
    with st.container(border=True):
        st.markdown(f"**Boussole de détection · test final · {nom}**")
        st.caption(f"Meilleur F1 en validation. Entraînement : {n_tr} lignes · "
                   f"Validation : {n_val} · Test : {n_te}")
        valeurs = [("Accuracy", accuracy_score(y_te, pred), "#8FB0D3"),
                   ("Précision", precision_score(y_te, pred, zero_division=0), "#2DD4BF"),
                   ("Recall", recall_score(y_te, pred, zero_division=0), "#FF6B3D"),
                   ("F1-score", f1_score(y_te, pred, zero_division=0), "#F5C451")]
        for col, (titre, val, coul) in zip(st.columns(4), valeurs):
            with col:
                st.altair_chart(jauge(titre, float(val), coul), width="content")

    col_cm, col_avis = st.columns([1, 1])
    with col_cm.container(border=True):
        st.markdown("**Matrice de confusion (test)**")
        st.altair_chart(graphique_confusion(y_te, pred), width="stretch")
        st.caption("Faux négatifs = attaques non détectées ; faux positifs = fausses alertes.")
    with col_avis.container(border=True):
        st.markdown("**Avis automatique**")
        for phrase in avis_modele(paquet, nom, y_te, pred):
            st.markdown("• " + phrase)
        modele = paquet["modeles"][nom]
        if hasattr(modele, "feature_importances_"):
            st.markdown("**Variables les plus influentes**")
            imp = pd.DataFrame({"Variable": paquet["colonnes"],
                                "Importance": modele.feature_importances_})
            imp = imp.sort_values("Importance", ascending=False).head(8)
            st.altair_chart(alt.Chart(imp).mark_bar(color="#2DD4BF").encode(
                x="Importance:Q", y=alt.Y("Variable:N", sort="-x", title=None)).properties(height=200),
                width="stretch")

    with st.container(border=True):
        st.markdown("**Comparaison des algorithmes (jeu de validation)**")
        tableau = paquet["tableau"].reset_index(drop=True)
        barres = {m: st.column_config.ProgressColumn(m, min_value=0, max_value=1, format="%.3f")
                  for m in ["Accuracy", "Précision", "Recall", "F1-score"]}
        st.dataframe(tableau, hide_index=True, width="stretch", column_config=barres)
        long = tableau.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("Modèle:N", axis=alt.Axis(labelAngle=0), title=None),
            xOffset="Métrique:N",
            y=alt.Y("Valeur:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Métrique:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(domain=["Accuracy", "Précision", "Recall", "F1-score"],
                                            range=["#8FB0D3", "#2DD4BF", "#FF6B3D", "#F5C451"])),
            tooltip=["Modèle", "Métrique", alt.Tooltip("Valeur:Q", format=".3f")]),
            width="stretch")


def carte_verdict(proba: float):
    """Grande carte colorée avec le verdict et la probabilité d'attaque."""
    attaque = proba >= 0.5
    classe, titre = ("attaque", "Attaque probable") if attaque else ("normal", "Trafic normal")
    st.markdown(
        f'<div class="verdict {classe}"><div class="titre">{titre}</div>'
        f'<div class="sous">Probabilité d\'attaque estimée par le modèle : {proba * 100:.1f} %</div>'
        f'<div class="barre"><div style="width:{proba * 100:.1f}%"></div></div></div>',
        unsafe_allow_html=True)


def onglet_prediction():
    paquet = st.session_state.get("paquet")
    if paquet is None:
        st.info("Entraînez d'abord les modèles pour pouvoir prédire.")
        return
    spec = st.session_state.spec
    st.subheader("Prédire un flux réseau")
    nom = st.selectbox("Modèle utilisé", list(paquet["modeles"]),
                       index=list(paquet["modeles"]).index(paquet["meilleur"]))
    mode = st.radio("Mode", ["Un seul flux", "Fichier CSV"], horizontal=True)

    if mode == "Un seul flux":
        valeurs = {}
        colonnes = st.columns(3)
        for i, (nom_col, s) in enumerate(spec.items()):
            with colonnes[i % 3]:
                if s["type"] == "cat":
                    valeurs[nom_col] = st.selectbox(nom_col, s["valeurs"])
                elif s["type"] == "int":
                    valeurs[nom_col] = st.number_input(
                        nom_col, value=int(s["mediane"]), step=1,
                        help=f"Plage observée : {int(s['min'])} à {int(s['max'])}")
                else:
                    valeurs[nom_col] = st.number_input(
                        nom_col, value=float(round(s["mediane"], 3)), step=0.1, format="%.3f",
                        help=f"Plage observée : {s['min']:.3f} à {s['max']:.3f}")
        if st.button("Analyser ce flux", type="primary"):
            _, proba = predire(paquet, nom, pd.DataFrame([valeurs]))
            carte_verdict(float(proba[0]))
    else:
        st.caption("Colonnes attendues : " + ", ".join(spec))
        fichier = st.file_uploader("Flux à analyser (CSV)", type="csv", key="csv_pred")
        if fichier is not None:
            nouveau = pd.read_csv(fichier)
            manquantes = [c for c in spec if c not in nouveau.columns]
            if manquantes:
                st.error("Colonnes manquantes : " + ", ".join(manquantes))
            else:
                classes, probas = predire(paquet, nom, nouveau[list(spec)].fillna(0))
                sortie = nouveau.copy()
                sortie["prediction"] = np.where(classes == 1, "Attaque", "Normal")
                sortie["proba_attaque"] = probas.round(4)
                st.metric("Flux signalés comme attaque", f"{int((classes == 1).sum())} / {len(sortie)}", border=True)
                st.dataframe(sortie, width="stretch", hide_index=True)
                st.download_button("Télécharger les prédictions",
                                   sortie.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


# ----------------------------------------------------------------------------
# 9. APPLICATION PRINCIPALE
# ----------------------------------------------------------------------------
def application():
    with st.sidebar:
        st.image(str(BASE / "logo.png"), width=96)
        st.markdown("### Sentinelle DDoS")
        envoi = st.file_uploader("Utiliser un autre CSV", type="csv",
                                 help="Par défaut : dataset.csv du dossier de l'application.")
        if st.button("Se déconnecter", width="stretch"):
            st.session_state.clear()
            st.rerun()

    df = charger_csv(envoi.getvalue() if envoi else None)
    source = envoi.name if envoi else "dataset.csv"
    cible = st.sidebar.selectbox("Colonne cible (0 = normal, 1 = attaque)", df.columns,
                                 index=list(df.columns).index(detecter_cible(df)))
    st.sidebar.caption(f"Source : {source}")

    # Si le jeu de données change, on efface les anciens modèles
    signature = (source, df.shape, cible)
    if st.session_state.get("signature") != signature:
        for cle in ["paquet", "spec", "journal"]:
            st.session_state.pop(cle, None)
        st.session_state.signature = signature

    entete("Sentinelle DDoS", "Détection et prédiction d'attaques réseau")
    t1, t2, t3, t4 = st.tabs(["Observatoire", "Entraînement", "Résultats", "Prédiction"])
    with t1:
        onglet_donnees(df, cible)
    with t2:
        onglet_entrainement(df, cible)
    with t3:
        onglet_resultats()
    with t4:
        onglet_prediction()


if st.session_state.get("connecte"):
    application()
else:
    page_connexion()
