
from pathlib import Path
from incident_engine import build_incidents
from event_correlator import get_demo_events, correlate_events

import pandas as pd
import plotly.express as px
import streamlit as st
import json
import joblib
import numpy as np

st.set_page_config(
    page_title="CyberTrace AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CyberTrace AI")
st.caption("Détection comportementale et analyse des flux réseau")

path = Path("data/processed/detection_results.parquet")

if not path.exists():
    st.error("Fichier absent. Lance d'abord python detector.py.")
    st.stop()

df = pd.read_parquet(path)

required = {"Prediction", "Label", "Anomaly_Score"}
if not required.issubset(df.columns):
    st.error(f"Colonnes manquantes : {required - set(df.columns)}")
    st.stop()

df["Label"] = df["Label"].fillna("Unknown").astype(str)
df["Prediction"] = df["Prediction"].fillna("Unknown").astype(str)

# Les labels servent à évaluer le modèle, pas à produire ses prédictions.
df["Is_Attack"] = df["Label"].str.strip().str.lower().ne("benign")
df["Is_Detected_Attack"] = (
    df["Is_Attack"] & df["Prediction"].eq("Suspect")
)
df["Is_Missed_Attack"] = (
    df["Is_Attack"] & df["Prediction"].eq("Normal")
)
df["Is_False_Alarm"] = (
    ~df["Is_Attack"] & df["Prediction"].eq("Suspect")
)

total = len(df)
suspects = int(df["Prediction"].eq("Suspect").sum())
attacks = int(df["Is_Attack"].sum())
detected = int(df["Is_Detected_Attack"].sum())
missed = int(df["Is_Missed_Attack"].sum())

st.markdown("### Vue d'ensemble")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flux analysés", f"{total:,}")
c2.metric("Flux suspects", f"{suspects:,}")
c3.metric("Attaques détectées", f"{detected:,}")
c4.metric("Attaques manquées", f"{missed:,}")

if attacks:
    st.caption(
        f"Recall sur cet échantillon : {detected / attacks:.1%}. "
        "Les résultats dépendent du jeu de test."
    )

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("### Prédictions du modèle")
    counts = (
        df["Prediction"]
        .value_counts()
        .rename_axis("Prédiction")
        .reset_index(name="Nombre")
    )
    fig = px.pie(
        counts,
        names="Prédiction",
        values="Nombre",
        hole=0.45,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Types de trafic dans le test")
    label_counts = (
        df["Label"]
        .value_counts()
        .head(10)
        .rename_axis("Label")
        .reset_index(name="Nombre")
    )
    fig = px.bar(
        label_counts,
        x="Label",
        y="Nombre",
        title="Top 10 des labels",
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("### 🚨 Flux suspects")

suspect_df = df[df["Prediction"].eq("Suspect")].copy()
suspect_df = suspect_df.sort_values(
    "Anomaly_Score",
    ascending=True,
)

st.caption(
    "Un score plus faible indique un flux plus atypique "
    "selon ce modèle. Suspect ne signifie pas attaque confirmée."
)

display_cols = [
    col for col in [
        "Label",
        "Prediction",
        "Anomaly_Score",
        "Protocol",
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "SYN Flag Count",
        "Total Fwd Packets",
    ]
    if col in suspect_df.columns
]

st.dataframe(
    suspect_df[display_cols].head(100),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Télécharger les flux suspects (CSV)",
    data=suspect_df.to_csv(index=False).encode("utf-8"),
    file_name="flux_suspects.csv",
    mime="text/csv",
)

st.divider()
st.markdown("### ⚠️ Attaques manquées")

missed_df = df[df["Is_Missed_Attack"]].copy()
missed_df = missed_df.sort_values(
    "Anomaly_Score",
    ascending=False,
)

st.write(
    f"Le modèle a classé {missed:,} flux étiquetés comme attaques "
    "dans les données comme étant normaux."
)

st.dataframe(
    missed_df[display_cols].head(50),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.markdown("## 🧩 Incidents candidats")

st.caption(
    "Les groupes sont construits à partir des caractéristiques "
    "des flux suspects. Le niveau de risque est une priorité "
    "heuristique, pas une probabilité d'attaque."
)

incidents = build_incidents(df)


if incidents.empty:
    st.info("Aucun flux suspect à regrouper.")
else:
    st.metric("Groupes candidats", len(incidents))

    st.dataframe(
        incidents,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🔍 Examiner un incident")

    selected_id = st.selectbox(
        "Choisir un incident",
        incidents["Incident_ID"].tolist(),
    )

    incident = incidents.loc[
        incidents["Incident_ID"] == selected_id
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Incident", incident["Incident_ID"])
    col2.metric("Priorité", f'{incident["Priorite_score"]}/100')
    col3.metric("Niveau de risque", incident["Risque"])

    st.markdown("#### Pourquoi ce groupe est-il signalé ?")

    signal = incident["Signal"]

    explanations = {
        "Débit de paquets élevé":
            "Le débit de paquets est élevé par rapport aux flux suspects "
            "de cet échantillon. Ce signal peut justifier une investigation.",
        "Débit d'octets élevé":
            "Le débit d'octets est élevé par rapport aux flux suspects "
            "de cet échantillon.",
        "Présence de drapeaux SYN":
            "Le flux présente un ou plusieurs drapeaux SYN. "
            "Ce comportement n'est pas une preuve d'attaque à lui seul.",
        "Flux atypique à examiner":
            "Le modèle Isolation Forest a classé ces flux comme suspects "
            "en raison de leurs caractéristiques réseau."
    }

    st.write(explanations.get(signal, "Signal à examiner."))

    st.write(f'**Protocole :** {incident["Protocol"]}')
    st.write(f'**Nombre de flux dans le groupe :** {incident["Nombre_flux"]}')
    st.write(f'**Score moyen d’anomalie :** {incident["Score_moyen"]:.4f}')
    st.write(f'**Score d’anomalie minimal :** {incident["Score_min"]:.4f}')

    st.info(
        "La priorité est un indicateur heuristique. "
        "Ce groupe n'est pas nécessairement un incident confirmé."
    )

    st.markdown("#### Chronologie")

    st.warning(
        "Chronologie réelle indisponible : le dataset utilisé ne contient "
        "pas les horodatages ni les identifiants de machines nécessaires. "
        "Nous ajouterons un scénario synthétique pour illustrer cette "
        "fonctionnalité sans confondre simulation et données réelles."
    )

    st.download_button(
        "Télécharger les groupes candidats (CSV)",
        data=incidents.to_csv(index=False).encode("utf-8"),
        file_name="incidents_candidats.csv",
        mime="text/csv",
    )



st.divider()
st.header("🕒 Corrélation temporelle — Démonstration")

st.warning(
    "Les événements de cette section sont synthétiques. "
    "Les scores sont des règles heuristiques de démonstration, "
    "pas des probabilités réelles d'attaque."
)

window = st.slider(
    "Fenêtre de corrélation (minutes)",
    min_value=1,
    max_value=30,
    value=10,
)

events = get_demo_events()
groups = correlate_events(events, window_minutes=window)

SEVERITY_POINTS = {
    "Faible": 15,
    "Moyen": 35,
    "Élevé": 55,
    "Critique": 80,
}


def score_incident(group):
    """Calcule une priorité heuristique entre 0 et 100."""

    severity_values = [
        SEVERITY_POINTS.get(level, 15)
        for level in group["severity"]
    ]

    # Gravité moyenne des événements du groupe.
    base = sum(severity_values) / len(severity_values)

    event_types = set(group["event_type"])
    bonus = 0

    # Combinaison de signaux nécessitant une investigation.
    if (
        "Port scan" in event_types
        and "Échecs d'authentification" in event_types
    ):
        bonus += 25

    if len(group) >= 3:
        bonus += 10

    duration = (
        group["timestamp"].max() - group["timestamp"].min()
    ).total_seconds() / 60

    if len(group) >= 2 and duration <= 5:
        bonus += 10

    score = min(100, round(base + bonus))

    if score >= 70:
        level = "Élevé"
    elif score >= 40:
        level = "Moyen"
    else:
        level = "Faible"

    reasons = []

    if max(severity_values) >= 55:
        reasons.append("Présence d'un événement de gravité élevée")

    if (
        "Port scan" in event_types
        and "Échecs d'authentification" in event_types
    ):
        reasons.append(
            "Scan réseau suivi d'échecs d'authentification"
        )

    if len(group) >= 3:
        reasons.append("Au moins trois événements corrélés")

    if len(group) >= 2 and duration <= 5:
        reasons.append("Événements rapprochés dans le temps")

    if not reasons:
        reasons.append(
            "Score fondé sur la gravité des événements observés"
        )

    return score, level, reasons


incident_records = []

for index, group in enumerate(groups, start=1):
    score, level, reasons = score_incident(group)

    incident_records.append({
        "Incident_ID": f"INC-DEMO-{index:03d}",
        "IP_source": group.iloc[0]["source_ip"],
        "IP_destination": group.iloc[0]["destination_ip"],
        "Début": group["timestamp"].min(),
        "Fin": group["timestamp"].max(),
        "Nombre_evenements": len(group),
        "Score_risque": score,
        "Gravité": level,
        "Raisons": reasons,
    })

incidents_demo = pd.DataFrame(incident_records)

if incidents_demo.empty:
    st.info("Aucun groupe trouvé pour cette fenêtre.")
else:
    st.subheader("📊 Incidents détectés")

    selected_levels = st.multiselect(
        "Filtrer par niveau de gravité",
        options=["Élevé", "Moyen", "Faible"],
        default=["Élevé", "Moyen", "Faible"],
    )

    filtered = incidents_demo[
        incidents_demo["Gravité"].isin(selected_levels)
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Groupes détectés", len(incidents_demo))
    c2.metric("Groupes affichés", len(filtered))
    c3.metric(
        "Groupes à risque élevé",
        int((filtered["Gravité"] == "Élevé").sum()),
    )

    st.dataframe(
        filtered.drop(columns=["Raisons"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🔍 Rapport détaillé d'un incident")

    if filtered.empty:
        st.info("Aucun incident ne correspond au filtre sélectionné.")
    else:
        selected_id = st.selectbox(
            "Choisir un incident pour le rapport",
            filtered["Incident_ID"].tolist(),
        )

        selected = filtered[
            filtered["Incident_ID"] == selected_id
        ].iloc[0]

        group_index = int(selected_id.split("-")[-1]) - 1
        group = groups[group_index]

        col1, col2, col3 = st.columns(3)
        col1.metric("Score de risque", f'{selected["Score_risque"]}/100')
        col2.metric("Gravité", selected["Gravité"])
        col3.metric("Événements", int(selected["Nombre_evenements"]))

        st.markdown("#### Pourquoi cet incident est-il signalé ?")
        for reason in selected["Raisons"]:
            st.write(f"- {reason}")

        st.markdown("#### Chronologie des événements")
        timeline = group.sort_values("timestamp").copy()
        timeline["timestamp"] = timeline["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        st.dataframe(
            timeline,
            use_container_width=True,
            hide_index=True,
        )

        report = {
            "incident_id": selected["Incident_ID"],
            "data_source": "synthetic_demo",
            "risk_score": int(selected["Score_risque"]),
            "severity": selected["Gravité"],
            "source_ip": selected["IP_source"],
            "destination_ip": selected["IP_destination"],
            "event_count": int(selected["Nombre_evenements"]),
            "reasons": selected["Raisons"],
            "window_minutes": window,
            "timeline": group.assign(
                timestamp=group["timestamp"].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ).to_dict(orient="records"),
            "recommendation": (
                "Examiner les événements et vérifier les journaux "
                "correspondants avant de conclure à une attaque."
            ),
        }

        st.download_button(
            "📥 Télécharger le rapport JSON",
            data=json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            file_name=f"{selected_id}.json",
            mime="application/json",
        )


st.divider()
st.header("📂 Analyser un nouveau fichier CSV")

st.write(
    "Charge un fichier CSV contenant des caractéristiques de flux réseau "
    "compatibles avec CICIDS2017. Choisis le détecteur selon l'objectif."
)

analysis_mode = st.radio(
    "Modèle à utiliser",
    options=["Isolation Forest — anomalies générales", "Random Forest — PortScan connu"],
    horizontal=True,
    help=(
        "Isolation Forest recherche des flux atypiques. Random Forest distingue "
        "les flux PortScan des flux bénins connus ; il ne remplace pas un détecteur "
        "général d'attaques."
    ),
)

uploaded_file = st.file_uploader(
    "Choisir un fichier CSV",
    type=["csv"],
    key="network_csv_upload",
)

if uploaded_file is not None:
    is_portscan_model = analysis_mode.startswith("Random Forest")
    bundle_path = Path(
        "data/processed/portscan_model_bundle.joblib"
        if is_portscan_model
        else "data/processed/model_bundle.joblib"
    )

    if not bundle_path.exists():
        st.error(
            f"Modèle sauvegardé introuvable : {bundle_path}. "
            "Exécute d'abord `python train_model_bundle.py`."
        )
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.error("Le fichier CSV est vide.")
            else:
                bundle = joblib.load(bundle_path)
                model = bundle["model"]
                imputer = bundle["imputer"]
                feature_cols = bundle["feature_cols"]

                available_features = [
                    col for col in feature_cols if col in uploaded_df.columns
                ]
                coverage = len(available_features) / max(len(feature_cols), 1)

                if coverage < 0.70:
                    st.error(
                        f"Fichier incompatible : seulement {len(available_features)} "
                        f"sur {len(feature_cols)} caractéristiques attendues sont présentes "
                        f"({coverage:.0%}). Fournis un CSV CICIDS2017 compatible."
                    )
                else:
                    X_upload = uploaded_df.reindex(columns=feature_cols).copy()
                    for col in feature_cols:
                        X_upload[col] = pd.to_numeric(X_upload[col], errors="coerce")
                    X_upload = X_upload.replace([np.inf, -np.inf], np.nan)
                    X_clean = imputer.transform(X_upload)

                    results = uploaded_df.copy()
                    if is_portscan_model:
                        probabilities = model.predict_proba(X_clean)[:, 1]
                        predictions = (probabilities >= 0.5).astype(int)
                        results["PortScan_Probability"] = probabilities
                        results["PortScan_Prediction"] = np.where(
                            predictions == 1, "PortScan probable", "Non-PortScan"
                        )
                        # Use a generic display column while clearly preserving model scope.
                        results["Prediction"] = np.where(
                            predictions == 1, "Suspect — PortScan", "Normal / autre"
                        )
                        st.warning(
                            "Ce classificateur reconnaît les PortScan appris à partir du "
                            "dataset CICIDS2017. « Non-PortScan » ne signifie pas « bénin » : "
                            "une autre famille d'attaque peut être présente."
                        )
                        score_col = "PortScan_Probability"
                    else:
                        predictions = model.predict(X_clean)
                        scores = model.decision_function(X_clean)
                        results["Anomaly_Score"] = scores
                        results["Prediction"] = [
                            "Suspect" if pred == -1 else "Normal"
                            for pred in predictions
                        ]
                        score_col = "Anomaly_Score"
                        st.caption(
                            "Le score d'anomalie est propre à Isolation Forest ; "
                            "un score plus faible indique un flux plus atypique."
                        )

                    total_upload = len(results)
                    total_suspect = int(
                        results["Prediction"].astype(str).str.startswith("Suspect").sum()
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Flux analysés", f"{total_upload:,}")
                    c2.metric("Flux signalés", f"{total_suspect:,}")
                    c3.metric(
                        "Taux signalé",
                        f"{100 * total_suspect / total_upload:.2f}%" if total_upload else "0%",
                    )

                    st.markdown("### Résultats de l'analyse")
                    sort_ascending = not is_portscan_model
                    if score_col in results.columns:
                        results_to_show = results.sort_values(
                            score_col, ascending=sort_ascending
                        )
                    else:
                        results_to_show = results
                    st.dataframe(
                        results_to_show.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Télécharger les résultats CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name=(
                            "cybertrace_portscan_analysis.csv"
                            if is_portscan_model
                            else "cybertrace_anomaly_analysis.csv"
                        ),
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Impossible d'analyser le fichier : {exc}")
