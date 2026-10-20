import pandas as pd


def build_incidents(df):
    """Construit des groupes candidats à l'investigation à partir des flux suspects.

    Sans adresse IP ni horodatage, ces groupes ne prouvent pas que les flux
    appartiennent au même incident réel. Le score de priorité est heuristique.
    """
    required = {"Prediction", "Protocol", "Anomaly_Score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes : {required - set(df.columns)}")

    suspects = df[df["Prediction"].astype(str).str.startswith("Suspect")].copy()
    if suspects.empty:
        return pd.DataFrame()

    # Ensure optional features exist so the aggregation remains robust.
    for col in ["Flow Packets/s", "Flow Bytes/s", "SYN Flag Count"]:
        if col not in suspects.columns:
            suspects[col] = 0
        suspects[col] = pd.to_numeric(suspects[col], errors="coerce").fillna(0)

    packet_rate = suspects["Flow Packets/s"]
    byte_rate = suspects["Flow Bytes/s"]
    syn_count = suspects["SYN Flag Count"]
    packet_threshold = packet_rate.quantile(0.90)
    byte_threshold = byte_rate.quantile(0.90)

    suspects["Signal"] = "Flux atypique à examiner"
    suspects.loc[syn_count >= 1, "Signal"] = "Présence de drapeaux SYN"
    suspects.loc[byte_rate >= byte_threshold, "Signal"] = "Débit d'octets élevé"
    suspects.loc[packet_rate >= packet_threshold, "Signal"] = "Débit de paquets élevé"
    suspects["Protocol"] = suspects["Protocol"].astype(str)

    incidents = (
        suspects.groupby(["Protocol", "Signal"], dropna=False)
        .agg(
            Nombre_flux=("Prediction", "size"),
            Score_moyen=("Anomaly_Score", "mean"),
            Score_min=("Anomaly_Score", "min"),
            Debit_paquets_moyen=("Flow Packets/s", "mean"),
            Debit_octets_moyen=("Flow Bytes/s", "mean"),
        )
        .reset_index()
    )
    max_volume = max(int(incidents["Nombre_flux"].max()), 1)
    volume_score = incidents["Nombre_flux"] / max_volume * 100
    score = incidents["Score_moyen"]
    score_min, score_max = score.min(), score.max()
    if score_max > score_min:
        atypicality_score = (score_max - score) / (score_max - score_min) * 100
    else:
        atypicality_score = pd.Series(50.0, index=incidents.index)

    incidents["Priorite_score"] = (0.6 * volume_score + 0.4 * atypicality_score).round(1)

    def risk_level(value):
        if value >= 70:
            return "Élevé"
        if value >= 40:
            return "Moyen"
        return "Faible"

    incidents["Risque"] = incidents["Priorite_score"].apply(risk_level)
    incidents = incidents.sort_values("Priorite_score", ascending=False).reset_index(drop=True)
    incidents.insert(0, "Incident_ID", [f"INC-{i:03d}" for i in range(1, len(incidents) + 1)])
    incidents["Statut"] = "À examiner"
    return incidents
