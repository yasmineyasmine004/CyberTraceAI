
import pandas as pd


def get_demo_events():
    """Événements entièrement synthétiques pour la démonstration."""

    events = [
        {
            "event_id": "EVT-001",
            "timestamp": "2026-10-18 09:00:00",
            "source_ip": "192.0.2.10",
            "destination_ip": "192.0.2.20",
            "event_type": "Port scan",
            "severity": "Moyen",
            "description": "Plusieurs ports ciblés en peu de temps",
        },
        {
            "event_id": "EVT-002",
            "timestamp": "2026-10-18 09:02:00",
            "source_ip": "192.0.2.10",
            "destination_ip": "192.0.2.20",
            "event_type": "Échecs d'authentification",
            "severity": "Élevé",
            "description": "Série fictive de tentatives de connexion échouées",
        },
        {
            "event_id": "EVT-003",
            "timestamp": "2026-10-18 09:04:00",
            "source_ip": "192.0.2.10",
            "destination_ip": "192.0.2.20",
            "event_type": "Flux réseau atypique",
            "severity": "Élevé",
            "description": "Comportement réseau signalé par le détecteur",
        },
        {
            "event_id": "EVT-004",
            "timestamp": "2026-10-18 09:20:00",
            "source_ip": "192.0.2.30",
            "destination_ip": "192.0.2.40",
            "event_type": "Connexion normale",
            "severity": "Faible",
            "description": "Événement indépendant de contrôle",
        },
    ]

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def correlate_events(events, window_minutes=10):
    """
    Regroupe les événements partageant une même IP source
    et une même IP destination dans une fenêtre temporelle.

    Cette règle est une démonstration simple, pas un moteur
    de corrélation de production.
    """
    if events.empty:
        return []

    events = events.sort_values("timestamp").copy()
    incidents = []

    for (src, dst), group in events.groupby(
        ["source_ip", "destination_ip"]
    ):
        group = group.sort_values("timestamp")
        current = []

        for _, event in group.iterrows():
            if not current:
                current = [event]
                continue

            elapsed = (
                event["timestamp"] - current[-1]["timestamp"]
            ).total_seconds() / 60

            if elapsed <= window_minutes:
                current.append(event)
            else:
                incidents.append(pd.DataFrame(current))
                current = [event]

        if current:
            incidents.append(pd.DataFrame(current))

    return incidents
