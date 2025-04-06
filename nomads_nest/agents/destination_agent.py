# agents/destination_agent.py
import gdown

import json
from sentence_transformers import SentenceTransformer, util
json_url = "https://raw.githubusercontent.com/anmolmunnolli/Agent-Alchemysts/refs/heads/main/nomads_nest/data/top_500_cities.json"
response = requests.get(json_url)
model = SentenceTransformer("all-MiniLM-L6-v2")

with open('data/top_500_cities.json') as f:
    destinations = response
def run(state):
    if "persona" not in state:
        return []

    user_embedding = model.encode(", ".join(state["persona"]), normalize_embeddings=True)
    scored_destinations = []

    for dest in destinations:
        if not dest.get("features") or "lat" not in dest or "lng" not in dest:
            continue  # Skip incomplete entries

        dest_embedding = model.encode(dest["features"], normalize_embeddings=True)
        sim_score = util.cos_sim(user_embedding, dest_embedding).item()

        scored_destinations.append({
            "name": dest["city"],
            "score": sim_score,
            "lat": dest["lat"],
            "lng": dest["lng"],
            "country": dest.get("country", "Unknown"),
            "features": dest["features"]
        })

    if not scored_destinations:
        print("⚠️ No destination matches found.")
        return []

    sorted_destinations = sorted(scored_destinations, key=lambda x: x["score"], reverse=True)
    return sorted_destinations[:3]
