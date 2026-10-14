import requests
import pandas as pd

url = "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"

params = {
    "rankingScheduleId": "FRS_Male_Football_20260611",
    "language": "en"
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Origin": "https://inside.fifa.com",
    "Referer": "https://inside.fifa.com/"
}

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=30
)

print("Status:", response.status_code)
print("Preview:", response.text[:300])

response.raise_for_status()

data = response.json()

rows = []

for team in data["Results"]:
    rows.append({
        "Team": team["TeamName"][0]["Description"],
        "FIFA_Rank": team["Rank"],
        "FIFA_Points": team["TotalPoints"],
        "Ranking_Date": "2026-06-11"
    })

df = pd.DataFrame(rows)

df = df.sort_values("FIFA_Rank")

df.to_csv("fifa_rankings_june_2026.csv", index=False)

print("Total teams:", len(df))
print(df.head(10))

print("Total teams:", len(df))
print("Missing values:\n", df.isnull().sum())
print("Duplicate teams:", df["Team"].duplicated().sum())
print("Continuation token:", data.get("ContinuationToken"))