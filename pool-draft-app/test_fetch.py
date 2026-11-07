import requests
import time

seasons = ["20232024", "20242025", "20252026"]
for s in seasons:
    print(f"Fetching {s}...")
    res = requests.get(f"https://api.nhle.com/stats/rest/en/skater/summary?cayenneExp=seasonId={s}%20and%20gameTypeId=2&limit=-1")
    print(f"{s}: {len(res.json()['data'])} players")
