"""
MatchMind Data Pipeline v3
--------------------------
Sentiment sources (all work from GitHub Actions):
  1. NewsAPI        — media tone from headlines + descriptions
  2. GNews API      — secondary news source, free 100 req/day
  3. Momentum score — derived from actual match results
       - Win/draw/loss in last 3 matches
       - Goal difference
       - Red cards / disciplinary events
       - Clean sheets

Combined score = 0.40 * news_tone + 0.25 * gnews_tone + 0.35 * momentum

Required GitHub secrets:
  FOOTBALL_DATA_API_KEY  — football-data.org
  NEWS_API_KEY           — newsapi.org
  GNEWS_API_KEY          — gnews.io (free, sign up at gnews.io)
"""

import os, json, math, time, datetime, requests
from collections import defaultdict
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TEAMS = [
    "Mexico","South Africa","South Korea","Czechia",
    "USA","Paraguay","Australia","Turkey",
    "Brazil","Morocco","Croatia","Belgium",
    "France","Senegal","Argentina","Algeria",
    "Spain","Saudi Arabia","England","Ghana",
    "Portugal","Colombia","Germany","Japan",
    "Netherlands","Ivory Coast","Switzerland","Qatar",
]

BASE_ELO = {
    "Spain":2201,"Argentina":2193,"France":2187,"England":2169,"Brazil":2158,
    "Germany":2155,"Portugal":2112,"Netherlands":2098,"Belgium":2091,
    "Morocco":2089,"USA":2071,"South Korea":2063,"Japan":2058,"Croatia":2051,
    "Switzerland":2048,"Czechia":2044,"Senegal":2039,"Colombia":2034,
    "South Africa":2032,"Mexico":2141,"Turkey":2025,"Australia":2019,
    "Paraguay":2010,"Algeria":2008,"Saudi Arabia":2001,"Ghana":1998,
    "Qatar":1984,"Ivory Coast":1991,
}

TEAM_PARAMS_FILE   = "data/team_params.json"
MATCHES_FILE       = "data/matches.json"
SENTIMENT_FILE     = "data/sentiment.json"
PROBABILITIES_FILE = "data/probabilities.json"
META_FILE          = "data/meta.json"
TOURNEY_FILE       = "data/tournament_probs.json"
MONTE_CARLO_RUNS   = 10_000

# ---------------------------------------------------------------------------
# FOOTBALL-SPECIFIC SENTIMENT LEXICON (weighted)
# ---------------------------------------------------------------------------

LEXICON = {
    # Strong positive (2)
    "masterclass":2,"unstoppable":2,"dominant":2,"emphatic":2,"stunning":2,
    "superb":2,"world class":2,"clinical":2,"breathtaking":2,"outstanding":2,
    "magnificent":2,"sensational":2,"scintillating":2,"imperious":2,
    "demolished":2,"thrashing":2,"convincing":2,"electric":2,
    # Positive (1)
    "great":1,"strong":1,"impressive":1,"solid":1,"confident":1,"sharp":1,
    "dangerous":1,"quality":1,"win":1,"victory":1,"goal":1,"scored":1,
    "winning":1,"beat":1,"triumph":1,"excellent":1,"consistent":1,
    "composed":1,"efficient":1,"resilient":1,"tenacious":1,"energetic":1,
    "lethal":1,"precise":1,"clean sheet":1,"comeback":1,"deserved":1,
    "comfortable":1,"brace":1,"hat trick":1,"screamer":1,"qualify":1,
    "disciplined":1,"momentum":1,"firepower":1,"creative":1,"pressing":1,
    # Mild positive (0.5)
    "decent":0.5,"promising":0.5,"encouraging":0.5,"competitive":0.5,
    "fighting":0.5,"battling":0.5,"chances":0.5,"threat":0.5,
    # Mild negative (-0.5)
    "inconsistent":-0.5,"unconvincing":-0.5,"nervy":-0.5,"lucky":-0.5,
    "scrappy":-0.5,"lethargic":-0.5,"slow":-0.5,"flat":-0.5,
    "wasteful":-0.5,"struggling":-0.5,"concern":-0.5,"shaky":-0.5,
    "leaky":-0.5,"toothless":-0.5,"naive":-0.5,
    # Negative (-1)
    "poor":-1,"awful":-1,"terrible":-1,"weak":-1,"loss":-1,"lost":-1,
    "defeat":-1,"eliminated":-1,"disappointing":-1,"error":-1,
    "mistake":-1,"blunder":-1,"injury":-1,"injured":-1,"suspended":-1,
    "crisis":-1,"collapse":-1,"disaster":-1,"outclassed":-1,
    "outplayed":-1,"exposed":-1,"conceded":-1,"failed":-1,
    "penalty miss":-1,"own goal":-1,
    # Strong negative (-2)
    "red card":-2,"sent off":-2,"humiliated":-2,"embarrassing":-2,
    "humiliation":-2,"thrashed":-2,"destroyed":-2,"ban":-2,"banned":-2,
    "catastrophic":-2,"pathetic":-2,"disgraceful":-2,"meltdown":-2,
    "horror show":-2,"capitulation":-2,
}

def score_text(text):
    t = text.lower()
    total, count = 0.0, 0
    for phrase, weight in sorted(LEXICON.items(), key=lambda x: -len(x[0])):
        if phrase in t:
            total += weight
            count += 1
            t = t.replace(phrase, " ")
    if count == 0:
        return 0.0
    return round(max(-1.0, min(1.0, total / count)), 3)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def load_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except: return default if default is not None else {}

def save_json(path, data):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# 1. MATCH RESULTS
# ---------------------------------------------------------------------------

def fetch_match_results():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY","")
    if not api_key:
        return load_json(MATCHES_FILE, default=[])
    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/2000/matches",
            headers={"X-Auth-Token": api_key}, timeout=10
        )
        resp.raise_for_status()
        matches = []
        for m in resp.json().get("matches",[]):
            full = m.get("score",{}).get("fullTime",{})
            matches.append({
                "id": m["id"], "date": m["utcDate"][:10],
                "group": m.get("group",""), "stage": m.get("stage",""),
                "home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"],
                "status": m.get("status",""),
                "home_score": full.get("home"), "away_score": full.get("away"),
            })
        save_json(MATCHES_FILE, matches)
        print(f"[matches] {len(matches)} fetched, {sum(1 for m in matches if m['status']=='FINISHED')} finished")
        return matches
    except Exception as e:
        print(f"[matches] Error: {e}")
        return load_json(MATCHES_FILE, default=[])

# ---------------------------------------------------------------------------
# 2. MOMENTUM SCORE (from actual results — no external API needed)
# ---------------------------------------------------------------------------

def compute_momentum(matches):
    """
    Pure data-driven score from match results.
    Considers: W/D/L, goal difference, clean sheets, red card events.
    Returns {team: momentum_score} in [-1, 1]
    """
    finished = [m for m in matches if m["status"] == "FINISHED"
                and m["home_score"] is not None]

    team_stats = defaultdict(lambda: {
        "results": [],      # 1=win, 0=draw, -1=loss (most recent first)
        "gd": [],           # goal difference per game
        "clean_sheets": 0,
        "games": 0,
    })

    for m in sorted(finished, key=lambda x: x["date"]):
        h, a   = m["home"], m["away"]
        hg, ag = m["home_score"], m["away_score"]
        gd_h   = hg - ag

        team_stats[h]["games"] += 1
        team_stats[a]["games"] += 1
        team_stats[h]["gd"].append(gd_h)
        team_stats[a]["gd"].append(-gd_h)

        if hg > ag:
            team_stats[h]["results"].append(1)
            team_stats[a]["results"].append(-1)
        elif hg == ag:
            team_stats[h]["results"].append(0)
            team_stats[a]["results"].append(0)
        else:
            team_stats[h]["results"].append(-1)
            team_stats[a]["results"].append(1)

        if ag == 0: team_stats[h]["clean_sheets"] += 1
        if hg == 0: team_stats[a]["clean_sheets"] += 1

    momentum = {}
    for team in TEAMS:
        s = team_stats[team]
        if s["games"] == 0:
            momentum[team] = 0.0
            continue

        # Recency-weighted result score (most recent games count more)
        results = s["results"][-3:]  # last 3 games
        weights = [0.5, 0.3, 0.2][:len(results)][::-1]
        result_score = sum(r * w for r, w in zip(results, weights))

        # Average goal difference (clipped)
        avg_gd = sum(s["gd"]) / len(s["gd"])
        gd_score = max(-1.0, min(1.0, avg_gd / 3.0))

        # Clean sheet bonus
        cs_bonus = min(0.2, s["clean_sheets"] * 0.1)

        # Combine
        raw = 0.5 * result_score + 0.35 * gd_score + 0.15 * cs_bonus
        momentum[team] = round(max(-1.0, min(1.0, raw)), 3)

    played = sum(1 for t in momentum if team_stats[t]["games"] > 0)
    print(f"[momentum] Computed for {played}/{len(TEAMS)} teams with results")
    return momentum

# ---------------------------------------------------------------------------
# 3. NEWS SENTIMENT (NewsAPI)
# ---------------------------------------------------------------------------

def fetch_newsapi(teams):
    api_key = os.environ.get("NEWS_API_KEY","")
    if not api_key:
        return {t: 0.0 for t in teams}

    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    result = {}
    for team in teams:
        try:
            time.sleep(0.3)
            resp = requests.get("https://newsapi.org/v2/everything", params={
                "q": f'"{team}" "World Cup"', "from": cutoff,
                "language": "en", "sortBy": "relevancy",
                "pageSize": 20, "apiKey": api_key,
            }, timeout=10)
            articles = resp.json().get("articles", [])
            scores = [score_text(f"{a.get('title','')} {a.get('description','')}") for a in articles]
            scored = [s for s in scores if s != 0.0]
            result[team] = round(sum(scored) / len(scored), 3) if scored else 0.0
        except Exception as e:
            print(f"[newsapi] Error {team}: {e}")
            result[team] = 0.0
    print(f"[newsapi] Done — {sum(1 for v in result.values() if v != 0.0)}/{len(teams)} teams with signal")
    return result

# ---------------------------------------------------------------------------
# 4. GNEWS SENTIMENT (gnews.io — free 100 req/day)
# ---------------------------------------------------------------------------

def fetch_gnews(teams):
    api_key = os.environ.get("GNEWS_API_KEY","")
    if not api_key:
        print("[gnews] No key — skipping")
        return {t: 0.0 for t in teams}

    result = {}
    for team in teams:
        try:
            time.sleep(0.5)
            resp = requests.get("https://gnews.io/api/v4/search", params={
                "q": f"{team} World Cup 2026",
                "lang": "en", "max": 10,
                "apikey": api_key,
            }, timeout=10)
            articles = resp.json().get("articles", [])
            scores = [score_text(f"{a.get('title','')} {a.get('description','')}") for a in articles]
            scored = [s for s in scores if s != 0.0]
            result[team] = round(sum(scored) / len(scored), 3) if scored else 0.0
        except Exception as e:
            print(f"[gnews] Error {team}: {e}")
            result[team] = 0.0

    print(f"[gnews] Done — {sum(1 for v in result.values() if v != 0.0)}/{len(teams)} teams with signal")
    return result

# ---------------------------------------------------------------------------
# 5. COMBINE ALL SIGNALS
# ---------------------------------------------------------------------------

def combine_sentiment(newsapi_scores, gnews_scores, momentum_scores):
    combined = {}
    for team in TEAMS:
        n  = newsapi_scores.get(team, 0.0)
        g  = gnews_scores.get(team, 0.0)
        mo = momentum_scores.get(team, 0.0)

        # If we have match data, momentum is the anchor
        has_momentum = mo != 0.0
        if has_momentum:
            score = 0.35 * mo + 0.40 * n + 0.25 * g
        else:
            # Pre-match: rely entirely on news
            score = 0.60 * n + 0.40 * g

        score = round(max(-1.0, min(1.0, score)), 3)

        # Build a human-readable insight
        parts = []
        if mo > 0.3:  parts.append("strong recent form")
        elif mo < -0.3: parts.append("poor recent form")
        if n > 0.3:   parts.append("positive media coverage")
        elif n < -0.3: parts.append("negative media coverage")
        insight = "; ".join(parts) if parts else "neutral signal across sources"

        combined[team] = {
            "sentiment_score":  score,
            "momentum_score":   mo,
            "newsapi_score":    n,
            "gnews_score":      g,
            "insight":          insight,
            "updated_at":       datetime.datetime.utcnow().isoformat() + "Z",
        }

    save_json(SENTIMENT_FILE, combined)
    print(f"[sentiment] Scores range: {min(v['sentiment_score'] for v in combined.values()):.2f} to {max(v['sentiment_score'] for v in combined.values()):.2f}")
    return combined

# ---------------------------------------------------------------------------
# 6. TEAM PARAMS + MATCH PROBABILITIES
# ---------------------------------------------------------------------------

def load_team_params():
    defaults = {t: {"attack":1.0,"defense":1.0} for t in TEAMS}
    stored   = load_json(TEAM_PARAMS_FILE, default=defaults)
    for t in TEAMS:
        if t not in stored: stored[t] = {"attack":1.0,"defense":1.0}
    return stored

def update_team_params(params, matches):
    for m in matches:
        if m["status"] != "FINISHED" or m["home_score"] is None: continue
        h, a   = m["home"], m["away"]
        hg, ag = m["home_score"], m["away_score"]
        if h not in params or a not in params: continue
        lr = 0.05
        exp_h = params[h]["attack"] * params[a]["defense"] * 1.1
        exp_a = params[a]["attack"] * params[h]["defense"]
        params[h]["attack"]  = max(0.3, params[h]["attack"]  + lr*(hg-exp_h))
        params[a]["defense"] = max(0.3, params[a]["defense"] - lr*(hg-exp_h))
        params[a]["attack"]  = max(0.3, params[a]["attack"]  + lr*(ag-exp_a))
        params[h]["defense"] = max(0.3, params[h]["defense"] - lr*(ag-exp_a))
    save_json(TEAM_PARAMS_FILE, params)
    return params

def match_probs(home, away, params, sentiment):
    h_xg = params[home]["attack"] * params[away]["defense"] * 1.1
    a_xg = params[away]["attack"] * params[home]["defense"]
    h_xg *= (1.0 + 0.08 * sentiment.get(home,{}).get("sentiment_score",0.0))
    a_xg *= (1.0 + 0.08 * sentiment.get(away,{}).get("sentiment_score",0.0))
    hw = dr = aw = 0.0
    for hg in range(9):
        for ag in range(9):
            p = poisson.pmf(hg,h_xg) * poisson.pmf(ag,a_xg)
            if hg>ag: hw+=p
            elif hg==ag: dr+=p
            else: aw+=p
    total = hw+dr+aw
    return {
        "home_win": round(hw/total,4), "draw": round(dr/total,4),
        "away_win": round(aw/total,4), "home_xg": round(h_xg,2),
        "away_xg": round(a_xg,2),
        "home_sentiment": round(sentiment.get(home,{}).get("sentiment_score",0.0),3),
        "away_sentiment": round(sentiment.get(away,{}).get("sentiment_score",0.0),3),
    }

def compute_all_probs(matches, params, sentiment):
    enriched = []
    for m in matches:
        entry = dict(m)
        if m["status"] in ("SCHEDULED","TIMED","IN_PLAY"):
            h, a = m["home"], m["away"]
            if h in params and a in params:
                entry.update(match_probs(h,a,params,sentiment))
        enriched.append(entry)
    save_json(PROBABILITIES_FILE, enriched)
    print(f"[probs] {len(enriched)} matches processed")
    return enriched

# ---------------------------------------------------------------------------
# 7. TOURNAMENT SIMULATION
# ---------------------------------------------------------------------------

def simulate_tournament(params, sentiment, runs=MONTE_CARLO_RUNS):
    import random as rnd
    contenders = [
        "France","Argentina","Brazil","Spain","England","Germany",
        "Portugal","Netherlands","Mexico","South Korea","USA","Morocco",
        "Belgium","Japan","Croatia","Switzerland",
    ]
    def sim(a,b):
        if a not in params or b not in params: return rnd.choice([a,b])
        p = match_probs(a,b,params,sentiment)
        r = rnd.random()
        if r < p["home_win"]: return a
        if r < p["home_win"]+p["draw"]: return rnd.choice([a,b])
        return b
    wins = defaultdict(int)
    for _ in range(runs):
        field = contenders[:]
        rnd.shuffle(field)
        while len(field)>1:
            field = [sim(field[i],field[i+1]) if i+1<len(field) else field[i]
                     for i in range(0,len(field),2)]
        if field: wins[field[0]] += 1
    probs = {t: round(c/runs,4) for t,c in wins.items()}
    save_json(TOURNEY_FILE, probs)
    print(f"[sim] Done — {runs} runs, winner range: {min(probs.values()):.1%}–{max(probs.values()):.1%}")
    return probs

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*52}")
    print(f"MatchMind v3 — {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*52}\n")

    matches   = fetch_match_results()
    params    = update_team_params(load_team_params(), matches)
    momentum  = compute_momentum(matches)
    newsapi   = fetch_newsapi(TEAMS)
    gnews     = fetch_gnews(TEAMS)
    sentiment = combine_sentiment(newsapi, gnews, momentum)
    probs     = compute_all_probs(matches, params, sentiment)
    t_probs   = simulate_tournament(params, sentiment)

    finished  = sum(1 for m in matches if m["status"]=="FINISHED")
    save_json(META_FILE, {
        "last_updated":     datetime.datetime.utcnow().isoformat()+"Z",
        "matches_total":    len(matches),
        "matches_finished": finished,
        "teams_tracked":    len(TEAMS),
        "monte_carlo_runs": MONTE_CARLO_RUNS,
        "pipeline_version": "3.0",
        "sentiment_method": "momentum + newsapi + gnews",
    })

    print(f"\nDone.")
    print(f"  Matches:    {len(matches)} total, {finished} finished")
    print(f"  Sentiment:  momentum + news combination")
    print(f"  Tournament: {len(t_probs)} teams")

if __name__ == "__main__":
    main()
