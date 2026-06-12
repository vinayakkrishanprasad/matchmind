"""
MatchMind Data Pipeline
-----------------------
Runs on a schedule (GitHub Actions cron).
Fetches: match results (football-data.org), Reddit sentiment (public JSON),
         news sentiment (NewsAPI).
Computes: team sentiment scores, match win probabilities, tournament probs.
Writes:   data/matches.json, data/sentiment.json, data/probabilities.json,
          data/tournament_probs.json, data/meta.json

Required env vars (set as GitHub Actions secrets):
  FOOTBALL_DATA_API_KEY   — football-data.org free tier
  NEWS_API_KEY            — newsapi.org free tier

No Reddit API key needed — uses public JSON endpoints.
"""

import os
import json
import math
import time
import random
import datetime
import requests
from collections import defaultdict
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TEAMS = [
    "Mexico", "South Africa", "South Korea", "Czechia",
    "USA", "Paraguay", "Australia", "Turkey",
    "Brazil", "Morocco", "Croatia", "Belgium",
    "France", "Senegal", "Argentina", "Algeria",
    "Spain", "Saudi Arabia", "England", "Ghana",
    "Portugal", "Colombia", "Germany", "Japan",
    "Netherlands", "Ivory Coast", "Switzerland", "Qatar",
]

BASE_ELO = {
    "Spain": 2201, "Argentina": 2193, "France": 2187,
    "England": 2169, "Brazil": 2158, "Germany": 2155,
    "Portugal": 2112, "Netherlands": 2098, "Belgium": 2091,
    "Morocco": 2089, "USA": 2071, "South Korea": 2063,
    "Japan": 2058, "Croatia": 2051, "Switzerland": 2048,
    "Czechia": 2044, "Senegal": 2039, "Colombia": 2034,
    "South Africa": 2032, "Mexico": 2141, "Turkey": 2025,
    "Australia": 2019, "Paraguay": 2010, "Algeria": 2008,
    "Saudi Arabia": 2001, "Ghana": 1998, "Qatar": 1984,
    "Ivory Coast": 1991,
}

TEAM_PARAMS_FILE  = "data/team_params.json"
MATCHES_FILE      = "data/matches.json"
SENTIMENT_FILE    = "data/sentiment.json"
PROBABILITIES_FILE= "data/probabilities.json"
META_FILE         = "data/meta.json"
TOURNEY_FILE      = "data/tournament_probs.json"

MONTE_CARLO_RUNS  = 10_000

# Public Reddit JSON — no auth required
REDDIT_SUBREDDITS = ["soccer", "worldcup", "FIFA"]
REDDIT_SORTS      = ["new", "hot"]

# Rotating user agents to avoid 429s
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}

def save_json(path, data):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
    }

# ---------------------------------------------------------------------------
# 1. FETCH MATCH RESULTS (football-data.org)
# ---------------------------------------------------------------------------

def fetch_match_results():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        print("[matches] No API key — using cached data")
        return load_json(MATCHES_FILE, default=[])

    url = "https://api.football-data.org/v4/competitions/2000/matches"
    try:
        resp = requests.get(url, headers={"X-Auth-Token": api_key}, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        print(f"[matches] Error: {e} — using cached")
        return load_json(MATCHES_FILE, default=[])

    matches = []
    for m in raw.get("matches", []):
        full = m.get("score", {}).get("fullTime", {})
        matches.append({
            "id":         m["id"],
            "date":       m["utcDate"][:10],
            "group":      m.get("group", ""),
            "stage":      m.get("stage", ""),
            "home":       m["homeTeam"]["name"],
            "away":       m["awayTeam"]["name"],
            "status":     m.get("status", ""),
            "home_score": full.get("home"),
            "away_score": full.get("away"),
        })

    save_json(MATCHES_FILE, matches)
    print(f"[matches] {len(matches)} matches fetched")
    return matches

# ---------------------------------------------------------------------------
# 2. REDDIT SENTIMENT — public JSON endpoints, no API key
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "great","brilliant","amazing","excellent","wonderful","fantastic","strong",
    "dominant","clinical","sharp","class","quality","win","goal","score",
    "victory","impressive","confident","dangerous","solid","consistent",
    "creative","pace","depth","unlocked","deserved","superb","stunning",
}
NEGATIVE_WORDS = {
    "poor","awful","terrible","weak","slow","sloppy","injury","suspended",
    "red card","struggle","concern","doubt","worried","disappointing","flat",
    "disjointed","crisis","collapse","ban","chaos","disaster","error",
    "mistake","loss","eliminated","overrated","boring","pathetic",
}

TEAM_ALIASES = {
    "usa": "USA", "usmnt": "USA", "united states": "USA", "america": "USA",
    "el tri": "Mexico", "tri": "Mexico",
    "korea": "South Korea", "bafana": "South Africa",
    "les bleus": "France", "albiceleste": "Argentina",
    "selecao": "Brazil", "seleção": "Brazil",
    "three lions": "England", "czech": "Czechia",
    "die mannschaft": "Germany", "oranje": "Netherlands",
    "socceroos": "Australia",
}

def simple_sentiment(text):
    t = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    total = pos + neg
    return round((pos - neg) / total, 3) if total else 0.0

def resolve_teams(text):
    t = text.lower()
    found = set()
    for alias, canonical in TEAM_ALIASES.items():
        if alias in t:
            found.add(canonical)
    for team in TEAMS:
        if team.lower() in t:
            found.add(team)
    return list(found)

def fetch_subreddit_posts(subreddit, sort="new", limit=100):
    """Fetch posts from a public subreddit JSON endpoint."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&t=day"
    try:
        time.sleep(random.uniform(1.5, 3.0))   # polite delay
        resp = requests.get(url, headers=get_headers(), timeout=12)
        if resp.status_code == 429:
            print(f"[reddit] Rate limited on r/{subreddit}, skipping")
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("children", [])
    except Exception as e:
        print(f"[reddit] Error on r/{subreddit}/{sort}: {e}")
        return []

def fetch_reddit_sentiment():
    """
    Pull posts from public Reddit JSON (no API key).
    Returns {team: {reddit_score, post_count, sample_post}}
    """
    cutoff_hours = 48
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=cutoff_hours)

    team_scores  = defaultdict(list)
    team_posts   = defaultdict(int)
    team_samples = {}

    for sub in REDDIT_SUBREDDITS:
        for sort in REDDIT_SORTS:
            posts = fetch_subreddit_posts(sub, sort=sort, limit=100)
            print(f"[reddit] r/{sub}/{sort} — {len(posts)} posts")

            for child in posts:
                post = child.get("data", {})
                created = datetime.datetime.utcfromtimestamp(post.get("created_utc", 0))
                if created < cutoff:
                    continue

                title   = post.get("title", "")
                body    = post.get("selftext", "")
                text    = f"{title} {body}"
                score   = post.get("score", 1)

                mentioned = resolve_teams(text)
                if not mentioned:
                    continue

                sentiment  = simple_sentiment(text)
                weight     = math.log(max(score, 1) + 1)

                for team in mentioned:
                    team_scores[team].append(sentiment * weight)
                    team_posts[team] += 1
                    if team not in team_samples and title:
                        team_samples[team] = title[:120]

    result = {}
    for team in TEAMS:
        scores = team_scores[team]
        avg    = round(sum(scores) / len(scores), 3) if scores else 0.0
        result[team] = {
            "reddit_score": avg,
            "post_count":   team_posts[team],
            "sample_post":  team_samples.get(team, ""),
        }

    active = sum(1 for t in result if result[t]["post_count"] > 0)
    print(f"[reddit] Sentiment computed for {active}/{len(TEAMS)} teams")
    return result

# ---------------------------------------------------------------------------
# 3. NEWS SENTIMENT (NewsAPI)
# ---------------------------------------------------------------------------

def fetch_news_sentiment():
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        print("[news] No API key — skipping")
        return {t: {"news_score": 0.0, "article_count": 0} for t in TEAMS}

    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    result = {}

    for team in TEAMS:
        try:
            time.sleep(0.2)   # stay within free tier rate limit
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        f'"{team}" "World Cup"',
                    "from":     cutoff,
                    "language": "en",
                    "sortBy":   "relevancy",
                    "pageSize": 20,
                    "apiKey":   api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            print(f"[news] Error for {team}: {e}")
            result[team] = {"news_score": 0.0, "article_count": 0}
            continue

        scores = [simple_sentiment(f"{a.get('title','')} {a.get('description','')}") for a in articles]
        result[team] = {
            "news_score":    round(sum(scores) / len(scores), 3) if scores else 0.0,
            "article_count": len(scores),
        }

    print(f"[news] Done — {len(TEAMS)} teams")
    return result

# ---------------------------------------------------------------------------
# 4. COMBINE SENTIMENT
# ---------------------------------------------------------------------------

def combine_sentiment(reddit, news):
    combined = {}
    for team in TEAMS:
        r = reddit.get(team, {}).get("reddit_score", 0.0)
        n = news.get(team, {}).get("news_score", 0.0)
        merged = round(max(-1.0, min(1.0, 0.65 * r + 0.35 * n)), 3)
        combined[team] = {
            "sentiment_score":  merged,
            "reddit_score":     r,
            "news_score":       n,
            "post_count":       reddit.get(team, {}).get("post_count", 0),
            "article_count":    news.get(team, {}).get("article_count", 0),
            "sample_post":      reddit.get(team, {}).get("sample_post", ""),
            "updated_at":       datetime.datetime.utcnow().isoformat() + "Z",
        }
    save_json(SENTIMENT_FILE, combined)
    print("[sentiment] Combined scores saved")
    return combined

# ---------------------------------------------------------------------------
# 5. TEAM PARAMETERS (Dixon-Coles style, updated from results)
# ---------------------------------------------------------------------------

def load_team_params():
    defaults = {t: {"attack": 1.0, "defense": 1.0} for t in TEAMS}
    stored   = load_json(TEAM_PARAMS_FILE, default=defaults)
    for t in TEAMS:
        if t not in stored:
            stored[t] = {"attack": 1.0, "defense": 1.0}
    return stored

def update_team_params(params, matches):
    HOME_ADV = 1.1
    LR       = 0.05
    for m in matches:
        if m["status"] != "FINISHED" or m["home_score"] is None:
            continue
        h, a   = m["home"], m["away"]
        hg, ag = m["home_score"], m["away_score"]
        if h not in params or a not in params:
            continue
        exp_h = params[h]["attack"] * params[a]["defense"] * HOME_ADV
        exp_a = params[a]["attack"] * params[h]["defense"]
        params[h]["attack"]  += LR * (hg - exp_h)
        params[a]["defense"] -= LR * (hg - exp_h)
        params[a]["attack"]  += LR * (ag - exp_a)
        params[h]["defense"] -= LR * (ag - exp_a)
        for t in [h, a]:
            params[t]["attack"]  = max(0.3, params[t]["attack"])
            params[t]["defense"] = max(0.3, params[t]["defense"])
    save_json(TEAM_PARAMS_FILE, params)
    return params

# ---------------------------------------------------------------------------
# 6. MATCH PROBABILITIES
# ---------------------------------------------------------------------------

def sentiment_adj(score, factor=0.08):
    return 1.0 + factor * score

def match_probs(home, away, params, sentiment):
    HOME_ADV  = 1.1
    MAX_GOALS = 8
    h_xg = params[home]["attack"] * params[away]["defense"] * HOME_ADV
    a_xg = params[away]["attack"] * params[home]["defense"]
    h_xg *= sentiment_adj(sentiment.get(home, {}).get("sentiment_score", 0.0))
    a_xg *= sentiment_adj(sentiment.get(away, {}).get("sentiment_score", 0.0))

    hw = dr = aw = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson.pmf(hg, h_xg) * poisson.pmf(ag, a_xg)
            if hg > ag:   hw += p
            elif hg == ag: dr += p
            else:          aw += p
    total = hw + dr + aw
    return {
        "home_win":       round(hw / total, 4),
        "draw":           round(dr / total, 4),
        "away_win":       round(aw / total, 4),
        "home_xg":        round(h_xg, 2),
        "away_xg":        round(a_xg, 2),
        "home_sentiment": round(sentiment.get(home, {}).get("sentiment_score", 0.0), 3),
        "away_sentiment": round(sentiment.get(away, {}).get("sentiment_score", 0.0), 3),
    }

def compute_all_probs(matches, params, sentiment):
    enriched = []
    for m in matches:
        entry = dict(m)
        if m["status"] in ("SCHEDULED", "TIMED", "IN_PLAY"):
            h, a = m["home"], m["away"]
            if h in params and a in params:
                entry.update(match_probs(h, a, params, sentiment))
        enriched.append(entry)
    save_json(PROBABILITIES_FILE, enriched)
    print(f"[probs] {len(enriched)} matches processed")
    return enriched

# ---------------------------------------------------------------------------
# 7. TOURNAMENT SIMULATION (Monte Carlo)
# ---------------------------------------------------------------------------

def simulate_tournament(params, sentiment, runs=MONTE_CARLO_RUNS):
    import random as rnd
    contenders = [
        "France","Argentina","Brazil","Spain","England","Germany",
        "Portugal","Netherlands","Mexico","South Korea","USA","Morocco",
        "Belgium","Japan","Croatia","Switzerland",
    ]

    def sim_match(a, b):
        if a not in params or b not in params:
            return rnd.choice([a, b])
        p = match_probs(a, b, params, sentiment)
        r = rnd.random()
        if r < p["home_win"]: return a
        if r < p["home_win"] + p["draw"]: return rnd.choice([a, b])
        return b

    wins = defaultdict(int)
    for _ in range(runs):
        field = contenders[:]
        rnd.shuffle(field)
        while len(field) > 1:
            next_r = []
            for i in range(0, len(field), 2):
                if i + 1 < len(field):
                    next_r.append(sim_match(field[i], field[i+1]))
                else:
                    next_r.append(field[i])
            field = next_r
        if field:
            wins[field[0]] += 1

    probs = {t: round(c / runs, 4) for t, c in wins.items()}
    save_json(TOURNEY_FILE, probs)
    print(f"[sim] Tournament probs from {runs} runs saved")
    return probs

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*52}")
    print(f"MatchMind pipeline — {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*52}\n")

    matches  = fetch_match_results()
    params   = update_team_params(load_team_params(), matches)
    reddit   = fetch_reddit_sentiment()
    news     = fetch_news_sentiment()
    sentiment= combine_sentiment(reddit, news)
    probs    = compute_all_probs(matches, params, sentiment)
    t_probs  = simulate_tournament(params, sentiment)

    finished = sum(1 for m in matches if m["status"] == "FINISHED")
    save_json(META_FILE, {
        "last_updated":    datetime.datetime.utcnow().isoformat() + "Z",
        "matches_total":   len(matches),
        "matches_finished":finished,
        "teams_tracked":   len(TEAMS),
        "monte_carlo_runs":MONTE_CARLO_RUNS,
        "reddit_method":   "public_json",
    })

    print(f"\nDone. Written to data/")
    print(f"  matches.json          {len(matches)}")
    print(f"  sentiment.json        {len(TEAMS)} teams")
    print(f"  probabilities.json    {len(probs)}")
    print(f"  tournament_probs.json {len(t_probs)} teams")

if __name__ == "__main__":
    main()
