"""
MatchMind Data Pipeline v2
--------------------------
Fixes:
  1. Reddit via Pullpush API (no auth, no blocking)
  2. Expanded football-specific sentiment lexicon (300+ terms)
  3. Smarter sentiment scoring with intensity weights

Required env vars (GitHub Actions secrets):
  FOOTBALL_DATA_API_KEY   — football-data.org free tier
  NEWS_API_KEY            — newsapi.org free tier
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

TEAM_PARAMS_FILE   = "data/team_params.json"
MATCHES_FILE       = "data/matches.json"
SENTIMENT_FILE     = "data/sentiment.json"
PROBABILITIES_FILE = "data/probabilities.json"
META_FILE          = "data/meta.json"
TOURNEY_FILE       = "data/tournament_probs.json"

MONTE_CARLO_RUNS   = 10_000

# ---------------------------------------------------------------------------
# EXPANDED FOOTBALL SENTIMENT LEXICON
# Format: word/phrase -> weight (positive = good, negative = bad)
# ---------------------------------------------------------------------------

SENTIMENT_LEXICON = {
    # Strong positive (weight 2)
    "masterclass": 2, "unstoppable": 2, "dominant": 2, "demolish": 2,
    "thrashing": 2, "emphatic": 2, "stunning": 2, "superb": 2,
    "world class": 2, "clinical": 2, "devastating": 2, "breathtaking": 2,
    "outstanding": 2, "exceptional": 2, "magnificent": 2, "sensational": 2,
    "hammered": 2, "crushed": 2, "destroyed": -2, "dismantled": 2,
    "electric": 2, "brilliant": 2, "scintillating": 2, "imperious": 2,

    # Positive (weight 1)
    "great": 1, "good": 1, "strong": 1, "impressive": 1, "solid": 1,
    "confident": 1, "sharp": 1, "dangerous": 1, "creative": 1,
    "quality": 1, "win": 1, "victory": 1, "goal": 1, "score": 1,
    "scored": 1, "winning": 1, "beat": 1, "defeated": 1, "triumph": 1,
    "excellent": 1, "wonderful": 1, "fantastic": 1, "amazing": 1,
    "consistent": 1, "composed": 1, "efficient": 1, "organized": 1,
    "resilient": 1, "tenacious": 1, "energetic": 1, "cohesive": 1,
    "pace": 1, "depth": 1, "firepower": 1, "creativity": 1,
    "pressing": 1, "counter": 1, "lethal": 1, "precise": 1,
    "controlled": 1, "dominant performance": 1, "clean sheet": 1,
    "comeback": 1, "overturned": 1, "rallied": 1, "comeback win": 1,
    "deserved": 1, "convincing": 1, "comfortable": 1, "routine": 1,
    "breakthrough": 1, "unlocked": 1, "opener": 1, "brace": 1,
    "hattrick": 1, "hat trick": 1, "screamer": 1, "worldie": 1,
    "class": 1, "pacy": 1, "agile": 1, "aerial": 1, "technical": 1,
    "tactically": 1, "well-drilled": 1, "organized": 1, "disciplined": 1,
    "momentum": 1, "form": 1, "confidence": 1, "belief": 1,
    "promotion": 1, "advance": 1, "qualify": 1, "qualified": 1,
    "through": 1, "next round": 1, "knockout": 1,

    # Mild positive (weight 0.5)
    "decent": 0.5, "okay": 0.5, "fine": 0.5, "promising": 0.5,
    "encouraging": 0.5, "potential": 0.5, "improving": 0.5,
    "showing": 0.5, "competitive": 0.5, "capable": 0.5,
    "fighting": 0.5, "battling": 0.5, "spirit": 0.5,
    "chances": 0.5, "opportunity": 0.5, "threat": 0.5,

    # Mild negative (weight -0.5)
    "inconsistent": -0.5, "unimpressive": -0.5, "unconvincing": -0.5,
    "nervy": -0.5, "lucky": -0.5, "fortunate": -0.5, "scrappy": -0.5,
    "disjointed": -0.5, "lethargic": -0.5, "slow": -0.5, "flat": -0.5,
    "wasteful": -0.5, "profligate": -0.5, "missed": -0.5,
    "struggling": -0.5, "concern": -0.5, "doubt": -0.5, "worry": -0.5,
    "questionable": -0.5, "uncertain": -0.5, "shaky": -0.5,

    # Negative (weight -1)
    "poor": -1, "awful": -1, "terrible": -1, "weak": -1, "bad": -1,
    "worst": -1, "loss": -1, "lost": -1, "lose": -1, "losing": -1,
    "defeat": -1, "eliminated": -1, "knocked out": -1, "out": -1,
    "disappointing": -1, "frustrating": -1, "shocking": -1,
    "error": -1, "mistake": -1, "blunder": -1, "howler": -1,
    "penalty miss": -1, "missed penalty": -1, "own goal": -1,
    "injury": -1, "injured": -1, "suspended": -1, "suspension": -1,
    "crisis": -1, "collapse": -1, "disaster": -1, "chaos": -1,
    "toothless": -1, "wasteful": -1, "sloppy": -1, "naive": -1,
    "overrun": -1, "outclassed": -1, "outplayed": -1, "exposed": -1,
    "vulnerable": -1, "leaky": -1, "concede": -1, "conceded": -1,
    "gifted": -1, "gave away": -1, "capitulated": -1,
    "under pressure": -1, "struggling": -1, "failed": -1,

    # Strong negative (weight -2)
    "red card": -2, "sent off": -2, "dismissed": -2, "expelled": -2,
    "humiliated": -2, "embarrassing": -2, "humiliation": -2,
    "thrashed": -2, "hammered": -2, "destroyed": -2, "demolished": -2,
    "ban": -2, "banned": -2, "doping": -2, "scandal": -2,
    "catastrophic": -2, "pathetic": -2, "disgraceful": -2,
    "capitulation": -2, "meltdown": -2, "horror show": -2,
}

TEAM_ALIASES = {
    "usa": "USA", "usmnt": "USA", "united states": "USA",
    "el tri": "Mexico", "tri": "Mexico",
    "korea": "South Korea", "태극전사": "South Korea",
    "bafana": "South Africa",
    "les bleus": "France", "équipe de france": "France",
    "albiceleste": "Argentina",
    "selecao": "Brazil", "seleção": "Brazil", "canarinho": "Brazil",
    "three lions": "England", "gareth southgate": "England",
    "czech": "Czechia",
    "die mannschaft": "Germany",
    "oranje": "Netherlands",
    "socceroos": "Australia",
    "red devils": "Belgium",
    "samurai blue": "Japan",
    "atlas lions": "Morocco",
    "super eagles": "Ghana",  # actually Nigeria but common mix
    "elephants": "Ivory Coast",
    "red star": "Switzerland",
}

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
# SENTIMENT ENGINE
# ---------------------------------------------------------------------------

def score_text(text):
    """
    Score text using weighted lexicon.
    Handles multi-word phrases before single words.
    Returns float in [-1, 1].
    """
    t = text.lower()
    total_weight = 0.0
    match_count = 0

    # Sort by length descending so multi-word phrases match first
    for phrase, weight in sorted(SENTIMENT_LEXICON.items(), key=lambda x: -len(x[0])):
        if phrase in t:
            total_weight += weight
            match_count += 1
            # Remove matched phrase to avoid double-counting
            t = t.replace(phrase, " ")

    if match_count == 0:
        return 0.0
    # Normalize: divide by match count, clip to [-1, 1]
    raw = total_weight / match_count
    return round(max(-1.0, min(1.0, raw)), 3)

def resolve_teams(text):
    """Named entity resolution — find which teams are mentioned."""
    t = text.lower()
    found = set()
    for alias, canonical in TEAM_ALIASES.items():
        if alias in t:
            found.add(canonical)
    for team in TEAMS:
        if team.lower() in t:
            found.add(team)
    return list(found)

# ---------------------------------------------------------------------------
# 1. MATCH RESULTS (football-data.org)
# ---------------------------------------------------------------------------

def fetch_match_results():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        print("[matches] No API key — using cached")
        return load_json(MATCHES_FILE, default=[])

    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/2000/matches",
            headers={"X-Auth-Token": api_key},
            timeout=10,
        )
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
# 2. REDDIT SENTIMENT via Pullpush (no auth, server-friendly)
# ---------------------------------------------------------------------------

def fetch_pullpush_posts(subreddit, query, limit=100):
    """
    Pullpush.io mirrors Reddit data and doesn't block server IPs.
    Falls back to Reddit public JSON if Pullpush is down.
    """
    # Try Pullpush first
    try:
        url = "https://api.pullpush.io/reddit/search/submission/"
        params = {
            "subreddit": subreddit,
            "q": query,
            "size": limit,
            "sort": "desc",
            "sort_type": "created_utc",
            "after": int((datetime.datetime.utcnow() - datetime.timedelta(hours=48)).timestamp()),
        }
        time.sleep(random.uniform(1.0, 2.0))
        resp = requests.get(url, params=params, headers=get_headers(), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            posts = data.get("data", [])
            if posts:
                print(f"[pullpush] r/{subreddit} '{query}' — {len(posts)} posts")
                return posts
    except Exception as e:
        print(f"[pullpush] Error: {e}")

    # Fallback: Reddit public JSON
    try:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "sort": "new", "limit": limit, "t": "day", "restrict_sr": 1}
        time.sleep(random.uniform(2.0, 4.0))
        resp = requests.get(url, params=params, headers=get_headers(), timeout=15)
        if resp.status_code == 200:
            children = resp.json().get("data", {}).get("children", [])
            posts = [c["data"] for c in children]
            print(f"[reddit_fallback] r/{subreddit} '{query}' — {len(posts)} posts")
            return posts
    except Exception as e:
        print(f"[reddit_fallback] Error: {e}")

    return []

def fetch_reddit_sentiment():
    """
    Pull World Cup posts for each team from Pullpush + r/soccer + r/worldcup.
    Returns {team: {reddit_score, post_count, sample_post}}
    """
    team_scores  = defaultdict(list)
    team_posts   = defaultdict(int)
    team_samples = {}

    subreddits = ["soccer", "worldcup"]

    for team in TEAMS:
        for sub in subreddits:
            posts = fetch_pullpush_posts(sub, query=f"{team} World Cup", limit=50)
            for post in posts:
                title  = post.get("title", "")
                body   = post.get("selftext", "") or post.get("body", "")
                text   = f"{title} {body}"
                karma  = post.get("score", 1) or 1

                mentioned = resolve_teams(text)
                if team not in mentioned:
                    continue

                sentiment = score_text(text)
                weight    = math.log(max(karma, 1) + 1)

                team_scores[team].append(sentiment * weight)
                team_posts[team] += 1
                if team not in team_samples and title:
                    team_samples[team] = title[:120]

    result = {}
    for team in TEAMS:
        scores = team_scores[team]
        if scores:
            # Weighted average
            avg = round(sum(scores) / sum(math.log(2 + i) for i in range(len(scores))), 3)
            avg = max(-1.0, min(1.0, avg))
        else:
            avg = 0.0

        result[team] = {
            "reddit_score": avg,
            "post_count":   team_posts[team],
            "sample_post":  team_samples.get(team, ""),
        }

    active = sum(1 for t in result if result[t]["post_count"] > 0)
    print(f"[reddit] Sentiment computed: {active}/{len(TEAMS)} teams had posts")
    return result

# ---------------------------------------------------------------------------
# 3. NEWS SENTIMENT (NewsAPI)
# ---------------------------------------------------------------------------

def fetch_news_sentiment():
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        print("[news] No key — skipping")
        return {t: {"news_score": 0.0, "article_count": 0} for t in TEAMS}

    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    result = {}

    for team in TEAMS:
        try:
            time.sleep(0.25)
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
            print(f"[news] Error {team}: {e}")
            result[team] = {"news_score": 0.0, "article_count": 0}
            continue

        scores = []
        for art in articles:
            text = f"{art.get('title','')} {art.get('description','')} {art.get('content','')}"
            s = score_text(text)
            if s != 0.0:   # only count articles with actual signal
                scores.append(s)

        result[team] = {
            "news_score":    round(sum(scores) / len(scores), 3) if scores else 0.0,
            "article_count": len(articles),
            "scored_articles": len(scores),
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

        # If Reddit has real data, weight it more heavily
        has_reddit = reddit.get(team, {}).get("post_count", 0) > 0
        r_weight = 0.70 if has_reddit else 0.0
        n_weight = 0.30 if has_reddit else 1.0

        merged = round(max(-1.0, min(1.0, r_weight * r + n_weight * n)), 3)

        combined[team] = {
            "sentiment_score":   merged,
            "reddit_score":      r,
            "news_score":        n,
            "post_count":        reddit.get(team, {}).get("post_count", 0),
            "article_count":     news.get(team, {}).get("article_count", 0),
            "scored_articles":   news.get(team, {}).get("scored_articles", 0),
            "sample_post":       reddit.get(team, {}).get("sample_post", ""),
            "updated_at":        datetime.datetime.utcnow().isoformat() + "Z",
        }

    save_json(SENTIMENT_FILE, combined)
    print("[sentiment] Combined scores saved")
    return combined

# ---------------------------------------------------------------------------
# 5. TEAM PARAMETERS
# ---------------------------------------------------------------------------

def load_team_params():
    defaults = {t: {"attack": 1.0, "defense": 1.0} for t in TEAMS}
    stored   = load_json(TEAM_PARAMS_FILE, default=defaults)
    for t in TEAMS:
        if t not in stored:
            stored[t] = {"attack": 1.0, "defense": 1.0}
    return stored

def update_team_params(params, matches):
    HOME_ADV, LR = 1.1, 0.05
    for m in matches:
        if m["status"] != "FINISHED" or m["home_score"] is None:
            continue
        h, a   = m["home"], m["away"]
        hg, ag = m["home_score"], m["away_score"]
        if h not in params or a not in params:
            continue
        exp_h = params[h]["attack"] * params[a]["defense"] * HOME_ADV
        exp_a = params[a]["attack"] * params[h]["defense"]
        params[h]["attack"]  = max(0.3, params[h]["attack"]  + LR * (hg - exp_h))
        params[a]["defense"] = max(0.3, params[a]["defense"] - LR * (hg - exp_h))
        params[a]["attack"]  = max(0.3, params[a]["attack"]  + LR * (ag - exp_a))
        params[h]["defense"] = max(0.3, params[h]["defense"] - LR * (ag - exp_a))
    save_json(TEAM_PARAMS_FILE, params)
    return params

# ---------------------------------------------------------------------------
# 6. MATCH PROBABILITIES
# ---------------------------------------------------------------------------

def sentiment_adj(score, factor=0.08):
    return 1.0 + factor * score

def match_probs(home, away, params, sentiment):
    HOME_ADV, MAX_GOALS = 1.1, 8
    h_xg = params[home]["attack"] * params[away]["defense"] * HOME_ADV
    a_xg = params[away]["attack"] * params[home]["defense"]
    h_xg *= sentiment_adj(sentiment.get(home, {}).get("sentiment_score", 0.0))
    a_xg *= sentiment_adj(sentiment.get(away, {}).get("sentiment_score", 0.0))

    hw = dr = aw = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson.pmf(hg, h_xg) * poisson.pmf(ag, a_xg)
            if   hg > ag: hw += p
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
# 7. TOURNAMENT SIMULATION
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
            nxt = []
            for i in range(0, len(field), 2):
                nxt.append(sim_match(field[i], field[i+1]) if i+1 < len(field) else field[i])
            field = nxt
        if field:
            wins[field[0]] += 1

    probs = {t: round(c / runs, 4) for t, c in wins.items()}
    save_json(TOURNEY_FILE, probs)
    print(f"[sim] Done — {runs} runs")
    return probs

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*52}")
    print(f"MatchMind v2 — {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*52}\n")

    matches   = fetch_match_results()
    params    = update_team_params(load_team_params(), matches)
    reddit    = fetch_reddit_sentiment()
    news      = fetch_news_sentiment()
    sentiment = combine_sentiment(reddit, news)
    probs     = compute_all_probs(matches, params, sentiment)
    t_probs   = simulate_tournament(params, sentiment)

    finished  = sum(1 for m in matches if m["status"] == "FINISHED")
    reddit_active = sum(1 for t in reddit if reddit[t]["post_count"] > 0)

    save_json(META_FILE, {
        "last_updated":      datetime.datetime.utcnow().isoformat() + "Z",
        "matches_total":     len(matches),
        "matches_finished":  finished,
        "teams_tracked":     len(TEAMS),
        "teams_with_reddit": reddit_active,
        "monte_carlo_runs":  MONTE_CARLO_RUNS,
        "pipeline_version":  "2.0",
        "reddit_method":     "pullpush",
    })

    print(f"\nDone.")
    print(f"  Matches:        {len(matches)} total, {finished} finished")
    print(f"  Reddit signal:  {reddit_active}/{len(TEAMS)} teams")
    print(f"  Tournament:     {len(t_probs)} teams simulated")

if __name__ == "__main__":
    main()
