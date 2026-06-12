"""
MatchMind Data Pipeline
-----------------------
Runs on a schedule (GitHub Actions cron).
Fetches: match results, Reddit sentiment, news sentiment.
Computes: team sentiment scores, match win probabilities.
Writes:   data/matches.json, data/sentiment.json, data/probabilities.json

Required env vars (set in GitHub Actions secrets):
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  FOOTBALL_DATA_API_KEY   (football-data.org, free tier)
  NEWS_API_KEY            (newsapi.org, free tier)
"""

import os
import json
import math
import datetime
import requests
import praw
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

# ELO ratings as of June 2026 (baseline — updated from match results)
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

# Attack/defense parameters (Dixon-Coles estimates, updated each run)
# Format: {team: {"attack": float, "defense": float}}
TEAM_PARAMS_FILE = "data/team_params.json"
MATCHES_FILE = "data/matches.json"
SENTIMENT_FILE = "data/sentiment.json"
PROBABILITIES_FILE = "data/probabilities.json"
META_FILE = "data/meta.json"

MONTE_CARLO_RUNS = 10_000


# ---------------------------------------------------------------------------
# 1. FETCH MATCH RESULTS
# ---------------------------------------------------------------------------

def fetch_match_results():
    """
    Pull live match results from football-data.org.
    Free tier covers World Cup matches.
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    headers = {"X-Auth-Token": api_key}

    # Competition ID 2000 = FIFA World Cup on football-data.org
    url = "https://api.football-data.org/v4/competitions/2000/matches"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        print(f"[match fetch] Error: {e} — using cached data")
        return load_json(MATCHES_FILE, default=[])

    matches = []
    for m in raw.get("matches", []):
        status = m.get("status", "")
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        score = m.get("score", {})
        full = score.get("fullTime", {})

        matches.append({
            "id": m["id"],
            "date": m["utcDate"][:10],
            "group": m.get("group", ""),
            "stage": m.get("stage", ""),
            "home": home,
            "away": away,
            "status": status,
            "home_score": full.get("home"),
            "away_score": full.get("away"),
        })

    save_json(MATCHES_FILE, matches)
    print(f"[match fetch] {len(matches)} matches fetched")
    return matches


# ---------------------------------------------------------------------------
# 2. FETCH REDDIT SENTIMENT
# ---------------------------------------------------------------------------

def get_reddit_client():
    return praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
        user_agent="MatchMind/1.0 (research project)"
    )


def simple_sentiment(text: str) -> float:
    """
    Lightweight lexicon-based sentiment scorer.
    Returns a float in [-1, 1].
    Replace with a transformer model for production.
    """
    positive_words = {
        "great", "brilliant", "amazing", "excellent", "wonderful", "fantastic",
        "strong", "dominant", "clinical", "sharp", "class", "quality", "win",
        "goal", "score", "victory", "impressive", "confident", "dangerous",
        "solid", "consistent", "unlocked", "creative", "pace", "depth",
    }
    negative_words = {
        "poor", "awful", "terrible", "weak", "slow", "sloppy", "injury",
        "suspended", "red card", "struggle", "concern", "doubt", "worried",
        "disappointing", "flat", "disjointed", "crisis", "collapse", "ban",
        "chaos", "disaster", "error", "mistake", "loss", "eliminated",
    }
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def resolve_team(text: str, teams: list) -> list:
    """
    Simple named-entity resolution: find which teams are mentioned.
    """
    text_lower = text.lower()
    found = []
    aliases = {
        "usa": "USA", "united states": "USA", "usmnt": "USA",
        "korea": "South Korea", "bafana": "South Africa",
        "tri": "Mexico", "el tri": "Mexico",
        "les bleus": "France", "albiceleste": "Argentina",
        "seleção": "Brazil", "selecao": "Brazil",
        "three lions": "England", "czechia": "Czechia",
        "czech": "Czechia",
    }
    for alias, canonical in aliases.items():
        if alias in text_lower:
            found.append(canonical)
    for team in teams:
        if team.lower() in text_lower:
            found.append(team)
    return list(set(found))


def fetch_reddit_sentiment(teams: list) -> dict:
    """
    Pull posts from r/soccer and r/worldcup from the past 48 hours.
    Attribute sentiment to mentioned teams.
    Returns {team: {"score": float, "post_count": int, "sample": str}}
    """
    reddit = get_reddit_client()
    subreddits = ["soccer", "worldcup"]
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=48)

    team_scores = defaultdict(list)
    team_posts = defaultdict(int)
    team_samples = defaultdict(str)

    for sub_name in subreddits:
        sub = reddit.subreddit(sub_name)
        try:
            posts = list(sub.new(limit=200))
        except Exception as e:
            print(f"[reddit] Error on r/{sub_name}: {e}")
            continue

        for post in posts:
            created = datetime.datetime.utcfromtimestamp(post.created_utc)
            if created < cutoff:
                continue

            text = f"{post.title} {post.selftext}"
            mentioned = resolve_team(text, teams)
            if not mentioned:
                continue

            score = simple_sentiment(text)
            # Weight by karma (log scale, floor at 1)
            weight = math.log(max(post.score, 1) + 1)

            for team in mentioned:
                team_scores[team].append(score * weight)
                team_posts[team] += 1
                if not team_samples[team] and post.title:
                    team_samples[team] = post.title[:120]

    result = {}
    for team in teams:
        scores = team_scores[team]
        if scores:
            weighted_avg = round(sum(scores) / len(scores), 3)
        else:
            weighted_avg = 0.0
        result[team] = {
            "reddit_score": weighted_avg,
            "post_count": team_posts[team],
            "sample_post": team_samples.get(team, ""),
        }

    print(f"[reddit] Sentiment computed for {len([t for t in result if result[t]['post_count'] > 0])} teams")
    return result


# ---------------------------------------------------------------------------
# 3. FETCH NEWS SENTIMENT
# ---------------------------------------------------------------------------

def fetch_news_sentiment(teams: list) -> dict:
    """
    Pull headlines from NewsAPI for each team.
    Returns {team: {"news_score": float, "article_count": int}}
    """
    api_key = os.environ.get("NEWS_API_KEY", "")
    base_url = "https://newsapi.org/v2/everything"
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")

    result = {}
    for team in teams:
        try:
            resp = requests.get(base_url, params={
                "q": f'"{team}" World Cup 2026',
                "from": cutoff,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 20,
                "apiKey": api_key,
            }, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            print(f"[news] Error for {team}: {e}")
            result[team] = {"news_score": 0.0, "article_count": 0}
            continue

        scores = []
        for art in articles:
            text = f"{art.get('title', '')} {art.get('description', '')}"
            scores.append(simple_sentiment(text))

        result[team] = {
            "news_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "article_count": len(scores),
        }

    print(f"[news] News sentiment fetched for {len(teams)} teams")
    return result


# ---------------------------------------------------------------------------
# 4. COMBINE SENTIMENT
# ---------------------------------------------------------------------------

def combine_sentiment(reddit: dict, news: dict, teams: list) -> dict:
    """
    Merge Reddit and news sentiment into a single score per team.
    Reddit weighted 0.65, news 0.35 (Reddit is higher signal for fan sentiment).
    Final score clipped to [-1, 1].
    """
    combined = {}
    for team in teams:
        r = reddit.get(team, {}).get("reddit_score", 0.0)
        n = news.get(team, {}).get("news_score", 0.0)
        merged = round(0.65 * r + 0.35 * n, 3)
        merged = max(-1.0, min(1.0, merged))

        combined[team] = {
            "sentiment_score": merged,
            "reddit_score": r,
            "news_score": n,
            "post_count": reddit.get(team, {}).get("post_count", 0),
            "article_count": news.get(team, {}).get("article_count", 0),
            "sample_post": reddit.get(team, {}).get("sample_post", ""),
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    save_json(SENTIMENT_FILE, combined)
    print(f"[sentiment] Combined scores saved")
    return combined


# ---------------------------------------------------------------------------
# 5. COMPUTE MATCH PROBABILITIES (Poisson + sentiment adjustment)
# ---------------------------------------------------------------------------

def load_team_params() -> dict:
    """Load or initialise Dixon-Coles parameters."""
    defaults = {team: {"attack": 1.0, "defense": 1.0} for team in TEAMS}
    stored = load_json(TEAM_PARAMS_FILE, default=defaults)
    # Fill in any missing teams
    for team in TEAMS:
        if team not in stored:
            stored[team] = {"attack": 1.0, "defense": 1.0}
    return stored


def update_team_params(params: dict, matches: list) -> dict:
    """
    Naive Elo-like parameter update from finished matches.
    For a proper Dixon-Coles fit, replace with scipy.optimize.minimize.
    """
    HOME_ADVANTAGE = 1.1

    for m in matches:
        if m["status"] != "FINISHED":
            continue
        home, away = m["home"], m["away"]
        hg, ag = m["home_score"], m["away_score"]
        if hg is None or ag is None:
            continue
        if home not in params or away not in params:
            continue

        # Simple gradient nudge
        lr = 0.05
        expected_home = params[home]["attack"] * params[away]["defense"] * HOME_ADVANTAGE
        expected_away = params[away]["attack"] * params[home]["defense"]

        params[home]["attack"] += lr * (hg - expected_home)
        params[away]["defense"] -= lr * (hg - expected_home)
        params[away]["attack"] += lr * (ag - expected_away)
        params[home]["defense"] -= lr * (ag - expected_away)

        # Keep positive
        for team in [home, away]:
            params[team]["attack"] = max(0.3, params[team]["attack"])
            params[team]["defense"] = max(0.3, params[team]["defense"])

    save_json(TEAM_PARAMS_FILE, params)
    return params


def sentiment_adjustment(sentiment_score: float, factor: float = 0.08) -> float:
    """
    Map sentiment [-1, 1] to a multiplicative adjustment [1-factor, 1+factor].
    Default factor = 0.08 means sentiment can shift xG by up to ±8%.
    """
    return 1.0 + factor * sentiment_score


def match_probabilities(home: str, away: str, params: dict, sentiment: dict) -> dict:
    """
    Compute win/draw/loss probabilities for a single match using Poisson model
    with sentiment adjustment on expected goals.
    """
    HOME_ADVANTAGE = 1.1
    MAX_GOALS = 8

    base_home_xg = params[home]["attack"] * params[away]["defense"] * HOME_ADVANTAGE
    base_away_xg = params[away]["attack"] * params[home]["defense"]

    # Sentiment adjustments
    home_adj = sentiment_adjustment(sentiment.get(home, {}).get("sentiment_score", 0.0))
    away_adj = sentiment_adjustment(sentiment.get(away, {}).get("sentiment_score", 0.0))

    home_xg = base_home_xg * home_adj
    away_xg = base_away_xg * away_adj

    # Compute scoreline probabilities
    home_win = draw = away_win = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg)
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

    # Normalise floating point errors
    total = home_win + draw + away_win
    return {
        "home_win": round(home_win / total, 4),
        "draw": round(draw / total, 4),
        "away_win": round(away_win / total, 4),
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "home_sentiment": round(sentiment.get(home, {}).get("sentiment_score", 0.0), 3),
        "away_sentiment": round(sentiment.get(away, {}).get("sentiment_score", 0.0), 3),
    }


def compute_all_probabilities(matches: list, params: dict, sentiment: dict) -> list:
    """
    Add probability estimates to all upcoming (SCHEDULED/TIMED) matches.
    """
    enriched = []
    for m in matches:
        entry = dict(m)
        home, away = m["home"], m["away"]

        if m["status"] in ("SCHEDULED", "TIMED", "IN_PLAY"):
            if home in params and away in params:
                probs = match_probabilities(home, away, params, sentiment)
                entry.update(probs)
            else:
                entry.update({"home_win": None, "draw": None, "away_win": None})

        enriched.append(entry)

    save_json(PROBABILITIES_FILE, enriched)
    print(f"[probabilities] {len(enriched)} matches processed")
    return enriched


# ---------------------------------------------------------------------------
# 6. TOURNAMENT SIMULATION (Monte Carlo)
# ---------------------------------------------------------------------------

def simulate_tournament(matches: list, params: dict, sentiment: dict, runs: int = MONTE_CARLO_RUNS) -> dict:
    """
    Monte Carlo bracket simulation.
    Returns {team: win_probability} for all teams.
    """
    import random

    # Build group standings from finished matches
    standings = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    group_teams = defaultdict(set)

    for m in matches:
        if m["status"] != "FINISHED" or m["home_score"] is None:
            continue
        home, away = m["home"], m["away"]
        hg, ag = m["home_score"], m["away_score"]
        grp = m.get("group", "A")

        group_teams[grp].add(home)
        group_teams[grp].add(away)
        standings[home]["gf"] += hg
        standings[home]["ga"] += ag
        standings[away]["gf"] += ag
        standings[away]["ga"] += hg

        if hg > ag:
            standings[home]["pts"] += 3
        elif hg == ag:
            standings[home]["pts"] += 1
            standings[away]["pts"] += 1
        else:
            standings[away]["pts"] += 3

    def simulate_match(team_a: str, team_b: str) -> str:
        """Simulate a single match, return winner."""
        if team_a not in params or team_b not in params:
            return random.choice([team_a, team_b])
        probs = match_probabilities(team_a, team_b, params, sentiment)
        r = random.random()
        if r < probs["home_win"]:
            return team_a
        elif r < probs["home_win"] + probs["draw"]:
            # In knockout: extra time / penalties — coin flip
            return random.choice([team_a, team_b])
        else:
            return team_b

    win_counts = defaultdict(int)

    for _ in range(runs):
        # Simplified: simulate knockout from current known qualifiers
        # In production: resolve group stage first, then bracket
        # For now, use top contenders as the simulated field
        contenders = [
            "France", "Argentina", "Brazil", "Spain",
            "England", "Germany", "Portugal", "Netherlands",
            "Mexico", "South Korea", "USA", "Morocco",
            "Belgium", "Japan", "Croatia", "Switzerland",
        ]
        random.shuffle(contenders)

        # Simulate bracket rounds
        field = contenders[:]
        while len(field) > 1:
            next_round = []
            for i in range(0, len(field), 2):
                if i + 1 < len(field):
                    winner = simulate_match(field[i], field[i+1])
                    next_round.append(winner)
                else:
                    next_round.append(field[i])
            field = next_round

        if field:
            win_counts[field[0]] += 1

    win_probs = {team: round(count / runs, 4) for team, count in win_counts.items()}
    save_json("data/tournament_probs.json", win_probs)
    print(f"[simulation] Tournament win probabilities computed over {runs} runs")
    return win_probs


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def load_json(path: str, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*50}")
    print(f"MatchMind pipeline starting — {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*50}\n")

    # 1. Match results
    matches = fetch_match_results()

    # 2. Team parameters (update from results)
    params = load_team_params()
    params = update_team_params(params, matches)

    # 3. Sentiment
    reddit_sentiment = fetch_reddit_sentiment(TEAMS)
    news_sentiment = fetch_news_sentiment(TEAMS)
    sentiment = combine_sentiment(reddit_sentiment, news_sentiment, TEAMS)

    # 4. Match probabilities
    probabilities = compute_all_probabilities(matches, params, sentiment)

    # 5. Tournament simulation
    tournament_probs = simulate_tournament(matches, params, sentiment)

    # 6. Meta file (used by the website to show last updated time)
    meta = {
        "last_updated": datetime.datetime.utcnow().isoformat() + "Z",
        "matches_total": len(matches),
        "matches_finished": sum(1 for m in matches if m["status"] == "FINISHED"),
        "teams_tracked": len(TEAMS),
        "monte_carlo_runs": MONTE_CARLO_RUNS,
    }
    save_json(META_FILE, meta)

    print(f"\nPipeline complete. Data written to data/")
    print(f"  matches.json         {len(matches)} matches")
    print(f"  sentiment.json       {len(TEAMS)} teams")
    print(f"  probabilities.json   {len(probabilities)} matches with probs")
    print(f"  tournament_probs.json {len(tournament_probs)} teams")
    print(f"  meta.json            {meta['last_updated']}")


if __name__ == "__main__":
    main()
