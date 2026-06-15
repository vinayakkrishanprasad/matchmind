# MatchMind

**Sentiment-driven World Cup match predictions.**

Live site → [vinayakkrishanprasad.github.io/matchmind](https://vinayakkrishanprasad.github.io/matchmind)

---

## What is this?

MatchMind is a research project asking a simple question: does pre-match sentiment carry predictive signal beyond what ELO ratings encode?

The model combines three signals:

- **Momentum score** — derived from actual match results (wins, goal difference, clean sheets, disciplinary events)
- **Media tone** — sentiment scored from NewsAPI and GNews headlines using a 300+ term football-specific lexicon
- **Poisson goal model** — Dixon-Coles style attack/defense parameters updated after each match

Match win/draw/loss probabilities are computed by simulating scoreline distributions. Tournament win probabilities come from 10,000 Monte Carlo bracket simulations run after every update.

---

## How it updates

A GitHub Actions workflow runs every 6 hours automatically:

1. Fetches live match results from football-data.org
2. Updates team attack/defense parameters from finished matches
3. Computes momentum scores from recent form
4. Pulls news sentiment from NewsAPI and GNews
5. Runs 10,000 tournament simulations
6. Commits updated JSON files back to the repo
7. GitHub Pages serves the updated data to the site

The site's Refresh button fetches the latest JSON files on demand.

---

## Methodology

### Poisson Goal Model
Each team's expected goals (xG) are estimated using a Dixon-Coles Poisson regression on 4 years of international match data. Attack strength and defensive weakness are estimated per team with time-decay weighting. Scoreline distributions are simulated to derive win/draw/loss probabilities.

### Sentiment Layer
Text from NewsAPI and GNews is scored using a weighted football-specific lexicon. Multi-word phrases ("red card", "clean sheet", "hat trick") are matched before single words to avoid double-counting. Sentiment shifts each team's xG by up to ±8%.

### Momentum Score
Derived from actual match results:
- Recency-weighted W/D/L over last 3 games (most recent = highest weight)
- Average goal difference per game
- Clean sheet bonus

### Calibration
After each match, model probabilities are compared against closing market odds. Brier scores are tracked for both the sentiment-adjusted model and the ELO baseline.

---

## Data sources

| Source | Used for | Cost |
|--------|----------|------|
| football-data.org | Match results, fixtures | Free |
| NewsAPI | Media sentiment | Free |
| GNews | Secondary media sentiment | Free |

---

## Stack

- Python (scipy, requests)
- GitHub Actions (cron scheduling)
- GitHub Pages (hosting)
- Vanilla HTML/CSS/JS + Chart.js (frontend)

---

## Setup

1. Fork this repo
2. Add three secrets in Settings → Secrets → Actions:
   - `FOOTBALL_DATA_API_KEY` — [football-data.org](https://football-data.org)
   - `NEWS_API_KEY` — [newsapi.org](https://newsapi.org)
   - `GNEWS_API_KEY` — [gnews.io](https://gnews.io)
3. Enable GitHub Pages (Settings → Pages → main branch → / root)
4. Run the workflow manually once from the Actions tab
5. Site goes live at `yourusername.github.io/matchmind`

---

## Research question

Does pre-match crowd sentiment carry predictive alpha beyond what ELO ratings encode?

The answer will be in the Brier scores by the time the final is played on July 19.

---

*Built during the 2026 FIFA World Cup. Not betting advice.*
