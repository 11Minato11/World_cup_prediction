# ⚽ World Cup 2026 Prediction — Monte Carlo Simulator

A powerful **Streamlit** web application that simulates the FIFA World Cup 2026 tournament using **Monte Carlo methods**, **Poisson Generalized Linear Models (GLM)**, **Elo ratings**, and **Dixon-Coles correction**. Predict winners, finalists, knockout probabilities, and group standings through thousands of probabilistic simulations.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Requirements](#data-requirements)
- [Usage Guide](#usage-guide)
- [Screenshots](#screenshots)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The FIFA World Cup 2026 introduces a new **48-team format** with 12 groups (A–L) and a **Round of 32** knockout stage. This simulator uses historical international match data (2014–2025), Elo ratings, and team form to run thousands of Monte Carlo simulations, producing probabilistic predictions for every stage of the tournament.

**Key capabilities:**
- 🏆 Predict World Cup winner, finalist, and 3rd place probabilities
- 📊 Simulate group stage standings (12 groups, 4 teams each)
- 🥅 Estimate knockout progression (R32 → R16 → QF → SF → Final)
- 📈 Visualize win probabilities with confidence intervals
- 🧠 Inspect model parameters and feature importance
- ⚙️ Customize simulation settings (Dixon-Coles, momentum, parallel processing)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Monte Carlo Engine** | Run 1,000 – 50,000 tournament simulations |
| **Poisson GLM** | Goal-scoring model with Elo difference, form, host advantage |
| **Dixon-Coles** | Low-score correlation correction (0-0, 1-0, 0-1, 1-1) |
| **Momentum Factor** | Win-streak bonus up to 12% |
| **Parallel Processing** | Multi-core simulation using all available CPU cores |
| **Interactive Charts** | Plotly bar charts, radar plots, and pie charts |
| **Group Simulation** | Simulate individual groups on-demand |
| **Export Results** | Download prediction tables as CSV |

---

## 🚀 Demo

*(Add a link to your deployed Streamlit app here, e.g., Streamlit Cloud)*

```
https://your-app-name.streamlit.app
```

---

## 📋 Prerequisites

Before running the app, ensure you have:

- **Python 3.9+** installed
- **Git** installed (optional, for cloning)
- Your simulation engine file: **`World_cup_bad.py`**
- Your datasets: **`results.csv`**, **`eloratings.csv`**, **`elo_wc2026.csv`**

---

## 🛠️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/11Minato11/World_cup_prediction.git
cd World_cup_prediction
```

### Step 2: Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `streamlit` — Web application framework
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `plotly` — Interactive visualizations
- `scikit-learn` — Machine learning utilities
- `scipy` — Statistical functions

---

## ▶️ How to Run

### Local Development

```bash
# Make sure your virtual environment is activated
streamlit run app.py
```

The app will automatically open in your default browser at:
```
http://localhost:8501
```

### Using the App

1. **Load Data** — Click **"📥 Load Data & Train Model"** in the sidebar. This loads your CSV datasets and trains the Poisson GLM.
2. **Configure Simulation** — Use the sidebar sliders to set:
   - Number of simulations (1,000 – 50,000)
   - Parallel processing toggle
   - Dixon-Coles correction toggle
   - Momentum factor toggle
3. **Run Simulation** — Click **"🚀 Run Simulation"** to execute the Monte Carlo engine.
4. **Explore Tabs** — Navigate through:
   - **📊 Overview** — Winner, finalist, and top contender probabilities
   - **📋 Groups** — 12 group standings with Elo ratings and per-group simulation
   - **🏆 Knockout** — Full knockout bracket visualization (R32 → Final)
   - **📈 Predictions** — Detailed probability tables with confidence intervals
   - **🧠 Model** — Feature importance, parameter values, and formula documentation
   - **⚙️ Settings** — Advanced parameter tuning and export options

---

## 📁 Project Structure

```
World_cup_prediction/
│
├── app.py                      # Main Streamlit application (UI + logic)
├── World_cup_bad.py            # Simulation engine (your model module) ⭐ REQUIRED
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
└── data/                       # Data files (not tracked by Git)
    ├── results.csv             # International match results (2014–2025)
    ├── eloratings.csv          # Historical Elo ratings
    └── elo_wc2026.csv          # Current Elo ratings for WC 2026 teams
```

> **Note:** `World_cup_bad.py` must be in the same directory as `app.py`. It is imported dynamically at runtime.

---

## 🧠 Model Architecture

### Poisson GLM Formula

The model predicts the logarithm of the expected goal rate (λ) using:

```
log(λ) = β₀ + β₁·ΔElo + β₂·ΔForm × 1.5 + β₃·ΔProgression + β₄·ΔExperience
         + β₅·Host + β₆·Major + β₇·Fatigue + EloBonus + Momentum
```

### Model Parameters

| Parameter | Coefficient | Description | Impact |
|-----------|-------------|-------------|--------|
| β₀ Intercept | +0.247 | Base scoring rate | Neutral |
| β₁ Elo Diff | +0.0021 | Team strength difference | Positive |
| β₂ Form Diff | +0.040 | Recent 5-match performance | Positive |
| β₃ Progression | +0.015 | Elo trajectory over time | Positive |
| β₄ Experience | +0.020 | World Cup history (appearances) | Positive |
| β₅ Host | +0.185 | Home advantage (USA/Mexico/Canada) | **High** |
| β₆ Major | +0.142 | Tournament weight (major vs. friendly) | Positive |
| β₇ Fatigue | −0.089 | Inverse Elo fatigue effect | Negative |

### Additional Components

| Component | Formula | Purpose |
|-----------|---------|---------|
| **EloBonus** | `0.15 × tanh((Elo − 1500)/500 × 2)` | Non-linear Elo boost for elite teams |
| **Momentum** | `1 + 0.03 × consecutive_wins` (max 12%) | Win streak bonus |
| **Fatigue** | `0.30 + 0.65/(1 + exp(−0.005 × (Elo − 1750)))` | Higher fatigue for lower-Elo teams |
| **Dixon-Coles τ** | `ρ = −0.075` for 0-0, 1-0, 0-1, 1-1 | Adjusts low-score dependence |

### Feature Importance (Radar)

1. **Elo Difference** — 85%
2. **Form** — 72%
3. **Host** — 65%
4. **Major Tournament** — 55%
5. **Progression** — 45%
6. **Experience** — 38%
7. **Fatigue** — 30%

---

## 📊 Data Requirements

The app expects three CSV datasets and one Python module:

### 1. `results.csv` — Match Results
```csv
date,home_team,away_team,home_score,away_score,tournament,country,neutral
2014-06-12,Brazil,Croatia,3,1,FIFA World Cup,Brazil,FALSE
...
```

### 2. `eloratings.csv` — Historical Elo Ratings
```csv
date,team,elo
2014-01-01,Brazil,2100
...
```

### 3. `elo_wc2026.csv` — Current Ratings for 2026 Teams
```csv
team,elo
Brazil,2050
France,2020
...
```

### 4. `World_cup_bad.py` — Simulation Engine
This module must expose the following functions/classes:
- `charger_donnees()` — Load datasets
- `calculer_progression_elo(elo_wc2026)` — Calculate Elo progression
- `ajouter_elo(data, elo_historique, progression)` — Merge Elo data
- `ajouter_formes(data)` — Add recent form metrics
- `ajouter_variables_contextuelles(data)` — Add host/fatigue/context vars
- `construire_dataset_poisson(data)` — Build training dataset
- `entrainer_modele(dataset)` — Train Poisson GLM
- `preparer_donnees_historiques(data, elo_hist, elo_wc2026)` — Prep sim data
- `SimulateurTournoi(modele, params, elo, formes, progression, verbose)` — Simulator class
- `simuler_groupe(simulateur, equipes, group_name)` — Group stage sim
- `run_monte_carlo_sequential(simulateur, n_simulations)` — Sequential MC
- `run_monte_carlo_parallel(simulateur, n_simulations)` — Parallel MC
- `GROUPES_DEF` — Dict mapping group letters to team lists
- `HOTES` — Set of host nation team names
- `EXPERIENCE_CDMS` — Dict mapping team names to WC appearance count

---

## 📸 Usage Guide

### 1. Sidebar — Load & Configure

![Sidebar](docs/sidebar.png) *(Add your own screenshot)*

- **Load Data** — One-click model training
- **Simulation Slider** — Adjust Monte Carlo iterations
- **Toggles** — Enable/disable Dixon-Coles, momentum, parallel processing

### 2. Overview Tab

![Overview](docs/overview.png) *(Add your own screenshot)*

Displays:
- 🏆 Winner, 🥈 Finalist, 🥉 3rd Place cards with probabilities and confidence intervals
- Top 15 contender bar chart
- Top 4 probability area chart
- Confederation qualification pie chart

### 3. Groups Tab

![Groups](docs/groups.png) *(Add your own screenshot)*

- 12 groups displayed in a 4-column grid
- Each team shows Elo rating, World Cup experience, and host badge
- **"▶️ Simulate Group X"** button runs a single-group simulation

### 4. Knockout Tab

![Knockout](docs/knockout.png) *(Add your own screenshot)*

- Visual bracket from Round of 32 through to the Final
- Predicted final matchup displayed prominently
- Knockout probability stacked bar chart by team

### 5. Predictions Tab

![Predictions](docs/predictions.png) *(Add your own screenshot)*

- Dropdown to filter by: Winner, Finalist, Semi-Final, Top 4, Qualified, Group 1st
- Full probability table with progress bars
- **📥 Download CSV** button for exporting results

### 6. Model Tab

![Model](docs/model.png) *(Add your own screenshot)*

- Parameter cards with color-coded impact (positive/negative/high)
- Feature importance radar chart (Plotly)
- Mathematical formula breakdown with EloBonus, Momentum, Fatigue, and Dixon-Coles τ

### 7. Settings Tab

![Settings](docs/settings.png) *(Add your own screenshot)*

- Fine-tune simulation parameters (ρ, decay λ, random seed)
- Model toggles (Dixon-Coles, momentum, host advantage, fatigue)
- Export configuration as JSON

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Interactive web application framework |
| **Plotly** | Interactive charts and visualizations |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning model utilities |
| **SciPy** | Statistical distributions and optimization |
| **Multiprocessing** | Parallel Monte Carlo execution |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and test locally with `streamlit run app.py`
4. **Commit**: `git commit -m "Add: your feature description"`
5. **Push**: `git push origin feature/your-feature-name`
6. **Open a Pull Request** on GitHub

### Ideas for contributions
- Add more visualization types (heatmaps, Sankey diagrams)
- Implement head-to-head team comparison tool
- Add historical tournament comparison (2022 vs 2026 predictions)
- Deploy to Streamlit Cloud with GitHub Actions CI/CD

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 11Minato11

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Streamlit** — For making Python web apps incredibly simple
- **Plotly** — For beautiful, interactive visualizations
- **EloRatings.net** — For historical football team strength metrics
- **Dixon & Coles (1997)** — For the seminal low-score dependency model
- **FIFA** — For the beautiful game and the 2026 World Cup format

---

## 📬 Contact

For questions, suggestions, or collaboration:

- **GitHub Issues**: [github.com/11Minato11/World_cup_prediction/issues](https://github.com/11Minato11/World_cup_prediction/issues)
- **Email**: *(your email here)*

---

<p align="center">
  <strong>Built with ⚽ and 🐍 for the love of football.</strong><br>
  <em>May the best team win — probabilistically.</em>
</p>
