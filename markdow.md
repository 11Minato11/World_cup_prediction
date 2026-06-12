# 🏆 Architecture du Simulateur de la Coupe du Monde 2026

Ce document détaille la structure logique du projet de prédiction de la Coupe du Monde 2026. Le projet passe d'une analyse statistique historique (Régression) à une simulation probabiliste d'univers parallèles (Monte Carlo).

---

## 📊 1. Les Fondations : Le Modèle de Poisson (Terminé)

Le rôle du modèle n'est pas de dire "qui va gagner", mais de calculer l'espérance mathématique de buts pour un match précis, en se basant sur le passé.

* **Variables retenues :** * `diff_elo` : La différence de force intrinsèque ($Elo_{Attaque} - Elo_{Defense}$).
    * `is_host` : L'avantage d'évoluer à domicile (ou sur son continent pour les pays hôtes).
* **L'Output du Modèle ($\lambda$) :**
    Grâce aux coefficients ($\beta$) appris par l'algorithme, on obtient $\lambda$, qui représente le nombre moyen de buts qu'une équipe est censée marquer contre son adversaire.
    $$\lambda = \exp(\beta_0 + \beta_1 \cdot \text{diff\_elo} + \beta_2 \cdot \text{is\_host})$$

---

## 🎲 2. Le Moteur Mathématique : Monte Carlo

Le football est régi par le hasard. Une équipe plus faible peut battre une équipe plus forte sur un seul match. Pour obtenir de vraies probabilités, nous allons créer **10 000 univers parallèles** dans lesquels la Coupe du Monde se joue du début à la fin.

### La fonction de match (Le cœur du moteur)
Pour chaque match, on utilise la loi de Poisson pour transformer le $\lambda$ en un score entier aléatoire.
```python
import numpy as np
buts_A = np.random.poisson(lambda_A)
buts_B = np.random.poisson(lambda_B)