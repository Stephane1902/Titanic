# Titanic : quels facteurs ont influencé la survie des passagers ?

Projet de Data Science réalisé en groupe dans le cadre de la formation Data Engineer (brief « Titanic : nettoyer, explorer et prédire la survie des passagers avec le Machine Learning »).

## Contexte

Un musée maritime prépare une exposition interactive sur le naufrage du Titanic et pose une question :

> **Quels facteurs ont réellement influencé les chances de survie des passagers ?**

Notre mission : explorer les archives (dataset Titanic, Kaggle), les nettoyer, créer de nouvelles variables, puis entraîner et comparer des modèles de classification pour identifier ce qui a pesé dans la survie.

## Contenu du dépôt

```
.
├── Brief_Titanic.ipynb        # Notebook commenté : EDA, nettoyage, feature engineering, modélisation
├── data/
│   └── dataset-titanic-6afac2fa9ddd8043393546.csv   # Données brutes (891 passagers, 12 colonnes)
├── Presentation_Titanic.pptx  # Slides de la présentation orale
├── requirements.txt           # Dépendances Python
└── README.md
```

## Données

Le fichier contient 891 passagers et 12 colonnes (source : [Kaggle - Titanic](https://www.kaggle.com/c/titanic)).

| Colonne | Description |
|---|---|
| `Survived` | **Cible** : 1 = survivant, 0 = décédé |
| `Pclass` | Classe du billet (1, 2, 3) |
| `Name` | Nom du passager (contient le titre : Mr, Mrs, Miss, Master…) |
| `Sex` | Sexe |
| `Age` | Âge (177 valeurs manquantes) |
| `SibSp` | Nombre de frères, sœurs ou conjoint à bord |
| `Parch` | Nombre de parents ou enfants à bord |
| `Ticket` | Numéro de billet |
| `Fare` | Prix du billet |
| `Cabin` | Numéro de cabine (687 valeurs manquantes) |
| `Embarked` | Port d'embarquement (S, C, Q) |

## Installation et exécution

**Prérequis** : Python 3.10 ou supérieur.

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd <nom-du-depot>

# 2. (Recommandé) créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter et ouvrir le notebook
jupyter notebook Brief_Titanic.ipynb
```

Puis exécuter toutes les cellules dans l'ordre (**Kernel → Restart & Run All**).

> **Important :** dans la cellule de chargement des données, le chemin du CSV doit pointer vers `data/dataset-titanic-6afac2fa9ddd8043393546.csv`.
> Si vous utilisez Google Colab, déposez le fichier CSV dans `/content/` à la place.

Les résultats sont reproductibles : le découpage train/test utilise `random_state=42`.

## Méthodologie

### 1. Analyse exploratoire (EDA)

- Structure du dataset (`shape`, `info`, `describe`) et recherche de valeurs manquantes.
- Distribution de l'âge selon la survie, taux de survie par classe, par sexe, par taille de famille et par titre.
- Distribution du prix du billet selon la survie, matrice de corrélation des variables numériques.

### 2. Nettoyage et transformation

Chaque choix est justifié dans le notebook.

| Problème | Traitement | Justification |
|---|---|---|
| `Age` manquant (177) | Remplacé par la moyenne | Garder les 891 passagers ; moyenne et médiane quasi identiques |
| `Cabin` manquante (687, soit 77 %) | Colonne supprimée | Trop de valeurs manquantes et 147 valeurs différentes, donc inexploitable |
| `Embarked` manquant (2) | Remplacé par le mode (S) | Seulement 2 cas, impact négligeable |
| `PassengerId`, `Ticket`, `Name` | Supprimées | Aucun pouvoir prédictif (le titre est extrait de `Name` avant suppression) |
| `Sex`, `Embarked`, `Title` | Encodage one-hot (`get_dummies`, `drop_first=True`) | Les modèles n'acceptent que des variables numériques |
| Variables numériques | Standardisation (`StandardScaler`, ajusté sur le train uniquement) | Nécessaire pour la régression logistique et le KNN ; inutile pour la Random Forest |

Le dataset final ne contient que des variables numériques.

### 3. Feature engineering

| Nouvelle variable | Construction | Intérêt |
|---|---|---|
| `Title` | Extraite du nom par expression régulière | Résume sexe, âge et statut social |
| `Total_Family` | `SibSp + Parch + 1` | Voyager seul ou en famille change les chances de survie |
| `is_child` | 1 si `Age < 18` | Les enfants ont été prioritaires à l'évacuation |

### 4. Modélisation

Trois modèles issus de trois familles d'algorithmes, pour comparer des approches différentes :

- **Régression logistique** : modèle simple de référence, dont les coefficients indiquent le sens de chaque facteur.
- **K plus proches voisins (KNN, k=7)** : approche par proximité entre passagers.
- **Random Forest (profondeur max 4)** : ensemble d'arbres de décision, capable de capter des combinaisons de variables.

Découpage 80 % entraînement / 20 % test (`random_state=42`), comparaison de la précision (accuracy) sur le jeu de test.

## Résultats

| Modèle | Score d'entraînement | Score de test |
|---|---|---|
| Régression logistique | 0.84 | 0.80 |
| K plus proches voisins | 0.85 | **0.83** |
| Random Forest | 0.84 | 0.81 |

Les trois modèles prédisent correctement environ 4 survivants sur 5. Les scores d'entraînement et de test sont proches, ce qui indique l'absence de surapprentissage marqué. L'écart entre les modèles est faible : quelques variables simples expliquent déjà l'essentiel de la survie.

## Conclusions pour le musée

D'après l'analyse exploratoire, les facteurs qui ressortent le plus sont :

1. **Le sexe** : les femmes ont survécu bien plus souvent que les hommes.
2. **La classe** : le taux de survie passe d'environ 63 % en 1re classe à environ 24 % en 3e classe.
3. **L'âge** : les jeunes enfants ont été favorisés, les passagers de plus de 60 ans rarement sauvés.

Le prix du billet (très lié à la classe) et la taille de la famille jouent aussi un rôle.

> Ces résultats montrent des **associations** avec la survie, pas des relations de cause à effet.

## Limites et pistes d'amélioration

- Évaluer l'apport de chaque nouvelle variable en comparant les scores avec et sans elle.
- Mesurer l'importance des variables dans les modèles (`feature_importances_`, coefficients, permutation importance).
- Regrouper les titres rares (Dr, Rev, Col, Major, etc.) et tester une extraction de la lettre de pont à partir de `Cabin`.
- Imputer l'âge par la médiane selon le titre plutôt que par la moyenne globale.
- Utiliser une validation croisée et d'autres métriques (précision, rappel, F1) pour une comparaison plus robuste.
- Tester d'autres modèles (XGBoost, Gradient Boosting) et optimiser les hyperparamètres.

## Outils

Python · Jupyter Notebook · pandas · NumPy · matplotlib · seaborn · scikit-learn · Git / GitHub
