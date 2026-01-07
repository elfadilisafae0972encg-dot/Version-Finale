# Rapport d'Analyse et de Modélisation Financière
## Introduction
### Contexte
Ce projet vise à analyser un ensemble de données financières provenant du New York Stock Exchange (NYSE), contenant des informations détaillées sur diverses entreprises sur plusieurs années. Le dataset comprend des indicateurs comptables, des ratios financiers, des données de performance, et des identifiants boursiers.

### Problématique
La prédiction des performances financières, et en particulier le bénéfice par action (Earnings Per Share - EPS), est un défi majeur pour les analystes financiers et les investisseurs. Un modèle prédictif fiable de l'EPS peut fournir des informations précieuses pour les décisions d'investissement et l'évaluation des entreprises. Le défi réside dans la grande dimensionnalité des données, la présence de valeurs manquantes, de valeurs aberrantes, et la nécessité de gérer des variables catégorielles et temporelles.

### Objectifs Les objectifs principaux de cette analyse sont les suivants :

Nettoyage et Préparation des Données : Effectuer un nettoyage et une préparation rigoureuse pour garantir la qualité, la cohérence et l'exploitabilité des données.
Exploration des Données (EDA) : Mener une analyse exploratoire approfondie pour comprendre les distributions des variables, identifier les relations entre elles, et détecter d'éventuels motifs ou anomalies.
Ingénierie des Caractéristiques (Feature Engineering) : Créer de nouvelles variables pertinentes à partir des données existantes pour enrichir le dataset et potentiellement améliorer la performance des modèles.
Développement et Évaluation de Modèles Prédictifs : Entraîner et évaluer plusieurs modèles de régression pour prédire le 'Earnings Per Share' (EPS), en comparant leurs performances.
Optimisation des Hyperparamètres et Validation Robuste : Utiliser des techniques de validation croisée et d'optimisation des hyperparamètres pour affiner le modèle le plus performant et obtenir une estimation plus fiable de sa capacité de généralisation.
## Méthodologie
Pour atteindre les objectifs fixés, une série d'étapes techniques rigoureuses a été mise en œuvre, chaque choix étant justifié par des considérations de qualité des données et de performance des modèles d'apprentissage automatique.

### 1. Nettoyage et Préparation des Données
Chargement des Données : Le fichier 'New York Stock Exchange-data.csv' a été chargé dans un DataFrame pandas. Une inspection initiale (head, info, describe) a révélé la structure du dataset, les types de données et la présence de colonnes superflues ou de valeurs manquantes.

Suppression de Colonnes Redondantes : La colonne 'Unnamed: 0' a été supprimée. Cette colonne était un index importé du fichier CSV et n'apportait aucune information utile pour l'analyse, sa suppression réduisant la complexité du dataset sans perte d'information.

Conversion des Types de Données :

La colonne 'Period Ending' a été convertie en type datetime. Cette conversion est essentielle pour permettre l'extraction de caractéristiques temporelles (année, mois, jour, trimestre) et faciliter l'analyse des tendances sur le temps.
La colonne 'For Year' a été examinée. Des valeurs manifestement erronées (années antérieures à 1900) ont été remplacées par NaN pour éviter toute distorsion statistique. Ensuite, la colonne a été convertie en type entier nullable (Int64), ce qui permet de conserver les NaN tout en représentant correctement les années fiscales.
Gestion des Doublons : Le dataset a été vérifié pour la présence de lignes dupliquées. Aucune duplication n'a été détectée, assurant ainsi que chaque observation est unique et représentative.

Imputation des Valeurs Manquantes : Pour plusieurs colonnes numériques ('Cash Ratio', 'Current Ratio', 'For Year', 'Earnings Per Share', 'Estimated Shares Outstanding', 'Quick Ratio'), les valeurs manquantes ont été imputées par la médiane. Le choix de la médiane est justifié par sa robustesse aux valeurs aberrantes, qui sont fréquentes dans les données financières. L'utilisation de la médiane permet de maintenir la distribution des données et d'éviter d'introduire des biais significatifs par rapport à la moyenne, qui serait plus sensible aux extrêmes.

Encodage des Variables Catégorielles : La colonne 'Ticker Symbol', la seule variable catégorielle nominale, a été traitée avec le One-Hot Encoding à l'aide de pd.get_dummies(). Cette technique crée des variables binaires pour chaque catégorie unique du symbole boursier, évitant ainsi d'imposer un ordre artificiel entre les catégories et permettant aux modèles d'apprentissage automatique de les interpréter correctement.

Mise à l'échelle des Données Numériques : La Standardisation (Z-score scaling), implémentée via StandardScaler de sklearn.preprocessing, a été appliquée à toutes les colonnes numériques (à l'exception de 'Period Ending' et des colonnes booléennes résultant du one-hot encoding). Cette méthode transforme les données pour qu'elles aient une moyenne de zéro et un écart-type de un. La standardisation est cruciale pour de nombreux algorithmes d'apprentissage automatique qui sont sensibles à l'échelle des caractéristiques (par exemple, la régression linéaire, les méthodes basées sur les distances, et les modèles avec régularisation L1/L2), garantissant une convergence plus rapide et de meilleures performances.

### 2. Ingénierie des Caractéristiques (Feature Engineering)
À partir de la colonne Period Ending, des caractéristiques temporelles supplémentaires ont été extraites pour enrichir le modèle avec des informations sur la temporalité des données financières :

Year (Année)
Month (Mois)
Day (Jour)
Quarter (Trimestre)
Is_End_Of_Year (Indicateur booléen si la date est le 31 décembre), utile pour capturer les effets de clôture d'exercice.
Ces caractéristiques permettent aux modèles de capter des motifs saisonniers ou cycliques qui pourraient influencer l'EPS, tels que des tendances de fin d'année ou des variations trimestrielles des performances des entreprises.

### 3. Exploration des Données (EDA)
L'EDA a été menée pour visualiser la distribution des principales caractéristiques numériques et leurs relations. Bien que les graphiques ne puissent pas être affichés directement ici, voici les observations clés:

Distributions des Caractéristiques Numériques (Histograms): Des histogrammes ont été générés pour des colonnes clés telles que 'Accounts Payable', 'After Tax ROE', 'Earnings Per Share', et 'Total Revenue'. Ces visualisations ont révélé des distributions variées, dont certaines étaient asymétriques (skewed) ou présentaient des pics, indiquant une concentration de valeurs dans certaines plages. La mise à l'échelle (StandardScaler) a permis de normaliser ces distributions en termes de moyenne et d'écart-type, mais les formes intrinsèques (asymétrie, kurtosis) des distributions originales demeurent.

Exemple d'Histograms:

### Exemple de code pour générer les histogrammes
import matplotlib.pyplot as plt
import seaborn as sns

selected_numerical_cols = ['Accounts Payable', 'After Tax ROE', 'Earnings Per Share', 'Total Revenue']

plt.figure(figsize=(15, 10))
for i, col in enumerate(selected_numerical_cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
Détection des Outliers (Boxplots): Des boxplots ont été utilisés pour les mêmes caractéristiques numériques afin d'identifier la présence de valeurs aberrantes. Comme souvent dans les données financières, de nombreux outliers ont été observés, en particulier pour des variables comme 'Accounts Payable' ou 'Total Revenue'. Ces outliers peuvent indiquer des événements exceptionnels pour certaines entreprises ou simplement des entreprises de tailles très différentes. La médiane, utilisée pour l'imputation des valeurs manquantes, est moins affectée par ces valeurs extrêmes.

Exemple de Boxplots:

### Exemple de code pour générer les boxplots
import matplotlib.pyplot as plt
import seaborn as sns

selected_numerical_cols = ['Accounts Payable', 'After Tax ROE', 'Earnings Per Share', 'Total Revenue']

plt.figure(figsize=(15, 10))
for i, col in enumerate(selected_numerical_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()
Analyse de Corrélation (Heatmap): Une carte de chaleur (heatmap) de la matrice de corrélation a été générée pour visualiser les relations linéaires entre les caractéristiques numériques. Cette analyse a révélé des corrélations variées : certaines paires de variables étaient fortement corrélées (positivement ou négativement), ce qui est attendu dans les données financières où de nombreux indicateurs sont interdépendants. D'autres paires montraient des corrélations faibles ou nulles.

Exemple de Heatmap de Corrélation:

### Exemple de code pour générer la heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

numerical_features_for_corr = df.select_dtypes(include=['float64', 'Int64']).columns.tolist()
numerical_features_for_corr = [col for col in numerical_features_for_corr if not col.startswith('Ticker_')]
if 'Period Ending' in numerical_features_for_corr:
    numerical_features_for_corr.remove('Period Ending')

correlation_matrix = df[numerical_features_for_corr].corr()

mask = np.triu(correlation_matrix)
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix,
            mask=mask,
            annot=False,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()
Cette phase d'EDA a été cruciale pour confirmer la nécessité des étapes de nettoyage et de préparation, et pour orienter les choix de modélisation en comprenant mieux la nature des données.

### 4. Développement et Évaluation des Modèles Prédictifs
Sélection de la Variable Cible et Séparation des Données
Variable Cible : 'Earnings Per Share' (EPS) a été choisi comme variable cible (y) pour la prédiction, en raison de son importance fondamentale en finance pour évaluer la rentabilité d'une entreprise.
Variables Explicatives : Toutes les autres colonnes, après nettoyage, encodage et ingénierie de caractéristiques, ont été utilisées comme variables explicatives (X). La colonne 'Period Ending' a été explicitement exclue de X car elle est de type datetime et ses informations ont été extraites dans d'autres caractéristiques numériques.
Séparation des Données : Les données ont été divisées en ensembles d'entraînement (80%) et de test (20%) en utilisant train_test_split avec un random_state=42 pour assurer la reproductibilité des résultats.
X_train shape: (1424, 528)
X_test shape: (357, 528)
y_train shape: (1424,)
y_test shape: (357,)
## Modèles de Régression Entraînés et Évalués
Trois modèles de régression ont été entraînés sur l'ensemble d'entraînement et évalués sur l'ensemble de test, en utilisant les métriques suivantes :

MAE (Mean Absolute Error) : Mesure la moyenne des erreurs absolues entre les prédictions et les vraies valeurs. Moins sensible aux outliers que le MSE.
MSE (Mean Squared Error) : Mesure la moyenne des carrés des erreurs. Pénalise davantage les grandes erreurs et est sensible aux outliers.
R-squared (Coefficient de Détermination) : Indique la proportion de la variance de la variable dépendante qui est prévisible à partir des variables indépendantes. Un R2 proche de 1 indique un bon ajustement du modèle.
Régression Linéaire (Linear Regression)

Justification : Un modèle de base, simple et interprétable, servant de référence pour évaluer la complexité et les performances d'autres modèles plus avancés.
Résultats :
Mean Absolute Error (MAE): 0.3507
Mean Squared Error (MSE): 0.5748
R-squared (R2) Score: 0.5083
Random Forest Regressor

Justification : Un modèle d'ensemble puissant, robuste aux outliers, capable de capturer des relations non linéaires et de gérer un grand nombre de caractéristiques. Il réduit le surapprentissage par l'agrégation de multiples arbres de décision.
Résultats :
Mean Absolute Error (MAE): 0.1500
Mean Squared Error (MSE): 0.2823
R-squared (R2) Score: 0.7585
Gradient Boosting Regressor

Justification : Autre modèle d'ensemble très performant, qui construit des arbres de décision de manière séquentielle, chaque nouvel arbre corrigeant les erreurs des précédents. Il est réputé pour sa capacité à obtenir une haute précision.
Résultats :
Mean Absolute Error (MAE): 0.2053
Mean Squared Error (MSE): 0.2746
R-squared (R2) Score: 0.7651
5. Validation Croisée pour une Évaluation Robuste (Gradient Boosting Regressor)
Pour obtenir une estimation plus fiable de la performance du Gradient Boosting Regressor et évaluer sa capacité de généralisation, une validation croisée K-Fold a été appliquée (KFold(n_splits=5, shuffle=True, random_state=42)).

Scores R-squared par pli:

Fold 1: 0.7652
Fold 2: 0.8356
Fold 3: 0.7259
Fold 4: 0.8480
Fold 5: 0.4880
Statistiques agrégées:

Mean R-squared across all folds: 0.7325
Standard deviation of R-squared across all folds: 0.1303
La moyenne des scores R-squared (0.7325) est proche de celle obtenue sur le set de test initial, mais l'écart-type élevé (0.1303) indique une certaine variabilité. Le score plus faible du pli 5 (0.4880) suggère que le modèle peut être moins performant sur certains sous-ensembles de données, potentiellement en raison de la distribution des données ou de la présence de cas plus difficiles à prédire dans ce pli spécifique.

## Résultats & Discussion
Comparaison des Performances des Modèles
Le tableau suivant récapitule les performances des trois modèles sur l'ensemble de test :

Model	MAE	MSE	R2 Score
Linear Regression	0.35066	0.574765	0.508322
Random Forest Regressor	0.150015	0.282331	0.758482
Gradient Boosting Regressor	0.205294	0.274554	0.765135
Analyse :

La Régression Linéaire sert de ligne de base, expliquant environ 50.83% de la variance de l'EPS, ce qui est modéré et indique que les relations ne sont pas purement linéaires.
Les modèles d'ensemble, Random Forest Regressor et Gradient Boosting Regressor, surclassent largement la régression linéaire, démontrant la capacité de ces algorithmes à capturer des relations plus complexes dans les données.
Le Random Forest Regressor présente le MAE le plus bas (0.1500), suggérant qu'il réalise en moyenne les prédictions les plus proches des vraies valeurs.
Le Gradient Boosting Regressor obtient le meilleur R-squared (0.7651) et le MSE le plus bas (0.2746). Un R2 de 0.7651 signifie que le modèle explique plus de 76% de la variance de l'EPS, ce qui est un résultat très satisfaisant pour la prédiction de données financières.
Globalement, le Gradient Boosting Regressor est identifié comme le modèle le plus performant sur cet ensemble de données, offrant le meilleur équilibre entre la minimisation de l'erreur quadratique moyenne et la maximisation de la variance expliquée.

Analyse des Erreurs du Modèle (Exemple pour le Gradient Boosting Regressor)
Bien que le R2 soit élevé, il est essentiel d'examiner les erreurs pour comprendre les limites du modèle.

Visualisation des Résidus : Un graphique des résidus (différence entre les prédictions et les vraies valeurs) par rapport aux valeurs prédites ou réelles pourrait révéler des motifs (hétéroscédasticité, biais) indiquant des problèmes dans le modèle. Idéalement, les résidus devraient être répartis aléatoirement autour de zéro.

### Exemple de code pour visualiser les résidus
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_gbr, y_test - y_pred_gbr, alpha=0.5)
plt.hlines(y=0, xmin=min(y_pred_gbr), xmax=max(y_pred_gbr), colors='red', linestyles='--')
plt.xlabel('Valeurs Prédites (EPS)')
plt.ylabel('Résidus (Vraie - Prédite)')
plt.title('Analyse des Résidus du Gradient Boosting Regressor')
plt.show()
Distribution des Erreurs : L'histogramme des résidus permettrait de voir si les erreurs sont normalement distribuées autour de zéro. Une distribution non normale ou fortement asymétrique pourrait indiquer que le modèle ne capture pas toutes les relations ou qu'il est influencé par des outliers.

### Exemple de code pour la distribution des erreurs
plt.figure(figsize=(8, 6))
sns.histplot(y_test - y_pred_gbr, kde=True)
plt.title('Distribution des Erreurs (Résidus) du Gradient Boosting Regressor')
plt.xlabel('Erreurs de Prédiction')
plt.ylabel('Fréquence')
plt.show()
Matrice de Confusion (non applicable directement pour la régression, mais pour la classification) : Si la tâche avait été une classification (par exemple, prédire si l'EPS sera positif ou négatif), une matrice de confusion serait utilisée. Pour une régression, des analyses d'erreurs plus spécifiques impliqueraient l'examen des observations où les erreurs sont les plus grandes pour comprendre pourquoi le modèle a eu des difficultés.

## Conclusion
### Synthèse des Performances
Cette analyse a démontré l'efficacité des modèles d'ensemble pour la prédiction de l'Earnings Per Share (EPS) sur les données financières du NYSE. Le Gradient Boosting Regressor s'est distingué comme le modèle le plus performant, atteignant un R-squared de 0.7651 et un MSE de 0.2746 sur l'ensemble de test, expliquant ainsi une part significative de la variance de l'EPS. Le Random Forest Regressor a également montré d'excellentes performances, particulièrement en termes de MAE.

La validation croisée a confirmé la robustesse générale du Gradient Boosting Regressor, avec une moyenne de R-squared de 0.7325 sur 5 plis, bien qu'une variabilité notable (écart-type de 0.1303) ait été observée, suggérant une sensibilité à la composition des sous-ensembles de données.

### Limites du Modèle Actuel
Malgré les bonnes performances, plusieurs limites peuvent être identifiées :

Variabilité de la Performance : L'écart-type élevé des scores de validation croisée indique que le modèle, bien que globalement bon, peut être moins stable sur certains segments de données. Ceci pourrait être dû à des données d'entreprises très spécifiques ou à des périodes particulières.
Interprétabilité : Les modèles d'ensemble sont des 'boîtes noires' comparés à la régression linéaire. Bien qu'ils soient performants, il est plus difficile d'expliquer directement et intuitivement la contribution de chaque caractéristique à une prédiction spécifique. Des techniques comme l'importance des caractéristiques ou les valeurs SHAP pourraient améliorer l'interprétabilité mais n'ont pas été explorées ici.
Traitement des Outliers : Bien que la standardisation et l'imputation par la médiane aient été utilisées, la persistance d'outliers (observée lors de l'EDA) pourrait toujours influencer les prédictions et la stabilité du modèle.
Hyperparamètres non optimisés : Les modèles initiaux ont été entraînés avec des hyperparamètres par défaut. Bien qu'une grille pour GridSearchCV ait été définie, l'étape d'optimisation n'a pas encore été finalisée pour le modèle GBR, ce qui pourrait potentiellement améliorer davantage les performances.
Nature Temporelle des Données : Bien que des caractéristiques temporelles aient été extraites, l'approche n'a pas explicitement utilisé des modèles de séries temporelles qui pourraient mieux capturer les dépendances temporelles et les tendances sous-jacentes pour des prédictions futures.
### Pistes d'Amélioration
Pour la suite du projet, les pistes d'amélioration suivantes sont envisagées :

Optimisation des Hyperparamètres : L'étape la plus immédiate est d'exécuter GridSearchCV avec la grille de paramètres définie pour le Gradient Boosting Regressor. Cela devrait permettre de trouver la combinaison optimale d'hyperparamètres pour maximiser la performance et la généralisation du modèle.
Gestion Avancée des Outliers : Explorer des techniques de traitement des valeurs aberrantes plus sophistiquées, telles que la Winsorisation, les transformations logarithmiques, ou des modèles robustes, pour améliorer la stabilité du modèle.
Sélection et Ingénierie de Caractéristiques Avancées : Effectuer une analyse approfondie de l'importance des caractéristiques et envisager des techniques de réduction de dimensionnalité (ex: PCA) ou la création de ratios financiers supplémentaires qui pourraient être plus prédictifs.
Exploration d'Autres Modèles : Tester d'autres algorithmes d'apprentissage automatique de pointe, comme XGBoost ou LightGBM, qui sont des variantes optimisées du gradient boosting et qui pourraient offrir de meilleures performances ou une meilleure efficacité computationnelle.
Approches Temporelles : Si l'objectif est de prédire l'EPS pour de futures périodes, l'intégration de techniques de séries temporelles (ex: LSTM pour les données de panel, ou modèles ARIMAX) pourrait être pertinente, en considérant les données comme des séries temporelles par entreprise.
Interprétabilité du Modèle : Utiliser des outils d'interprétabilité tels que SHAP (SHapley Additive exPlanations) pour comprendre les contributions des caractéristiques aux prédictions, ce qui est crucial dans un contexte financier.
En abordant ces pistes d'amélioration, il serait possible de construire un modèle encore plus robuste, précis et interprétable pour la prédiction de l'Earnings Per Share.
