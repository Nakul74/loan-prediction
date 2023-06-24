<h1>Projet 7: Implémentez un modèle de scoring</h1>

<h2>Contexte et problématique du projet</h2>

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser",  qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

<h2>But et intérêt du projet</h2>

Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique. Données disponible ici : https://www.kaggle.com/c/home-credit-default-risk/data

- Possibilité de s'appuyer sur un kernel Kaggle pour faciliter l’analyse exploratoire, la préparation des données et le feature engineering nécessaires à l’élaboration du modèle de scoring
- Utiliser des pipelines d'entrainement
- Élaborer et choisir parmi plusieurs modèle de classification celui qui aura les meilleurs résultats avec une technique d'optimisation des hyper-paramètres
- Formaliser des mesures et résultats de chaque expérimentation, afin de les analyser et de les comparer sur ML Flow
- Mettre en place une démarche de type MLOps d’automatisation et d’industrialisation de la gestion du cycle de vie du modèle. Liste des outils ici : https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P7/Outils+Open+Source+MLOps.pdf
- Créer une métrique personnalisée sous la forme d'une fonction coût répondant aux problématiques métier
- Utiliser un seuil de classification pour l'octroi d'un crédit
- Analyser la feature importance locale et globale avec SHAP
- Détecter éventuellement du Data Drift sur les principales features, entre les datas d’entraînement et les datas de production, au travers d'un tableau HTML d’analyse avec la librairie Evidently
- Rédiger une note méthodologique décrivant la méthodologie d'entraînement du modèle, le traitement du déséquilibre des classes, la fonction coût métier, l'algorithme d'optimisation et la métrique d'évaluation , un tableau de synthèse des résultats, l’interprétabilité globale et locale du modèle,les limites et les améliorations possibles, l’analyse du Data Drift


Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

- Visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
- Visualiser des informations descriptives relatives à un client (via un système de filtre).
- Comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

- Utiliser un logiciel de version de code pour assurer l’intégration du modèle

<h2>Compétences évaluées</h2>

- Réaliser un dashboard pour présenter son travail de modélisation
- Présenter son travail de modélisation à l'oral
- Évaluer les performances des modèles d’apprentissage supervisé selon différents critères
- Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- Définir et mettre en œuvre un pipeline d’entraînement des modèles
- Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé
- Déployer un modèle via une API dans le Web
- Définir et mettre en œuvre une stratégie de suivi de la performance d’un modèle
- Rédiger une note méthodologique afin de communiquer sa démarche de modélisation

<h2>Contenu du dépôt GitHub</h2>

- README.md: fichier de présentation du projet

- Répertoire Delaguillaumie_Alexandre_1_dashboard_et_API_022023:
  - Répertoire "fastapi":
    - fichier "main.py": code de l'API. peut retourner une prediction ou des shap_values d'un client
    - fichier "model.pkl": modèle de classification exporté via mlflow.sklearn.log_model
    - fichier "Procfile": script pour déployer le code sur Heroku
    - fichier "README.md": Explique le fonctionnement de l'API
    - fichier "requirements.txt": Contient toutes les librairies et dépendances nécéssaires pour faire tourner le code
    - fichier "sample_test_set.pickle": dataset sous format DataFrame de la librairie pandas
  - Répertoire "Streamlit":
    - fichier "Dashboard.py":
    - fichier "infos_client.pickle":
    - fichier "model.pkl": modèle de classification exporté via mlflow.sklearn.log_model
    - fichier "preprocessed_data.pickle": model pré-entrainé pour récupérer le nom des colonnes après traitement
    - fichier "pret_client.pickle":
    - fichier "Procfile": script pour déployer le code sur Heroku
    - fichier "README.md": Explique le fonctionnement de l'API
    - fichier "requirements.txt": Contient toutes les librairies et dépendances nécéssaires pour faire tourner le code
    - fichier "runtime.txt" : donne la bonne version de Python compatible avec Heroku et les librairies
    - fichier "sample_test_set.pickle": dataset sous format DataFrame de la librairie pandas
    - fichier "setup.sh": script de configuration de l'interface Streamlit

- Répertoire Delaguillaumie_Alexandre_1_dashboard_et_API_022023:
  - Répertoire "Data Drift":
    - fichier "data_drift.html": rapport d'analyse du datadrift au format html
    - répertoire "metrics": comprend l'intégralité des résultats d'analyse du Data Drift pour chaque colonne sus-nommée au format lisible sur MLFlow
    - Notebook "Data Drift": Contient l'intégralité du code nécéssaire à l'élaboration du rapport d'analyse
  - Notebook "modelisation": L'ensemble du travail de merge des tables, gestion des valeurs abérrantes, pre-processing, tests d'algorithmes, choix et enregistrement des paramètres via MLFlow, interprétabilité et exports

- fichier "Delaguillaumie_Alexandre_3_note_méthodologique_022023.pdf": note méthodologique du projet
- fichier "Delaguillaumie_Alexandre_4_presentation_022023.pdf": présentation du projet