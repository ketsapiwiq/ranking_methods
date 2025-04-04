<!--
SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>

SPDX-License-Identifier: MIT
-->

# rank-comparia



## Description

**TODO[squelette]: Remplacer cette section par un paragraphe de description du projet.**

Calcule des rangs pour le projet comparia

## Installation

**TODO[squelette]: Relire et compléter les étapes d'installation.**

```bash
# Récupération du code avec Git
git clone ${GITLAB_URL}
cd rank-comparia



# Installation des dépendances et du projet via poetry
# (poetry se chargera également de créer un environnement virtuel pour vous,
#  par défaut il sera dans le cache de poetry mais vous pouvez forcer poetry
#  à l'installer à la racine du dossier avec `POETRY_VIRTUALENVS_IN_PROJECT=1`)
poetry install
```


## Utilisation

**TODO[squelette]: Décrire comment utiliser le code ("_getting started_").**



## Contribution


Avant de contribuer au dépôt, il est nécessaire d'initialiser les _hooks_ de _pre-commit_ :

```bash
poetry run pre-commit install
```



<!--
***** TODO[squelette] ****
Décommenter les lignes suivantes et supprimer ce bloc si vous utilisez la publication
automatique via les jobs `package-publish-project` ou `package-publish-central`
du `.gitlab-ci.yml`.
Le job package-publish-central nécessite que la variable `CENTRAL_REGISTRY_ID`
soit configurée avec l'ID du dépôt central (52).
Cette variable est déjà configurée pour tous les projets au sein du groupe PEReN.
**************************

## Deployment

La bibliothèque est automatiquement publié dans les dépôts de paquets lors de la publication d'un tag de version.
Pour être reconnu le tag doit impérativement commencer par le caractère `v`,
puis être un numéro de version valide, par exemple `v1.2.4`.
Le dépôt doit également comporter un fichier `CHANGELOG.md`,
possédant une section formaté comme suit `## v<version> (<date du commit tagué[YYYY-MM-DD]>)`,
et décrivant les changements associés à la nouvelle version.  
Exemple:
```markdown
## v1.2.4 (2024-08-31)
### Features
- PDFs support
### Bug fixes
- Fix a memory leak
```
-->

## Licence

Ce projet est sous licence MIT. Une copie intégrale du texte
de la licence se trouve dans le fichier [`LICENSES/MIT.txt`](LICENSES/MIT.txt).
