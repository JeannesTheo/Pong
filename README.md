# VFA with ANN for solving Pong game

### Par : Sami El Kateb - Theo Jeannes

Notre fichier [td3.py](./td3.py) doit être lancé avec au moins un argument :
+ `play` si l'on souhaite ajouter des données
+ `watch` si l'on souhaite regarder l'agent jouer
D'autres arguments sont disponibles :
+ `--debug` : permet d'enregister les informations et les captures du jeu dans un dossier séparé
+ `--agent` : permet de choisir l'agent utilisé (`neural_net` ou `random_forest`)

Par exemple, pour lancer la capture de données dans le dossier `debug`, il est possible de faire :
```bash
python3 td3.py play --debug
```
Une fois un modèle entrainé, il suffira de faire :
```bash
python3 td3.py watch --agent neural_net
```
ou
```bash
python3 td3.py watch --agent random_forest
```
pour lancer le jeu avec l'agent choisi.

#### Network topology: plot the accuracy when varying the topology (number of layers, type of layers, number of neurons per layer)

Nous utilisons un fine-tuner pour trouver la meilleure configuration pour notre modèle. Tout les modèles testés sont
ensuite comparés, pour conserver le modèle qui donne les meilleurs résultats. Augmenter le nombre d'essais du fine-tuner
permet de tester plus de combinaisons, ce qui peut permettre d'obtenir de meilleurs résultats.
La cellule 13 du fichier [pong.ipynb](./pong.ipynb) permet d'afficher la précision et la loss des modèles testés.

#### What is the minimal size of data to allow efficient learning and generalisation?

Le jeu de données représente 14 000 images à la disposition du modèle. 25% de ces images sont réservées pour le test, ce
qui veut dire que le modèle apprend sur 10 500 images.

Malgré le nombre d'images, le modèle n'arrive pas à apprendre correctement. En effet, le modèle n'arrive pas à apprendre
à jouer suffisamment bien pour battre l'agent ennemi. Un plus grand nombre de données pourrait contribuer à un meilleur
modèle.

#### Try and use another estimator (use sklearn)

À partir de la cellule 17 du fichier [pong.ipynb](./pong.ipynb), nous utilisons un modèle de
type `RandomForestClassifier`, ce qui donne des résultats similaires au modèle précédent.
