---
title: PCB Defect Detection
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
suggested_hardware: cpu-basic
short_description: Inspect PCB defects with YOLO11 — français / English
pinned: false
tags:
  - computer-vision
  - object-detection
  - yolo
  - pcb
---

# Détection de défauts PCB · PCB Defect Detection

Détection de six types de défauts sur des circuits imprimés avec YOLO11. Le projet propose une interface web Gradio pour Hugging Face Spaces, une interface de bureau Tk et une commande pour entraîner, inspecter ou exporter un modèle.

**Les poids entraînés ne sont pas fournis.** Pour inspecter une image, utilisez votre checkpoint PCB `.pt`. Un modèle YOLO générique COCO est refusé. L'interface web peut démarrer avant que les poids soient configurés.

**English:** YOLO11 PCB defect detection with web, desktop and command-line interfaces. Trained weights are not included. Supply your own six-class PCB checkpoint. See the [English instructions](#english-instructions).

## Installation CPU et démarrage

Python **3.11 à 3.13** ; Hugging Face Spaces utilise **3.11**.

```bash
git clone https://github.com/alainpaluku/pcb-defect-detection.git
cd pcb-defect-detection
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Ouvrez `http://localhost:7860`. Sous Windows, activez l'environnement avec `.venv\Scripts\activate`.

Il existe **un seul fichier `requirements.txt`**, à la racine. Il sélectionne PyTorch CPU sous Linux et installe l'option web du projet. Les dépendances communes et les options sont déclarées dans **`pyproject.toml`** ; aucun autre fichier `requirements` n'est nécessaire. L'installation est éditable : les modifications du code sont immédiatement utilisées.

Après l'installation CPU ci-dessus, ajoutez uniquement les options souhaitées :

| Usage | Commande |
|---|---|
| Interface de bureau | `python -m pip install -e ".[gui]"` |
| Tests et vérifications du projet complet | `python -m pip install -e ".[gui,dev]"` |
| Export ONNX | `python -m pip install -e ".[onnx]"` |

Pour un environnement GPU, suivez la [section entraînement](#entraînement-local-et-kaggle) : le fichier CPU remplacerait une installation GPU de PyTorch sous Linux.

## Configurer les poids

Deux possibilités pour le service web :

- Créez `models/` et placez votre checkpoint dans `models/pcb_model.pt`.
- Configurez `PCB_MODEL_REPO` avec votre dépôt de modèles Hugging Face, par exemple `votre-compte/pcb-defect-model`.

`PCB_MODEL_PATH` permet de choisir un autre fichier local. Lorsqu'il est défini, il est prioritaire. Sinon, le service utilise `PCB_MODEL_REPO`, puis, si aucun dépôt n'est défini, `models/pcb_model.pt`. Une source explicite invalide déclenche une erreur. Seuls les poids provenant d'une source de confiance doivent être chargés.

Les six noms de classes doivent être présents dans le checkpoint. Leur ordre peut différer : le moteur traduit les résultats vers les identifiants suivants.

| ID | Classe | Français | English |
|---|---|---|---|
| 0 | `missing_hole` | Trou manquant | Missing drill hole |
| 1 | `mouse_bite` | Échancrure du cuivre | Copper edge notch |
| 2 | `open_circuit` | Piste interrompue | Broken trace |
| 3 | `short` | Court-circuit | Short circuit |
| 4 | `spur` | Excroissance de cuivre | Copper protrusion |
| 5 | `spurious_copper` | Cuivre parasite | Unwanted copper |

## Hugging Face Spaces

1. Créez un Space avec le SDK **Gradio**.
2. Envoyez les fichiers du projet, notamment `app.py`, `src/`, `gui_test/`, `pyproject.toml`, `requirements.txt`, `README.md` et `images/`. L'installation du paquet inclut `src` et `gui_test`, mais les dépendances graphiques restent facultatives. Gardez le bloc YAML en tête de ce README.
3. Dans **Settings → Variables and secrets**, définissez `PCB_MODEL_REPO` ou fournissez les poids locaux. Pour un dépôt privé, ajoutez `HF_TOKEN` comme **secret** de lecture.
4. Attendez la construction du Space puis lancez une inspection. Les poids Hub sont téléchargés et chargés à la première requête, puis réutilisés.

Le service traite une image à la fois et accepte jusqu'à 16 requêtes en attente. Il retourne une image annotée, un tableau et les résultats JSON. Les fichiers temporaires Gradio âgés d'au moins une heure sont nettoyés chaque heure. N'envoyez pas votre environnement `.venv/`, vos données d'entraînement ou vos secrets.

| Variable | Défaut | Fonction |
|---|---|---|
| `PCB_MODEL_PATH` | Non définie | Chemin local, relatif à la racine du projet ou absolu |
| `PCB_MODEL_REPO` | Non définie | Dépôt de modèles Hugging Face |
| `PCB_MODEL_FILENAME` | `pcb_model.pt` | Fichier à télécharger |
| `PCB_MODEL_REVISION` | `main` | Branche, tag ou commit du modèle |
| `HF_TOKEN` | Non défini | Secret d'accès aux poids privés |
| `PCB_DEVICE` | `cpu` | Périphérique d'inférence |
| `PCB_IMAGE_SIZE` | `640` | Taille d'inférence, multiple de 32 |
| `PCB_CPU_THREADS` | `2` | Threads PyTorch |
| `PCB_MAX_DETECTIONS` | `300` | Maximum de boîtes par image |
| `PCB_MAX_FILE_MB` | `20` | Limite de fichier en Mio |
| `PCB_MAX_PIXELS` | `20000000` | Limite de pixels avant décodage |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Adresse HTTP |
| `GRADIO_SERVER_PORT` | `7860` | Port HTTP |

`.env.example` décrit ces variables ; il n'est pas chargé automatiquement. La configuration fournie cible les Spaces CPU classiques. Le temps affiché mesure la prédiction et exclut le téléchargement des poids et le rendu.

Références : [dépendances Spaces](https://huggingface.co/docs/hub/spaces-dependencies), [configuration Spaces](https://huggingface.co/docs/hub/spaces-config-reference), [options de dépendances Python](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-and-requirements).

## Inspection et export

Exécutez les commandes depuis la racine du dépôt :

```bash
# FR : Inspecter une image. EN: Inspect one image.
python main.py detect images/01_short_01.jpg --model models/pcb_model.pt --conf 0.3 --iou 0.5

# FR : Inspecter un dossier récursivement. EN: Recursively inspect a folder.
python main.py detect images --model models/pcb_model.pt

# FR : Exporter vers ONNX. EN: Export to ONNX.
python -m pip install -e ".[onnx]"
python main.py export --model models/pcb_model.pt --format onnx --output output/exports
```

`--save` active l'enregistrement des images annotées ; l'inspection n'écrit pas d'image par défaut. Sans `--model`, la CLI cherche un checkpoint dans `models/pcb_model.pt`, `output/pcb_model.pt`, puis `pcb_model.pt`. Les variables `PCB_MODEL_*` concernent uniquement le service web.

`--format all` exporte ONNX et TorchScript. Les autres formats proposés par `--help` demandent leurs propres dépendances et plateformes ; l'option `onnx` ne les installe pas.

## Entraînement local et Kaggle

Utilisez un environnement séparé et installez d'abord le PyTorch adapté à votre GPU selon la [documentation officielle](https://pytorch.org/get-started/locally/). Installez ensuite uniquement le moteur :

```bash
python -m pip install -e .
python main.py train --data data/pcb-defects --epochs 50 --batch 16 --device 0
```

Sur Kaggle, ajoutez le dataset [PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects), activez Internet et un GPU, puis exécutez depuis le dépôt :

```python
!pip install -e .
!python main.py train --epochs 100 --device 0 --workers 2
```

Cette commande conserve le PyTorch GPU déjà installé s'il satisfait les dépendances. Pour deux GPU, utilisez `--device 0,1`. N'utilisez pas `requirements.txt` dans cet environnement GPU Linux.

Le pipeline attend des images et des annotations **VOC XML** portant le même nom sans extension. Les dossiers de classe permettent de départager les noms identiques :

```text
data/pcb-defects/
├── images/Short/01_short_01.jpg
└── Annotations/Short/01_short_01.xml
```

Les correspondances ambiguës, dimensions incorrectes, classes inconnues et images strictement identiques sont refusées. Les images sans XML ne sont pas utilisées ; un XML sans objet est un exemple négatif valide. Vérifiez la convention de coordonnées VOC de vos données.

Par défaut, les images partageant un préfixe numérique comme `01_` sont regroupées dans un même jeu pour limiter les fuites entre cartes. Il faut au moins deux groupes. Vérifiez cette convention sur un autre dataset ; `--split-mode image` autorise explicitement une séparation par image. Les options `--val-split` et `--seed` contrôlent le ratio et la graine.

Les sorties vont dans `output/`, ou `/kaggle/working/` sur Kaggle. Chaque entraînement reçoit son dossier `pcb_yolo*`. Le meilleur checkpoint de cette exécution est évalué, puis copié vers `pcb_model.pt` avec un `training_summary.json`. Ces deux fichiers représentent la dernière exécution. Le dataset converti contient un manifeste et seul un dossier généré portant ce marqueur peut être remplacé.

## Interface de bureau

```bash
python -m pip install -e ".[gui]"
python -m gui_test.app
```

Tkinter doit être disponible dans votre installation Python. L'interface utilise les poids `.pt`, les seuils de confiance et IoU, le zoom, le traitement récursif des dossiers et les exports JSON, CSV ou texte UTF-8. Les erreurs d'images sont distinguées des inspections sans détection.

## Structure et vérifications

```text
app.py              Interface web Gradio
main.py             Commandes d'inspection, d'entraînement et d'export
pyproject.toml      Dépendances communes, options et outils de qualité
requirements.txt    Installation web et choix CPU pour Linux / Spaces
src/                Configuration, moteur YOLO, données et entraînement
gui_test/           Interface de bureau facultative
images/             Six exemples utilisés par l'interface web
tests/              Tests de non-régression et d'intégration
.github/workflows/  Vérifications automatiques
```

```bash
python -m pip install -e ".[gui,dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Les commentaires de conception sont bilingues français/anglais. La CI cible Python 3.11 et 3.12. Les tests incluent une inférence réelle et un export TorchScript avec des poids aléatoires : ils vérifient le fonctionnement logiciel, pas la précision d'un modèle PCB entraîné.

Les exemples de `images/` proviennent du dataset PCB Defects cité ci-dessus et ne forment pas un jeu de validation indépendant. Aucune vitesse ni précision n'est garantie. Mesurez mAP, précision et rappel sur vos propres données ; une absence de détection ne certifie pas qu'une carte est sans défaut.

## English instructions

- **CPU/web:** create a Python 3.11 virtual environment, run `python -m pip install -r requirements.txt`, then `python app.py`. Open `http://localhost:7860`.
- **Dependencies:** there is one root `requirements.txt`. Shared dependencies and optional `web`, `gui`, `dev` and `onnx` features are declared in `pyproject.toml`. Add features with `python -m pip install -e ".[gui,dev]"`, for example.
- **Weights/Spaces:** supply trusted six-class PCB `.pt` weights locally or set `PCB_MODEL_REPO`. Use a Gradio Space with this README's YAML header and store `HF_TOKEN` as a secret for private models. `.env.example` is not loaded automatically.
- **GPU/Kaggle:** keep a suitable GPU PyTorch installation, then run `python -m pip install -e .` and `python main.py train`. Avoid the Linux CPU requirements file in that environment.
- **Desktop:** install `.[gui]` and run `python -m gui_test.app`; system Tkinter is required.
- **Validation:** the commands above cover inspection, export and quality checks. Random-weight integration tests and the six bundled images are not evidence of PCB detection accuracy.

## Licence · License

Le code est sous [licence MIT](LICENSE). Les dépendances, poids et datasets conservent leurs licences respectives, notamment la [licence Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

Project code uses the MIT license. External dependencies, weights and datasets retain their own licenses.

Auteur / Author : [Alain Paluku](https://github.com/alainpaluku).
