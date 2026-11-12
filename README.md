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

# PCB Defect Detection · Détection de défauts PCB

Détection de six types de défauts de circuits imprimés avec YOLO11. Le même moteur Python alimente une interface **Gradio pour Hugging Face Spaces**, une application de bureau Tk et les commandes d'entraînement, de détection et d'export.

**Les poids entraînés ne sont pas inclus.** Fournissez un modèle PCB `.pt` local ou hébergé dans un dépôt de modèles Hugging Face. L'application web démarre sans poids, mais une inspection nécessite un modèle entraîné sur les six classes ci-dessous. Un modèle YOLO générique COCO est refusé.

**English:** Detect six PCB defect classes with YOLO11 using a Gradio Space, a desktop application or the CLI. **Trained weights are not included.** Supply your own PCB `.pt` checkpoint locally or through the Hugging Face Hub. The web UI can start without weights; inference requires a compatible PCB model. See the [English guide](#english-guide).

## Classes

| ID canonique / Canonical ID | Classe / Class | Français | English |
|---|---|---|---|
| 0 | `missing_hole` | Trou manquant | Missing drill hole |
| 1 | `mouse_bite` | Échancrure du cuivre | Copper edge notch |
| 2 | `open_circuit` | Piste interrompue | Broken trace |
| 3 | `short` | Court-circuit | Short circuit |
| 4 | `spur` | Excroissance de cuivre | Copper protrusion |
| 5 | `spurious_copper` | Cuivre parasite | Unwanted copper |

Les noms du checkpoint font autorité : un ordre de classes différent est correctement traduit vers ces identifiants canoniques. Les variantes de casse, d'espaces et de tirets sont acceptées.

## Démarrage local

Python **3.11 à 3.13** ; la configuration Spaces cible **3.11**. Exécutez les commandes depuis la racine du dépôt.

```bash
git clone https://github.com/alainpaluku/pcb-defect-detection.git
cd pcb-defect-detection
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Ouvrez `http://localhost:7860`. Sous Windows, activez l'environnement avec `.venv\Scripts\activate`.

`requirements.txt` installe PyTorch CPU sur Linux. Pour entraîner sur GPU, utilisez **un environnement séparé**, installez la version PyTorch adaptée à votre matériel suivant la [documentation PyTorch](https://pytorch.org/get-started/locally/), puis `requirements-core.txt`. La variante [Ultralytics headless](https://docs.ultralytics.com/quickstart/#headless-install) évite les dépendances graphiques OpenCV sur les serveurs ; Tk affiche les images avec Pillow.

Pour une utilisation locale sans téléchargement, placez votre checkpoint dans `models/pcb_model.pt`. Ne chargez que des poids `.pt` provenant d'une source de confiance : ces fichiers ne constituent pas un format de données neutre.

## Déploiement sur Hugging Face Spaces

1. Créez un Space avec le SDK **Gradio**.
2. Copiez les fichiers de ce dépôt dans le dépôt du Space, en conservant le bloc YAML en tête de ce README. N'envoyez pas `.venv/`, `data/`, `output/` ou des secrets.
3. Dans **Settings → Variables and secrets**, configurez les variables ci-dessous. Le chemin le plus simple est un dépôt de modèles contenant votre `pcb_model.pt`.
4. Laissez le Space installer ses dépendances puis ouvrir l'interface. Lancez une inspection pour vérifier le chargement des poids.

| Variable | Valeur par défaut | Fonction |
|---|---|---|
| `PCB_MODEL_PATH` | Non définie | Chemin local explicite, prioritaire ; chemin relatif à la racine du projet |
| `PCB_MODEL_REPO` | Non définie | Dépôt Hugging Face, par exemple `votre-compte/pcb-defect-model` |
| `PCB_MODEL_FILENAME` | `pcb_model.pt` | Nom du fichier dans le dépôt de modèles |
| `PCB_MODEL_REVISION` | `main` | Branche, tag ou commit ; fixer un commit pour reproduire un déploiement |
| `HF_TOKEN` | Non défini | **Secret** de lecture requis seulement pour des poids privés ou à accès restreint |
| `PCB_DEVICE` | `cpu` | Périphérique d'inférence |
| `PCB_IMAGE_SIZE` | `640` | Taille d'inférence, multiple de 32 |
| `PCB_CPU_THREADS` | `2` | Nombre de threads PyTorch |
| `PCB_MAX_DETECTIONS` | `300` | Nombre maximal de détections par image |
| `PCB_MAX_FILE_MB` | `20` | Taille maximale du fichier importé en Mio (20 × 1024² octets) |
| `PCB_MAX_PIXELS` | `20000000` | Nombre maximal de pixels avant décodage |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Adresse d'écoute |
| `GRADIO_SERVER_PORT` | `7860` | Port HTTP |

Résolution des poids : `PCB_MODEL_PATH` explicite → dépôt `PCB_MODEL_REPO` → `models/pcb_model.pt`. Une source explicite invalide déclenche une erreur, sans remplacement silencieux par un autre modèle. `.env.example` est un exemple ; il n'est pas chargé automatiquement.

Les poids sont téléchargés à la première inspection, conservés dans le cache Hugging Face puis chargés une seule fois par application. L'interface ne permet pas aux visiteurs d'importer des checkpoints. Les erreurs détaillées figurent dans les journaux du Space.

Les choix de déploiement suivent les documentations officielles des [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio), de la [configuration Spaces](https://huggingface.co/docs/hub/spaces-config-reference) et du [cache de téléchargement Hub](https://huggingface.co/docs/huggingface_hub/en/guides/download).

### Ressources et limites

- Une inférence à la fois, avec une file d'attente de 16 requêtes pour borner la charge.
- Traitement en mémoire, sans création systématique de dossiers `runs/`.
- Limites de fichier et de pixels, conversion RGB et prise en compte de l'orientation EXIF.
- Nettoyage du cache temporaire Gradio chaque heure pour les fichiers âgés d'au moins une heure ; ce n'est pas un effacement immédiat après une requête.
- Le temps affiché mesure la prédiction, en excluant le téléchargement, le chargement des poids et le rendu. La première prédiction peut inclure l'initialisation du moteur.
- Les performances réelles dépendent du modèle, du processeur et de l'image. Réduire la taille d'inférence peut accélérer le traitement et faire perdre de petits défauts : mesurer ce compromis sur vos données.

La configuration fournie vise les Spaces CPU classiques. Une variante GPU nécessite un environnement PyTorch adapté ; ZeroGPU n'est pas intégré.

## Ligne de commande

```bash
# FR: Inspecter une image. EN: Inspect one image.
python main.py detect images/01_short_01.jpg --model models/pcb_model.pt --conf 0.3 --iou 0.5

# FR: Parcourir un dossier récursivement. EN: Recursively inspect a directory.
python main.py detect images --model models/pcb_model.pt

# FR: Entraîner un modèle. EN: Train a model.
python main.py train --data data/pcb-defects --epochs 50 --batch 16 --model yolo11m.pt

# FR: Export ONNX facultatif. EN: Optional ONNX export.
python -m pip install -r requirements-export.txt
python main.py export --model models/pcb_model.pt --format onnx --output output/exports
```

`--save` active l'enregistrement des images annotées par Ultralytics. L'inspection de dossiers retourne les chemins relatifs pour distinguer deux images de même nom. Les variables `PCB_MODEL_*` concernent le service web ; la CLI utilise `--model` ou la recherche locale dans `models/`, `output/`, puis la racine.

`--format all` exporte **ONNX et TorchScript**. Les formats `tflite`, `coreml` et `openvino` sont accessibles explicitement mais nécessitent leurs propres dépendances et plateformes ; ils ne sont pas installés avec l'application. Les erreurs d'export sont remontées et une destination existante n'est pas écrasée par la copie.

## Données et entraînement

Le pipeline attend des images et des annotations VOC XML correspondantes, par exemple celles du jeu [PCB Defects sur Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects). Le téléchargement du dataset n'est pas automatique.

```text
data/pcb-defects/
├── images/
│   ├── Missing_hole/01_missing_hole_01.jpg
│   └── Short/02_short_01.jpg
└── Annotations/
    ├── Missing_hole/01_missing_hole_01.xml
    └── Short/02_short_01.xml
```

Le nom sans extension permet l'association ; les dossiers de classe départagent les noms identiques. Une association encore ambiguë est refusée. Les images sans XML ne sont pas utilisées. Un XML sans objet constitue un exemple négatif valide. Les dimensions XML doivent correspondre à l'image. Les coordonnées sont traitées comme des coins en pixels, bornées à l'image, puis normalisées ; vérifiez la convention de coordonnées de tout autre dataset.

La séparation par défaut regroupe les images partageant le préfixe numérique avant `_`, par exemple `01_…`, pour garder les variantes d'une même carte dans un seul jeu. Au moins deux groupes sont requis. Cette convention vient des noms du dataset ; pour un autre jeu, vérifiez les identifiants de cartes. `--split-mode image` autorise une séparation par image, mais ne protège pas contre la présence de variantes d'une même carte des deux côtés. Les copies de contenu strictement identique sont refusées.

```bash
python main.py train --data data/pcb-defects --output output \
  --epochs 50 --batch 16 --img-size 640 --workers 2 \
  --val-split 0.2 --seed 42 --split-mode board
```

Le ratio peut s'écarter de 20 % lorsque les groupes ont des tailles différentes. Le manifeste `output/yolo_dataset/.pcb-dataset.json` conserve les associations et la séparation. Seul un dataset généré portant ce marqueur peut être remplacé. Un ancien dossier YOLO sans marqueur doit être conservé ailleurs ou une autre destination doit être choisie.

Les entraînements successifs utilisent `pcb_yolo`, `pcb_yolo2`, etc. Le meilleur checkpoint du nouvel entraînement est évalué puis copié vers `output/pcb_model.pt`. Ce dernier fichier et `training_summary.json` représentent **la dernière exécution** ; les dossiers d'entraînement précédents sont conservés. Les graphiques sont repris des artefacts Ultralytics, lorsqu'ils sont disponibles. Le JSON contient la configuration et les métriques `map50`, `map50_95`, `precision`, `recall`.

### Kaggle

Ajoutez le dataset à un notebook, activez Internet et le GPU, puis exécutez :

```python
!git clone https://github.com/alainpaluku/pcb-defect-detection.git
%cd pcb-defect-detection
!pip install -r requirements-core.txt
!python run_kaggle.py --epochs 100 --device 0 --workers 2
```

Pour deux GPU disponibles, utilisez `--device 0,1`. Ne remplacez pas le PyTorch GPU de Kaggle par `requirements.txt`, qui cible le déploiement CPU Linux. Les résultats sont placés dans `/kaggle/working/`, notamment `pcb_model.pt` et `training_summary.json`. Aucun export `.pth` ou ONNX implicite n'est annoncé.

## Application de bureau

```bash
python -m pip install -r gui_test/requirements.txt
python -m gui_test.app
```

Tkinter doit être disponible via votre installation Python ou le gestionnaire de paquets du système. L'application charge les modèles `.pt`, permet les réglages de confiance et IoU, traite les dossiers récursivement et exporte les résultats en JSON, CSV ou texte UTF-8. Les échecs d'images d'un lot sont affichés séparément des inspections sans détection.

## Architecture et qualité

```text
app.py                       Interface Gradio / Gradio UI
main.py, run_kaggle.py        Points d'entrée / Entry points
src/config.py                Configuration validée / Validated settings
src/model.py                 Adaptateur YOLO / YOLO adapter
src/detector.py               Résultats structurés / Structured results
src/space.py                  Service web et poids Hub / Web service and Hub weights
src/data_ingestion.py         VOC, séparation et manifeste / VOC, splits and manifest
src/trainer.py                Entraînement et artefacts / Training and artifacts
gui_test/                    Interface de bureau / Desktop UI
tests/                       Régressions et intégration / Regression and integration tests
docs/AUDIT.md                Constats et corrections / Findings and fixes
```

Les commentaires de conception sont rédigés en anglais et en français. Les identifiants restent en anglais pour conserver une API cohérente.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

La CI exécute ces vérifications avec Python 3.11 et 3.12. Les tests utilisent des fichiers temporaires, des doubles du moteur et un modèle YOLO à poids aléatoires pour vérifier l'intégration sans télécharger de checkpoint. Ce modèle de test n'évalue pas la qualité de détection. Consultez le [rapport d'audit](docs/AUDIT.md) pour les vérifications et limites.

## Résultats et limites scientifiques

Les images de `results/` sont des artefacts historiques. Le dépôt ne contient ni le checkpoint entraîné ni un ensemble complet de mesures permettant de reproduire les anciens chiffres. Aucune précision, durée d'entraînement ou accélération n'est donc garantie par ce README.

Mesurez les résultats sur un jeu de test indépendant, idéalement séparé par carte physique et acquisition. **mAP@0.5 n'est pas la précision** : les quatre mesures restent distinctes. Une absence de détection à un seuil donné ne certifie pas qu'une carte est sans défaut.

## English guide

1. Create a Python 3.11 virtual environment, install `requirements.txt`, then run `python app.py`.
2. Put trusted PCB weights in `models/pcb_model.pt`, or set `PCB_MODEL_REPO`, `PCB_MODEL_FILENAME` and optionally `PCB_MODEL_REVISION`. An explicit `PCB_MODEL_PATH` takes precedence. Store `HF_TOKEN` as a Space secret if the model is private. `.env.example` is documentation, not an automatically loaded dotenv file.
3. Create a **Gradio** Space and upload this repository while preserving the README YAML header. The app starts without weights and loads them on the first inference. It rejects checkpoints with incompatible class names. This repository does not deploy itself or include a public model.
4. Upload an image, set confidence and IoU, and inspect the annotated result, table and JSON. CPU inference is serialized, with a 16-request queue, two PyTorch threads, a 20 MiB upload limit and a 20-megapixel decoded-image limit. Temporary Gradio files are cleaned hourly once at least one hour old.
5. CLI examples above also apply in English. `detect` accepts files or recursive directories; `--save` opts into writing annotations. `export --format all` means ONNX plus TorchScript. Other export backends need separate dependencies.
6. Training uses paired VOC XML files, rejects unknown classes and ambiguous image matches, preserves annotated negative images, and creates deterministic board-group splits. Numeric filename prefixes identify boards by default; verify that convention for your dataset. `--split-mode image` is an explicit alternative. A manifest records membership and guards generated-directory replacement.
7. GPU training uses `requirements-core.txt` with a suitable GPU PyTorch installation, not the Linux CPU deployment requirements. Kaggle uses the same CLI through `run_kaggle.py`. New runs receive distinct directories; `pcb_model.pt` and `training_summary.json` are aliases for the latest run.
8. For the optional desktop UI, install `gui_test/requirements.txt` and run `python -m gui_test.app`; system Tkinter is required. Source comments explaining design decisions are bilingual.

Historical screenshots are not reproducible benchmarks. Validate accuracy on independent PCB data; a successful software test or a lack of detections is not evidence of manufacturing quality.

## Licence / License

Le code du projet conserve sa [licence MIT](LICENSE). Les dépendances, les poids et les datasets gardent leurs licences respectives ; consultez notamment la [licence Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). Aucune licence de dataset ou de poids n'est accordée par ce dépôt.

Project code retains its [MIT license](LICENSE). Dependencies, datasets and model weights retain their own licenses. The repository does not grant rights to external weights or datasets.

Auteur / Author: [Alain Paluku](https://github.com/alainpaluku).
