# Audit et corrections · Audit and fixes

Date : 12 novembre 2026 / November 12, 2026.

Référence analysée / Reviewed baseline : `25ce102`, dépôt `alainpaluku/pcb-defect-detection`.
Branche de travail locale / Local working branch : `refactor/huggingface-clean-code`.

## Périmètre / Scope

Lecture de tous les modules Python de `src/`, `gui_test/`, des points d'entrée, des tests initiaux, des fichiers de dépendances, de la configuration Git et des README. Inventaire des images et des artefacts historiques. Les fichiers binaires de `results/` ne constituent pas une preuve reproductible des métriques du modèle ; aucun poids entraîné n'est présent à la référence examinée.

All Python modules, entry points, original tests, dependency files, Git configuration and README files were reviewed. Historical images were inventoried. Trained weights are absent from the reviewed revision, so historical detection metrics cannot be reproduced from the repository alone.

## Problèmes corrigés / Fixed issues

| Domaine / Area | Constat initial / Initial finding | Correction |
|---|---|---|
| Configuration | Les instances partageaient les mêmes dataclasses mutables. | `default_factory`, paramètres indépendants et validation des valeurs. |
| Seuils / Thresholds | `conf or default` et `iou or default` ignoraient zéro. | Distinguer `None` de `0` ; refuser NaN et les valeurs hors de `[0, 1]`. |
| Modèle / Model | Les classes étaient associées à une liste fixe sans vérifier les poids. | Vérifier la tâche et les six noms ; traduire les identifiants selon les noms du checkpoint. |
| Chargement / Loading | La GUI annonçait le chargement avant que le modèle soit réellement ouvert. | Charger et vérifier les poids avant d'annoncer le succès ; conserver le modèle précédent en cas d'échec. |
| Inférence / Inference | Certains réglages définis n'étaient pas transmis ; sauvegardes implicites. | Transmettre IoU, taille, nombre maximal de boîtes, périphérique et NMS ; pas d'écriture par défaut. |
| Concurrence / Concurrency | Prédicteur partagé et lectures de widgets depuis un worker. | Verrou réentrant du prédicteur ; valeurs Tk capturées avant lancement ; livraison des résultats via une file sur le fil principal. |
| GUI IoU | Le curseur IoU ne changeait pas la prédiction. | Transmettre la valeur pour les inspections individuelles et par lots. |
| GUI résultats / Results | Résumés périmés après zéro détection et erreurs différées fragiles. | Distinguer une liste vide d'une absence de résultat ; livrer les exceptions via la file ; effacer les résultats périmés. |
| Exports texte / Text | Des séquences `\\n` apparaissaient littéralement. | Vrais sauts de ligne, UTF-8, CSV avec en-tête même sans détection. |
| Lots / Batches | Parcours incomplet et collisions entre noms identiques. | Parcours récursif trié et identifiants de chemins distincts ; échecs séparés des résultats négatifs. |
| Visualisation | Le chemin demandé n'était pas utilisé. | Sauver exactement au chemin demandé en convertissant BGR vers RGB. |
| XML | Recherche répétée des images, classes ignorées silencieusement, erreurs de coordonnées. | Index unique, appariement contrôlé, noms normalisés, dimensions vérifiées, coordonnées finies et boîtes non vides ; parseur XML durci. |
| Négatifs / Negatives | Les XML sans objet étaient supprimés du dataset. | Conserver une image avec annotation vide comme exemple négatif valide. |
| Séparation / Splitting | Mélange possible entre variantes d'une carte et mutation du générateur aléatoire global. | Groupes par préfixe numérique, générateur local, graine et manifeste explicites. |
| Intégrité / Integrity | Des images de même nom pouvaient s'écraser ; des anciens fichiers restaient après reconstruction. | Noms de sortie dérivés du chemin, refus des copies de contenu identique, préparation dans un répertoire temporaire et remplacement des seuls jeux portant un marqueur. |
| Entraînement / Training | Configuration personnalisée non transmise à l'ingestion et options étrangères à la détection. | Même configuration sur tout le pipeline et paramètres de détection pris en charge. |
| Checkpoint | Chemin `pcb_yolo/weights/best.pt` supposé, y compris pour une nouvelle exécution. | Lire les chemins réels de l'entraîneur et évaluer le checkpoint de cette exécution. |
| Validation | Les valeurs du checkpoint pouvaient diriger les sorties vers un ancien chemin. | Répertoire `pcb_validation*` explicitement placé dans la destination configurée. |
| Export modèle / Model export | L'option de format était ignorée, `save()` interprété comme un chemin et `--output` inutilisé. | Exports réels, chemins de retour vérifiés par intégration, destination appliquée et erreurs remontées. |
| Artefacts / Artifacts | `.pth` présenté comme un state dict et promesses de formats non générés. | Conserver le vrai `.pt`, exporter à la demande et écrire un rapport JSON de configuration et métriques. |
| Démarrage / Startup | Imports d'entraînement et configuration globale des logs au simple import. | Imports différés et logs configurés aux points d'entrée. |
| Kaggle | Installation implicite à chaque lancement et changement de dossier global. | Même CLI que le local ; installation explicite ; conservation du PyTorch GPU de Kaggle. |
| Dépendances / Dependencies | Paquets lourds d'export et d'entraînement installés pour tous les usages. | Séparer moteur, application CPU, développement, bureau et export ONNX ; un seul fournisseur OpenCV headless. |
| Compatibilité | Versions récentes de Pillow incompatibles avec Gradio. | Contraindre Pillow à `<13` et vérifier la résolution et `pip check`. |
| Images invalides / Invalid images | Le patch Pillow d'Ultralytics pouvait réclamer un codec facultatif absent. | Transformer cette absence en erreur d'image contrôlée ; aucune installation dynamique depuis le Space. |
| Hugging Face | Aucun point d'entrée web ni configuration Spaces. | `app.py`, métadonnées README, poids locaux ou Hub, chargement différé et variables documentées. |
| Ressources / Resources | Pas de borne explicite pour le service web. | File de 16 requêtes, concurrence 1, limites de fichier/pixels, threads CPU et cache temporaire nettoyé périodiquement. |
| Documentation | Chiffres contradictoires et vitesses présentées comme acquises. | README bilingue avec commandes exécutables, exigences de poids et limites explicites ; retrait des garanties non reproductibles. |
| Qualité / Quality | Tests largement limités aux imports, certains déjà incohérents avec le code. | Régressions sur les comportements, fichiers temporaires, intégration YOLO réelle, Ruff et CI Python 3.11/3.12. |

## Vérifications effectuées / Validation performed

Environnement local : Python 3.11.18, PyTorch `2.10.0+cpu`, torchvision `0.25.0+cpu`, Ultralytics headless `8.4.210`, Gradio `6.31.0`, Pillow `12.3.0`, NumPy `2.4.6`.

- `python -m ruff check .` : conforme / passed.
- `python -m ruff format --check .` : conforme / passed.
- `python -m pytest -q` : **62 tests réussis / 62 passed**.
- `python -m pip check` : aucune incompatibilité / no broken requirements.
- `git diff --check` : aucune erreur d'espacement / no whitespace errors.
- Instanciation réelle de la fenêtre Tk, lecture des curseurs et résumé sans détection : réussie / actual Tk startup, slider values and empty-result summary passed.
- Démarrage HTTP de Gradio sans poids et erreur explicite à l'appel de l'API : réussis / HTTP startup without weights and explicit missing-weights error passed.
- Import d'une image via `gradio_client`, inférence YOLO et retour image/tableau/JSON : réussi / image upload, YOLO inference and image/table/JSON response passed.
- Création d'un YOLO11n à six classes et poids aléatoires, sauvegarde `.pt`, inférence, export TorchScript puis rechargement : réussis / actual random-weight inference and TorchScript round trip passed.
- Une époque d'entraînement CPU sur quatre images synthétiques, évaluation du meilleur checkpoint et génération du `.pt` et du rapport JSON : réussies / one-epoch synthetic CPU training, evaluation and artifact generation passed.

Les avertissements de dépréciation TorchScript viennent de PyTorch pendant l'export facultatif. Ils ne provoquent pas d'échec. L'export ONNX est disponible mais n'a pas été exécuté dans cette vérification locale.

PyTorch emits deprecation warnings during the optional TorchScript export. Tests pass. ONNX export was not exercised in this local verification.

## Limites restantes / Remaining limits

- Aucun poids PCB entraîné n'a été fourni ; les tests d'intégration utilisent un modèle initialisé aléatoirement. Ils vérifient les chemins logiciels, pas la qualité de détection.
- Aucun entraînement complet du dataset Kaggle, benchmark CPU/GPU ou mesure de précision de production n'a été effectué.
- La convention de regroupement par préfixe numérique doit être vérifiée pour un autre dataset. Elle ne remplace pas une identification fiable des cartes et acquisitions.
- Les mécanismes Hub sont testés avec un téléchargement simulé ; un dépôt de poids réel et son accès privé ne sont pas disponibles dans ce travail.
- La configuration Spaces est préparée et le serveur local est testé ; aucun Space n'a été publié. Le navigateur intégré étant indisponible, le rendu visuel web n'a pas fait l'objet d'une inspection manuelle ; la construction Gradio et ses appels HTTP ont été vérifiés.
- La CI est configurée pour Python 3.11 et 3.12 mais n'a pas été exécutée sur GitHub. La vérification locale utilise Python 3.11.

No trained PCB checkpoint or full dataset training was used to claim accuracy. Real Hub authentication, hosted Space deployment, browser layout inspection, GPU performance and the remote CI matrix remain unverified. Local functional tests do not certify manufacturing quality.

## Évolution d'API / API changes

Cette refactorisation constitue la version 3. Les métriques sont nommées `map50`, `map50_95`, `precision` et `recall`. Une inspection sans détection porte le statut `NO_DETECTIONS`, plutôt que de déclarer la carte conforme. La configuration s'utilise par instance (`Config().data`), et la normalisation des classes passe par `Config.get_class_id`. La détection par dossier utilise des chemins relatifs. Les points d'entrée de la CLI, de Kaggle et du bureau restent disponibles.

Version 3 uses unambiguous metric names, `NO_DETECTIONS` for empty inspections, instance-owned configuration and relative keys for recursive directory results. CLI, Kaggle and desktop entry points remain available.
