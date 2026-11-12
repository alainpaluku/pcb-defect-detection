# Poids entraînés · Trained weights

FR : Aucun poids entraîné n'est livré avec le dépôt. Placez un checkpoint YOLO de détection entraîné sur les six classes PCB dans `models/pcb_model.pt`. La CLI et l'application de bureau utilisent ce fichier ; l'interface web peut aussi le télécharger depuis un dépôt de modèles Hugging Face. Consultez les variables `PCB_MODEL_*` dans le [README principal](../README.md#déploiement-sur-hugging-face-spaces).

EN: No trained checkpoint is included. Place a YOLO detection checkpoint trained on the six PCB classes in `models/pcb_model.pt`. CLI and desktop tools can use that file; the web app can also download weights from a Hugging Face model repository. See the [main README](../README.md#english-guide).

Les fichiers `.pt` sont ignorés par Git. Ne chargez que des poids de confiance. Un modèle COCO générique ne détecte pas les défauts PCB et est refusé. L'export ONNX est une commande facultative, sans vitesse garantie ; l'interface de bureau et le Space utilisent `.pt`.

`.pt` files are ignored by Git. Load trusted checkpoints only. Generic COCO weights are rejected. ONNX export is optional and comes with no latency guarantee; desktop and Space interfaces use `.pt`.
