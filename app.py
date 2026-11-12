"""EN: Hugging Face Spaces entry point. FR: Point d'entrée Hugging Face Spaces."""

import logging
import os

# EN: Disable dependency telemetry and runtime installs. FR: Désactiver télémétrie et installations à l'exécution.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import gradio as gr

from src.config import PROJECT_ROOT
from src.model import ModelLoadError
from src.space import InferenceService, SpaceSettings

logger = logging.getLogger(__name__)


def create_app(service: InferenceService | None = None) -> gr.Blocks:
    """EN: Building the UI never downloads weights. FR: Construire l'interface sans télécharger de poids."""
    service = service or InferenceService()

    def predict(image, confidence, iou):
        try:
            return service.predict(image, confidence, iou)
        except (ValueError, FileNotFoundError, ModelLoadError) as error:
            raise gr.Error(str(error)) from error
        except Exception as error:
            logger.exception("Inference failed / Échec de l’inférence")
            raise gr.Error(
                "Modèle indisponible ou inférence impossible. Vérifiez les poids et les journaux. / "
                "Model unavailable or inference failed. Check weights and logs."
            ) from error

    with gr.Blocks(title="PCB Defect Detection", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            "# Détection de défauts PCB · PCB Defect Detection\n"
            "Chargez une image de circuit imprimé, ajustez les seuils et lancez l’inspection.\n\n"
            "Upload a PCB image, adjust the thresholds and run the inspection."
        )
        with gr.Row():
            with gr.Column():
                image = gr.File(
                    label="Image PCB / PCB image", file_types=["image"], type="filepath"
                )
                confidence = gr.Slider(0, 1, value=0.3, step=0.01, label="Confiance / Confidence")
                iou = gr.Slider(0, 1, value=0.5, step=0.01, label="Seuil IoU / IoU threshold")
                run = gr.Button("Inspecter / Inspect", variant="primary")
                examples = sorted(str(path) for path in (PROJECT_ROOT / "images").glob("*.jpg"))
                if examples:
                    gr.Examples(
                        examples=[[path] for path in examples], inputs=[image], cache_examples=False
                    )
            with gr.Column():
                output = gr.Image(
                    label="Résultat annoté / Annotated result", type="pil", format="png"
                )
                summary = gr.Textbox(label="Résumé / Summary", interactive=False)
        table = gr.Dataframe(
            headers=["Classe / Class", "Confiance / Confidence", "x1", "y1", "x2", "y2"],
            datatype=["str", "number", "number", "number", "number", "number"],
            interactive=False,
        )
        details = gr.JSON(label="Résultats structurés / Structured results")
        gr.Markdown(
            "Une absence de détection ne garantit pas l’absence de défaut. / "
            "No detection does not guarantee a defect-free board."
        )
        run.click(
            predict,
            [image, confidence, iou],
            [output, table, summary, details],
            api_name="predict",
            concurrency_limit=1,
        )
    return demo.queue(max_size=16, default_concurrency_limit=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = SpaceSettings.from_env()
    demo = create_app(InferenceService(settings))
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        max_file_size=f"{settings.max_file_mb}mb",
        show_error=False,
    )


if __name__ == "__main__":
    main()
