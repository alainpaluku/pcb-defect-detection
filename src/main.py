"""EN: Command-line entry point. FR: Point d'entrée en ligne de commande."""

import argparse
import logging
from pathlib import Path

from src.config import Config, DataConfig, InferenceConfig, ModelConfig
from src.utils import print_detection_results


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCB defect detection / Détection de défauts PCB")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train / Entraîner")
    train.add_argument("--data", type=Path)
    train.add_argument("--model", default="yolo11m.pt")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--img-size", type=int, default=640)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--device", default=None)
    train.add_argument("--output", type=Path, default=Config.get_output_path())
    train.add_argument("--val-split", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--split-mode", choices=["board", "image"], default="board")
    detect = commands.add_parser("detect", help="Inspect / Inspecter")
    detect.add_argument("source", type=Path)
    detect.add_argument("--model", type=Path)
    detect.add_argument("--conf", type=float, default=0.3)
    detect.add_argument("--iou", type=float, default=0.5)
    detect.add_argument("--device", default="cpu")
    detect.add_argument("--img-size", type=int, default=640)
    detect.add_argument("--save", action="store_true")
    export = commands.add_parser("export", help="Export / Exporter")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument(
        "--format",
        choices=["onnx", "torchscript", "tflite", "coreml", "openvino", "all"],
        default="onnx",
    )
    export.add_argument("--output", type=Path)
    return parser


def cmd_train(args) -> None:
    from src.trainer import TrainingManager

    config = Config(
        model=ModelConfig(
            name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            workers=args.workers,
            device=args.device,
        ),
        data=DataConfig(
            val_split=args.val_split,
            random_seed=args.seed,
            group_by_board=args.split_mode == "board",
        ),
        output_path=args.output,
    )
    print(TrainingManager(args.data, config).run_pipeline())


def cmd_detect(args) -> None:
    from src.detector import PCBInspector

    config = Config(
        inference=InferenceConfig(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            img_size=args.img_size,
            device=args.device,
        )
    )
    inspector = PCBInspector(args.model, config)
    if args.source.is_dir():
        for name, detections in inspector.inspect_batch(args.source, save=args.save).items():
            print(name)
            print_detection_results(detections)
    elif args.source.is_file():
        print_detection_results(inspector.inspect(args.source, save=args.save))
    else:
        raise FileNotFoundError(args.source)


def cmd_export(args) -> None:
    from src.model import PCBDetector

    detector = PCBDetector(args.model)
    detector.ensure_pcb_model()
    if args.format == "all":
        print(detector.export_multiple_formats(output=args.output))
    else:
        print(detector.export(args.format, output=args.output))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        {"train": cmd_train, "detect": cmd_detect, "export": cmd_export}[args.command](args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error / Erreur: {error}\n")


if __name__ == "__main__":
    main()
