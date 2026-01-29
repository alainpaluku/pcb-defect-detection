"""Main entry point for PCB Defect Detection System."""

import argparse
from pathlib import Path

from src.config import Config
from src.utils import print_detection_results


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='PCB Defect Detection with YOLO11',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, help='Path to dataset')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--img-size', type=int, default=640, help='Image size')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run detection on images')
    detect_parser.add_argument('source', type=str, help='Image or directory path')
    detect_parser.add_argument('--model', type=str, help='Path to trained model')
    detect_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    detect_parser.add_argument('--save', action='store_true', help='Save results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--model', type=str, required=True, help='Path to model')
    export_parser.add_argument('--format', type=str, default='onnx', 
                              choices=['onnx', 'torchscript', 'tflite', 'coreml', 'openvino', 'all'],
                              help='Export format (use "all" for multiple formats)')
    export_parser.add_argument('--output', type=str, help='Output directory')
    
    return parser


def cmd_train(args: argparse.Namespace) -> None:
    """Execute training command."""
    from src.trainer import TrainingManager
    
    config = Config.create(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size
    )
    
    trainer = TrainingManager(
        data_path=Path(args.data) if args.data else None,
        config=config
    )
    metrics = trainer.run_pipeline()
    print(f"\nFinal detection precision: {metrics.get('precision_detection', 0):.4f}")


def cmd_detect(args: argparse.Namespace) -> None:
    """Execute detection command."""
    from src.detector import PCBInspector
    
    inspector = PCBInspector(model_path=args.model)
    source = Path(args.source)
    
    if source.is_dir():
        results = inspector.inspect_batch(source, conf=args.conf, save=args.save)
        for img_name, detections in results.items():
            print(f"\n{img_name}:")
            print_detection_results(detections)
    else:
        detections = inspector.inspect(source, conf=args.conf, save=args.save)
        print_detection_results(detections)


def cmd_export(args: argparse.Namespace) -> None:
    """Execute export command."""
    from src.model import PCBDetector
    
    detector = PCBDetector(model_path=args.model)
    
    if args.format == 'all':
        print("🔄 Exporting to all supported formats...")
        exported_paths = detector.export_multiple_formats()
        
        print(f"\n✅ Export completed! Files:")
        for format_name, path in exported_paths.items():
            print(f"   • {format_name.upper()}: {path}")
    else:
        print(f"🔄 Exporting to {args.format}...")
        path = detector.export(format=args.format)
        print(f"✅ Exported: {path}")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    commands = {
        'train': cmd_train,
        'detect': cmd_detect,
        'export': cmd_export,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
