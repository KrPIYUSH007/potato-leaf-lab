import argparse
from .download import download
from .prepare import prepare, validate_manifest


def main():
    parser = argparse.ArgumentParser(description='Potato Leaf Lab data and model experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    get = sub.add_parser('download', help='Download and verify Mendeley dataset version 1')
    get.add_argument('--data', default='data/raw')
    get.add_argument('--workers', type=int, default=8, choices=range(1, 17))
    prep = sub.add_parser('prepare', help='Audit and create immutable grouped image partitions')
    prep.add_argument('--data', default='data/raw')
    prep.add_argument('--out', default='reports/data-v1')
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--dhash-distance', type=int, default=4)
    prep.add_argument('--groups', help='Optional CSV: path,group for known same-plant/session images')
    check = sub.add_parser('validate', help='Verify partitions and optionally all image file hashes')
    check.add_argument('--manifest', default='reports/data-v1/manifest.csv')
    check.add_argument('--data', help='Dataset root; when supplied, recompute every file hash')
    environment = sub.add_parser('training-env', help='Verify PyTorch and CUDA execution')
    environment.add_argument('--out', default='reports/training_environment.json')
    baseline = sub.add_parser('baseline', help='Train and validate frozen InceptionV3 + SVM')
    baseline.add_argument('--data', default='data/raw')
    baseline.add_argument('--manifest', default='reports/data-v1/manifest.csv')
    baseline.add_argument('--out', default='runs/baseline-v1')
    baseline.add_argument('--report', default='reports/baseline-v1')
    baseline.add_argument('--cache', default='runs/feature-cache')
    baseline.add_argument('--batch-size', type=int, default=16)
    baseline.add_argument('--workers', type=int, default=2)
    baseline.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    predict = sub.add_parser('predict', help='Classify an image with a saved baseline bundle')
    predict.add_argument('--model', default='runs/baseline-v1')
    predict.add_argument('--image', required=True)
    predict.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.command == 'download':
        download(args.data, args.workers)
    elif args.command == 'prepare':
        prepare(args.data, args.out, args.seed, args.dhash_distance, args.groups)
    elif args.command == 'validate':
        rows = validate_manifest(args.manifest, args.data)
        print(f'Validated {len(rows)} rows; no group or exact-duplicate leakage.')
    else:
        from . import baseline as model_commands
        if args.command == 'training-env':
            model_commands.check_environment(args.out)
        elif args.command == 'baseline':
            model_commands.run_baseline(args)
        else:
            import json
            from pathlib import Path
            metadata = json.loads((Path(args.model) / 'metadata.json').read_text(encoding='utf-8'))
            if metadata.get('schema_version') == 5:
                from .v5 import predict
                predict(args.model, args.image, args.device)
            else:
                model_commands.predict_image(args.model, args.image, args.device)


if __name__ == '__main__':
    main()
