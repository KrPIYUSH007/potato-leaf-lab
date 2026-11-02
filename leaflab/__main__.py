import argparse
from .download import download
from .prepare import prepare, validate_manifest


def main():
    parser = argparse.ArgumentParser(description='Potato Leaf Lab data preparation')
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
    args = parser.parse_args()
    if args.command == 'download':
        download(args.data, args.workers)
    elif args.command == 'prepare':
        prepare(args.data, args.out, args.seed, args.dhash_distance, args.groups)
    else:
        rows = validate_manifest(args.manifest, args.data)
        print(f'Validated {len(rows)} rows; no group or exact-duplicate leakage.')


if __name__ == '__main__':
    main()
