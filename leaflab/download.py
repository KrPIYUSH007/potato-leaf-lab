"""Download the cited Mendeley dataset with publisher SHA-256 checks."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

DATASET = 'ptz377bwb8'
VERSION = 1
BASE = 'https://data.mendeley.com'
ALIASES = {'phytopthora': 'phytophthora'}
CLASSES = ('bacteria', 'fungi', 'healthy', 'nematode', 'pest', 'phytophthora', 'virus')


def fetch_json(url):
    request = Request(url, headers={'Accept': 'application/vnd.mendeley-public-dataset.1+json',
                                   'User-Agent': 'PotatoLeafLab/0.1 (academic dataset preparation)'})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def catalogue():
    folders = fetch_json(f'{BASE}/public-api/datasets/{DATASET}/folders/{VERSION}')
    records = []
    for folder in folders:
        label = ALIASES.get(folder['name'].lower(), folder['name'].lower())
        if label not in CLASSES:
            continue
        start = 0
        while True:
            query = urlencode({'folder_id': folder['id'], 'version': VERSION,
                               '$start': start, '$limit': 1000})
            files = fetch_json(f'{BASE}/public-api/datasets/{DATASET}/files?{query}')
            for item in files:
                detail = item['content_details']
                filename = item['filename']
                if filename != Path(filename).name or '\\' in filename or filename in {'.', '..'}:
                    raise ValueError(f'Unsafe remote filename: {filename}')
                url = detail['download_url']
                if urlparse(url).scheme != 'https' or urlparse(url).hostname != 'data.mendeley.com':
                    raise ValueError('Unexpected dataset download host')
                records.append({'label': label, 'source_folder': folder['name'],
                                'path': f'{label}/{filename}', 'file_id': item['id'],
                                'sha256': detail['sha256_hash'], 'bytes': detail['size'], 'url': url})
            if len(files) < 1000:
                break
            start += len(files)
    if {r['label'] for r in records} != set(CLASSES) or len({r['path'] for r in records}) != len(records):
        raise ValueError('Catalogue has missing categories or colliding file paths')
    return records


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_one(record, root):
    path = root / record['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']:
        return 'verified existing'
    temp = path.with_suffix(path.suffix + '.part')
    last_error = None
    for attempt in range(4):
        try:
            request = Request(record['url'], headers={'User-Agent': 'PotatoLeafLab/0.1'})
            with urlopen(request, timeout=90) as response, temp.open('wb') as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            if temp.stat().st_size != record['bytes'] or sha256(temp) != record['sha256']:
                raise ValueError('Downloaded size or SHA-256 does not match publisher metadata')
            temp.replace(path)
            return 'downloaded'
        except Exception as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{record['path']}: {last_error}")


def download(root, workers=8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = catalogue()
    print(f'Catalogue: {len(records)} images, {sum(r["bytes"] for r in records)/1e9:.2f} GB', flush=True)
    metadata = {'dataset_id': DATASET, 'version': VERSION,
                'doi': '10.17632/ptz377bwb8.1', 'license': 'CC BY 4.0', 'files': records}
    (root / 'source_catalogue.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, r, root): r for r in records}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
            if n % 100 == 0 or n == len(records):
                print(f'Checked {n}/{len(records)}; failures {len(errors)}', flush=True)
    if errors:
        (root / 'download_errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
        raise RuntimeError(f'{len(errors)} downloads failed. Rerun to retry; verified files are reused.')
    (root / 'download_errors.json').unlink(missing_ok=True)
    print('All publisher file hashes verified.', flush=True)
