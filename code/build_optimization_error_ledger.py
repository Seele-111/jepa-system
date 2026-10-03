"""Build a new, exclusive diagnostic ledger from frozen formal outer OOF.

Example (the destination MUST NOT exist):
  python -B code/build_optimization_error_ledger.py --output <new-directory>
  python -B code/build_optimization_error_ledger.py --output <new-directory> --compare none

Use --report <report.json> and optional --primary-key primary_v10_nested for new
versions; without a key exactly one primary_*_nested must be present.
Default comparisons: content-grouped v1, formal nested v4/v5/v8/v9. Missing optional
reports are explicitly listed; inconsistent present reports are errors. No
fullfit fallback, training, prediction re-execution, video decoding or network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from zipfile import BadZipFile

from optimized_error_ledger import (EXPERIMENTS, GROUPED, ORIGINAL, LedgerError,
                                    build_error_ledger, require, write_ledger)

ROOT = Path(__file__).resolve().parents[1]


def check_output_path(root, output, report=None):
    raw = Path(output)
    if os.path.lexists(raw):
        raise FileExistsError('output already exists; choose a new path: ' + str(raw))
    resolved, root = raw.resolve(), Path(root).resolve()
    protected = (ORIGINAL, GROUPED, *EXPERIMENTS.values(), Path('code'), Path('docs'),
                 Path('models'), Path('data'))
    if report is not None:
        protected += ((root / report).resolve().parent,)
    require(not any(resolved.is_relative_to((root / p).resolve()) for p in protected),
            'output cannot be inside frozen inputs or source/model/data directories')
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='workspace containing frozen output assets')
    parser.add_argument('--output', type=Path, required=True, help='brand-new output directory; no overwrite/resume')
    parser.add_argument('--report', type=Path, help='explicit workspace-local completed nested report')
    parser.add_argument('--primary-key', help='primary_*_nested key; omit only when unambiguous')
    parser.add_argument('--compare', default='v1,v4,v5,v8,v9', help='comma-separated existing formal versions, or none')
    args = parser.parse_args(argv)
    try:
        output = check_output_path(args.root, args.output, args.report)
        comparisons = (() if args.compare.strip().lower() == 'none' else
                       tuple(v.strip() for v in args.compare.split(',')))
        ledger = build_error_ledger(args.root, comparisons, args.report, args.primary_key)
        ledger['provenance']['cli_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_ledger(output, ledger)
    except (OSError, LedgerError, KeyError, TypeError, ValueError, BadZipFile, EOFError) as exc:
        print('error-ledger: ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({'status': 'complete', 'output': str(output),
        'dataset': ledger['summary']['dataset'], 'primary_pipeline': ledger['summary']['primary_pipeline'],
        'pipelines': list(ledger['summary']['pipelines']),
        'video_rows': len(ledger['videos']), 'event_rows': len(ledger['events']),
        'prediction_rows': len(ledger['predictions']), 'artifacts': 11,
        'fullfit_predictions_used': False, 'semantic_cause': 'unknown'}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
