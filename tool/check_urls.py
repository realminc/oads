#!/usr/bin/env python3
"""Probe every upstream URL in the dataset manifests.

`tool/gen/ds/schema/packs.json` records one or more `urls` per pack file.
This checker issues a `HEAD` against each (falling back to a one-byte
`GET` for hosts that reject `HEAD`) and reports which upstreams are alive.
Pack files often mirror the same bytes at several URLs; a file counts as
alive when at least one of its URLs answers.

    python3 tool/check_urls.py                 # probe all packs
    python3 tool/check_urls.py --packs mnist   # probe selected packs
    python3 tool/check_urls.py --verbose       # print every URL result

Exits 1 when any pack has a file with no live URL. Network-dependent; run
it as a release gate or upstream audit, not in the offline test suite.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.error
import urllib.request

SCHEMA = os.path.join(os.path.dirname(__file__), 'gen', 'ds', 'schema', 'packs.json')
TIMEOUT = 15
UA = {'User-Agent': 'oads-url-check/1.0'}


def probe(url: str) -> tuple[int | None, str]:
	"""Return (status, detail); status None means a transport failure."""
	for method, headers in (
		('HEAD', {}),
		('GET', {'Range': 'bytes=0-0'}),
	):
		try:
			req = urllib.request.Request(url, method=method, headers=headers | UA)
			with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
				return r.status, f'{method} {r.status}'
		except urllib.error.HTTPError as e:
			if method == 'HEAD' and e.code in (400, 403, 405, 501):
				continue
			return e.code, f'{method} {e.code}'
		except (urllib.error.URLError, TimeoutError, OSError) as e:
			return None, f'{method} {e.reason if isinstance(e, urllib.error.URLError) else e}'
	return None, 'unreachable'


def check_file(pack: str, entry: dict) -> dict:
	results = [(u, *probe(u)) for u in entry['urls']]
	alive = any(s is not None and 200 <= s < 400 for _, s, _ in results)
	return {'pack': pack, 'file': entry['name'], 'alive': alive, 'results': results}


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument('--packs', nargs='*', default=None, help='probe only these pack names')
	ap.add_argument('--workers', type=int, default=24)
	ap.add_argument('--verbose', action='store_true', help='print every URL result')
	ap.add_argument('--no_recheck', action='store_true',
		help='skip the serial re-probe of dead URLs')
	args = ap.parse_args()

	packs = json.load(open(SCHEMA))['packs']
	if args.packs:
		known = {p['name'] for p in packs}
		missing = [n for n in args.packs if n not in known]
		if missing:
			print(f'unknown packs: {", ".join(missing)}')
			return 2
		packs = [p for p in packs if p['name'] in args.packs]

	jobs = [(p['name'], f) for p in packs for f in p['files']]
	outcomes: list[dict] = []
	with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
		for out in ex.map(lambda j: check_file(*j), jobs):
			outcomes.append(out)
			if args.verbose:
				for url, status, detail in out['results']:
					mark = 'ok' if status is not None and 200 <= status < 400 else 'DEAD'
					print(f'{mark:4} {out["pack"]:>20} {detail:>18} {url}')

	dead = [o for o in outcomes if not o['alive']]
	# Parallel sweeps trip upstream rate limits (NIST, Caltech return 403);
	# re-probe the dead set serially before reporting it.
	if dead and not args.no_recheck:
		import time
		still_dead = []
		for o in dead:
			time.sleep(2)
			o['results'] = [(u, *probe(u)) for u, _, _ in o['results']]
			o['alive'] = any(s is not None and 200 <= s < 400 for _, s, _ in o['results'])
			if not o['alive']:
				still_dead.append(o)
			else:
				print(f'recovering {o["pack"]} {o["file"]} on serial re-probe')
		dead = still_dead
	for o in dead:
		for url, status, detail in o['results']:
			print(f'DEAD {o["pack"]:>20} {detail:>18} {url}')

	by_pack: dict[str, list[bool]] = {}
	for o in outcomes:
		by_pack.setdefault(o['pack'], []).append(o['alive'])
	dead_packs = sorted(n for n, v in by_pack.items() if not all(v))
	print(f'\n{len(outcomes)} files probed across {len(by_pack)} packs; '
		f'{len(dead)} files with no live URL in {len(dead_packs)} packs')
	if dead_packs:
		print('dead packs:', ', '.join(dead_packs))
	return 1 if dead else 0


if __name__ == '__main__':
	raise SystemExit(main())
