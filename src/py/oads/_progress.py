"""Terminal download progress for pack fetches.

FetchProgress is the default renderer bound to `oads._native` progress
callbacks: one line per member with a bar, percentage, byte counts, speed,
elapsed time, and ETA — the tfds-style download readout. It updates in place
on a TTY and falls back to a start/done line pair on pipes and logs.
"""

import sys
import time

_WIDTH = 28


def _size(value: float) -> str:
	for unit in ("B", "KiB", "MiB", "GiB"):
		if value < 1024.0 or unit == "GiB":
			return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
		value /= 1024.0
	return f"{value:,.1f} GiB"


def _duration(seconds: float) -> str:
	seconds = max(0, int(seconds))
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	if hours:
		return f"{hours}h{minutes:02d}m"
	if minutes:
		return f"{minutes}m{seconds:02d}s"
	return f"{seconds}s"


class FetchProgress:
	"""Render fetch progress samples on a stream.

	Call the instance — `(file, fetched, total)` — or pass it as the
	`progress` argument of a loader or `pack.ensure_pack`.
	"""

	def __init__(self, stream=None):
		self._stream = stream if stream is not None else sys.stderr
		self._tty = self._stream.isatty()
		self._file = None
		self._start = 0.0
		self._last = 0.0

	def __call__(self, file: str, fetched: int, total: int) -> None:
		now = time.monotonic()
		if file != self._file:
			if self._tty and self._file is not None:
				self._stream.write("\n")
			self._file = file
			self._start = now
			self._last = now
			if not self._tty:
				self._stream.write(f"oads: downloading {file}\n")
				self._stream.flush()
		if not self._tty:
			if fetched >= total:
				self._stream.write(f"oads: {file} complete\n")
				self._stream.flush()
			return
		if now - self._last < 0.1 and fetched < total:
			return
		self._last = now
		elapsed = now - self._start
		speed = fetched / elapsed if elapsed > 0 else 0.0
		eta = (total - fetched) / speed if speed > 0 else 0.0
		ratio = min(fetched / total, 1.0) if total else 1.0
		done = int(ratio * _WIDTH)
		bar = "█" * done + "░" * (_WIDTH - done)
		self._stream.write(
			f"\r{file} {ratio * 100:5.1f}%|{bar}| "
			f"{_size(fetched)}/{_size(total)} "
			f"{_size(speed)}/s elapsed {_duration(elapsed)}"
			f" eta {_duration(eta)}   "
		)
		self._stream.flush()
		if fetched >= total:
			self._stream.write("\n")
			self._stream.flush()


def resolve(progress):
	"""Map the public `progress` argument to a native callback or None.

	`True` installs the default FetchProgress renderer, a callable is passed
	through to `oads._native`, and `None`/`False` disables reporting.
	"""
	if progress is True:
		return FetchProgress()
	if progress:
		return progress
	return None
