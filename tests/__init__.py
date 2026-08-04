"""Marks `tests` as a package so `from tests import spec_harness` resolves.

With this file present, pytest walks up to the first directory *without* an
`__init__.py` — the repo root — and puts that on `sys.path`, which is the
behaviour both the tests and `experiments/` rely on.
"""
