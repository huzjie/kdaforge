PY ?= python3

.PHONY: doctor bench train test serve demo

doctor:
	$(PY) -m kdaforge doctor

bench:
	$(PY) -m kdaforge bench

train:
	$(PY) -m kdaforge train

test:
	$(PY) -m unittest discover -s tests -t .

serve:
	$(PY) -m kdaforge serve

demo:
	$(PY) scripts/demo.py
