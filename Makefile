SHELL := /bin/bash

SHEET ?=
TAB ?= Sheet1
BASE_DIR ?= .
IMAGES_DIR ?= web_low_res
RADIAL_KEYWORD ?= A praying monk (earthenware toy).
DR_INPUT ?= english/english_mpnet_embedding_matrix.csv
DR_OUTPUT ?= english/tsne_coords.csv
DR_METHOD ?= tsne
DR_PLOT ?= english/tsne_projection.pdf

.PHONY: setup similarity reorder enrich mst cluster radial dr pipeline

setup:
	python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dr]"

similarity:
	@test -n "$(SHEET)" || (echo "SHEET is required" && exit 1)
	sw-similarity --sheet "$(SHEET)" --sheet-tab "$(TAB)" --output-dir "$(BASE_DIR)"

reorder:
	sw-reorder --base-dir "$(BASE_DIR)" --method average

enrich:
	@test -n "$(SHEET)" || (echo "SHEET is required" && exit 1)
	sw-enrich --sheet "$(SHEET)" --sheet-tab "$(TAB)" --base-dir "$(BASE_DIR)"

mst:
	sw-mst --input english/english_4454.csv --images-dir "$(IMAGES_DIR)" --output english/english_mst.pdf

cluster:
	sw-cluster --input english/english_4454.csv --output english/hierarchical_clustering.pdf

radial:
	sw-radial --input english/english_4454.csv --keyword "$(RADIAL_KEYWORD)" --output radial.json

dr:
	sw-dr --input "$(DR_INPUT)" --method "$(DR_METHOD)" --output "$(DR_OUTPUT)" --plot "$(DR_PLOT)"

pipeline: similarity dr reorder enrich mst cluster radial
