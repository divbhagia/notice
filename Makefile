# Define shell
SHELL := /bin/bash

# Python path
export PYTHONPATH := $(shell pwd)

# Manuscript name
MANUSCRIPT := figures_and_tables

# Scripts for main program
SCRIPTS := $(shell ls ./scripts/*.py)
SCRIPTS := $(filter-out ./scripts/00_init.py, $(SCRIPTS))
SCRIPTS := $(filter-out ./scripts/01_clean_data.py, $(SCRIPTS))

# Default target
all: init_data mainprog manuscript

# Remove output and data folders
clean: 
	rm -r ./output/
	rm -r ./data/
	echo "Removed output and data folders."

# Initialize, download and process raw data
init_data:
	echo "Downloading raw data..."
	python ./scripts/00_init.py
	echo "Processing raw data..."
	python ./scripts/01_clean_data.py
	echo "Data processing done."

# Run scripts for the main program (log saved to output/mainprog.log)
mainprog:
	mkdir -p ./output
	for script in $(SCRIPTS); do \
		echo $$script; \
		python $$script; \
	done 2>&1 | tee ./output/mainprog.log

# Compile manuscript
manuscript:
	cd ./draft; \
	pdflatex $(MANUSCRIPT).tex; \
	bibtex $(MANUSCRIPT); \
	pdflatex $(MANUSCRIPT).tex; \
	pdflatex $(MANUSCRIPT).tex; \
	rm -f $(MANUSCRIPT).aux $(MANUSCRIPT).bbl $(MANUSCRIPT).blg $(MANUSCRIPT).log $(MANUSCRIPT).out $(MANUSCRIPT).toc