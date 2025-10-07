.PHONY: health ingest ts cs frontier all test

health:
	python main.py health

ingest:
	python main.py ingest

ts:
	python main.py ts

cs:
	python main.py cs

frontier:
	python main.py frontier

all:
	python main.py all

test:
	pytest
