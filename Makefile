.PHONY: health ingest ts cs frontier validate dashboard all test

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

validate:
	python main.py validate

dashboard:
	python main.py dashboard

all:
	python main.py all

test:
	pytest
