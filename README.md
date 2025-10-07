# Asset Pricing Machine
Reproducible pipeline for CAPM / Zero-Beta CAPM and mean-variance analysis.
See docs/PROMPT.md for the system prompt and /report for the paper.

## Quick Usage

```bash
# one-off self-check
python main.py health

# build processed data
python main.py ingest

# run CAPM (time-series) and cross-section
python main.py ts
python main.py cs

# run efficient frontier
python main.py frontier

# run everything end-to-end (with a health check at start and end)
python main.py all

# or via Makefile
make health
make ingest
make ts
make cs
make frontier
make all
make test
```
