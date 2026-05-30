# ESG Risk Analyzer

This project reads `companies.csv`, sends each company to an LLM (Gemini) for ESG risk analysis, validates the structured JSON response, and writes results to `esg_risk_output.csv`.

## Project Overview

Key behaviors:

- Loads company data with `pandas`.
- Builds a strict JSON-only prompt per company.
- Calls Gemini via the Google generative client.
- Extracts and parses JSON responses and validates required fields.
- Continues processing on per-company failures and logs errors.
- Saves partial results atomically so progress is durable and the run can resume.

## Installation

Create a Python virtual environment, activate it, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or `.venv\\Scripts\\Activate.ps1` on Windows PowerShell
pip install -r requirements.txt
```

## Environment

Create a `.env` file in the project root using `.env.example`:

```
GEMINI_API_KEY=your_api_key_here
```

Important: do not commit `.env` to source control. `.gitignore` already excludes it.

## Usage / Run

Basic run (serial, safe defaults):

```bash
python main.py
```

Examples with options:

- Run with 3 workers (concurrent):

```bash
python main.py --workers 3
```

- Increase retry attempts and provide fallback models (comma-separated):

```bash
python main.py --max-retries 5 --fallback-models gemini-2.5-flash,gpt-4o-mini
```

Notes:

- The script enforces a global `RATE_DELAY` between API calls and will retry on quota (429) errors using server-suggested retry delays and exponential backoff. If your Google Cloud project has limited/free-tier quota, some runs may yield partial results.
- The script will resume from an existing `esg_risk_output.csv` if present and skip rows that already contain analysis results.

## Output

The script writes `esg_risk_output.csv` with these columns:

- `company_name`
- `sector`
- `country`
- `primary_esg_risk`
- `risk_category` (one of `Environmental`, `Social`, `Governance`)
- `summary` (expected to be exactly two sentences — please ensure model obeys prompt)

If an API call fails for a company the error is logged and processing continues for remaining rows. Partial results are saved after each company.

## Submission notes

- Do not commit your `.env` or local virtual environment. Keep API keys secret and rotate them if exposed.
- For grading or demonstration where quotas are limited, include a cleaned `sample_output.csv` (mock/sanitized) rather than pushing a live `esg_risk_output.csv` produced with a private key.

## License & Acknowledgements

This repository uses `google-generativeai` (deprecated) in requirements; migrate to the newer `google.genai` client when feasible.

Questions or issues: open a GitHub issue in the target repository.