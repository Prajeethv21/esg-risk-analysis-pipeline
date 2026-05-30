from __future__ import annotations

import json
import logging
import os
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import re
import random


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "companies.csv"
OUTPUT_CSV = PROJECT_ROOT / "esg_risk_output.csv"
LOG_FILE = PROJECT_ROOT / "esg_risk_analysis.log"
MODEL_NAME = "gemini-2.5-flash"
VALID_RISK_CATEGORIES = {"Environmental", "Social", "Governance"}
# Rate limiting: minimum seconds between requests (global)
RATE_DELAY = 1.0
MAX_RETRIES = 3
WAIT_UNTIL_SUCCESS = False
FALLBACK_MODELS: list[str] = []
# runtime populated model objects (primary + fallbacks)
MODELS: list[genai.GenerativeModel] = []

# Shared state for rate limiter and file writes
_last_call_time: list[float] = [0.0]
_call_time_lock = threading.Lock()
_file_lock = threading.Lock()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


def load_companies(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    companies = pd.read_csv(csv_path)
    required_columns = {"company_name", "sector", "country"}
    missing_columns = required_columns - set(companies.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    return companies


def build_prompt(company_name: str, sector: str, country: str) -> str:
    return (
        "You are an ESG analyst.\n\n"
        "Analyze the following company.\n\n"
        f"Company Name: {company_name}\n"
        f"Sector: {sector}\n"
        f"Country: {country}\n\n"
        "Identify the most significant ESG risk facing this company.\n\n"
        "Return ONLY valid JSON.\n\n"
        '{"primary_esg_risk": "", "risk_category": "", "summary": ""}\n\n'
        "Rules:\n"
        "- risk_category must be exactly one of Environmental, Social, Governance\n"
        "- summary must contain exactly 2 sentences\n"
        "- return only JSON\n"
        "- no markdown\n"
        "- no extra text"
    )


def _extract_json_text(raw_text: str) -> str:
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    start_index = cleaned_text.find("{")
    end_index = cleaned_text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return cleaned_text[start_index : end_index + 1]

    return cleaned_text


def validate_response(payload: dict[str, Any]) -> dict[str, str]:
    primary_esg_risk = str(payload.get("primary_esg_risk", "")).strip()
    risk_category = str(payload.get("risk_category", "")).strip()
    summary = str(payload.get("summary", "")).strip()

    if not primary_esg_risk:
        raise ValueError("primary_esg_risk is missing or empty")

    if not summary:
        raise ValueError("summary is missing or empty")

    if risk_category not in VALID_RISK_CATEGORIES:
        raise ValueError(
            "risk_category must be exactly one of Environmental, Social, Governance"
        )

    return {
        "primary_esg_risk": primary_esg_risk,
        "risk_category": risk_category,
        "summary": summary,
    }


def analyze_company(
    model: genai.GenerativeModel,
    company_name: str,
    sector: str,
    country: str,
) -> dict[str, str]:
    prompt = build_prompt(company_name, sector, country)

    # Try the configured models (primary + fallbacks) until one succeeds.
    last_error: Exception | None = None
    model_names_tried: list[str] = []
    for model_obj in MODELS:
        model_names_tried.append(getattr(model_obj, "name", str(model_obj)))
        attempt = 0
        while True:
            try:
                with _call_time_lock:
                    now = time.time()
                    elapsed = now - _last_call_time[0]
                    if elapsed < RATE_DELAY:
                        time.sleep(RATE_DELAY - elapsed)
                    _last_call_time[0] = time.time()

                response = model_obj.generate_content(prompt)
                last_error = None
                break  # success for this model
            except ResourceExhausted as exc:
                last_error = exc
                msg = str(exc)
                # server-suggested retry seconds
                m = re.search(r"Please retry in (\d+(?:\.\d+)?)s", msg)
                if m:
                    base_wait = float(m.group(1))
                else:
                    m2 = re.search(r"retry_delay \{\s*seconds: (\d+)", msg)
                    base_wait = float(m2.group(1)) if m2 else 1.0

                # exponential backoff with jitter
                backoff = base_wait * (2 ** attempt)
                jitter = random.uniform(0, min(5.0, backoff * 0.1))
                wait = min(600.0, backoff + jitter) + 0.5

                attempt += 1
                should_retry = attempt <= MAX_RETRIES or WAIT_UNTIL_SUCCESS
                if should_retry:
                    logging.warning("ResourceExhausted for %s on model %s; retrying in %.1fs (attempt %d/%s)", company_name, getattr(model_obj, "name", "<model>"), wait, attempt, "∞" if WAIT_UNTIL_SUCCESS else MAX_RETRIES)
                    time.sleep(wait)
                    continue
                else:
                    logging.error("Giving up on model %s for %s after %d attempts: %s", getattr(model_obj, "name", "<model>"), company_name, attempt - 1, exc)
                    break
            except Exception as exc:  # pragma: no cover - network and SDK errors are environment dependent
                raise RuntimeError(f"Gemini API call failed for {company_name}: {exc}") from exc

        if last_error is None:
            # we have a response variable from a successful call
            break
        else:
            logging.info("Model %s failed for %s; trying next model if available", getattr(model_obj, "name", "<model>"), company_name)

    if last_error is not None and not WAIT_UNTIL_SUCCESS:
        raise RuntimeError(f"All models failed for {company_name}; last error: {last_error}") from last_error

    raw_text = getattr(response, "text", "") or ""
    json_text = _extract_json_text(raw_text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON returned for {company_name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Response for {company_name} was not a JSON object")

    return validate_response(payload)


def _worker_wrapper(model: genai.GenerativeModel, idx: int, company_name: str, sector: str, country: str) -> dict[str, str]:
    try:
        return analyze_company(model, company_name, sector, country)
    except Exception:
        # analyze_company already logs details; return empty analysis so caller can continue
        return {"primary_esg_risk": "", "risk_category": "", "summary": ""}


def save_result(results: list[dict[str, str]], output_path: Path) -> None:
    output_frame = pd.DataFrame(
        results,
        columns=[
            "company_name",
            "sector",
            "country",
            "primary_esg_risk",
            "risk_category",
            "summary",
        ],
    )
    # Atomic, thread-safe write with retries for transient locks
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with _file_lock:
                # write to temp file then replace to avoid partial writes
                with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=output_path.parent, newline="", encoding="utf-8") as tmp:
                    output_frame.to_csv(tmp.name, index=False)
                os.replace(tmp.name, output_path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(1)
            else:
                break

    raise PermissionError(f"Unable to write {output_path}: the file is open in another program.") from last_error


def main() -> int:
    setup_logging()
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default 1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per model on quota errors (default 3)")
    parser.add_argument("--wait-until-success", action="store_true", help="Keep retrying until success (may block indefinitely)")
    parser.add_argument("--fallback-models", type=str, default="", help="Comma-separated fallback model names to try if primary model fails")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("Missing GEMINI_API_KEY in environment. Create a .env file from .env.example.")
        return 1

    logging.info("google-generativeai version: %s", package_version("google-generativeai"))
    logging.info("Configured model: %s", MODEL_NAME)
    logging.info("GEMINI_API_KEY loaded: %s", "yes" if api_key else "no")

    try:
        companies = load_companies(INPUT_CSV)
    except Exception as exc:
        logging.exception("Failed to load input CSV: %s", exc)
        return 1

    genai.configure(api_key=api_key)
    # build model objects: primary + optional fallbacks
    global MODELS, MAX_RETRIES, WAIT_UNTIL_SUCCESS, FALLBACK_MODELS
    MAX_RETRIES = max(1, int(args.max_retries))
    WAIT_UNTIL_SUCCESS = bool(args.wait_until_success)
    FALLBACK_MODELS = [m.strip() for m in args.fallback_models.split(",") if m.strip()]
    model_names = [MODEL_NAME] + FALLBACK_MODELS
    MODELS = [genai.GenerativeModel(n) for n in model_names]
    model = MODELS[0]

    total_companies = len(companies)
    # initialize results with placeholders to preserve order when using concurrency
    results: list[dict[str, str] | None] = [None] * total_companies

    # resume support: load existing output and skip completed rows
    completed_indices: set[int] = set()
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            # map existing rows to indices in the input companies by company_name
            name_to_index = {str(v).strip(): i for i, v in companies["company_name"].items()}
            for _, erow in existing.iterrows():
                name = str(erow.get("company_name", "")).strip()
                if not name:
                    continue
                idx = name_to_index.get(name)
                if idx is None:
                    continue
                # consider row completed if any of the analysis fields is non-empty
                if any(str(erow.get(k, "")).strip() for k in ("primary_esg_risk", "risk_category", "summary")):
                    results[idx] = {
                        "company_name": name,
                        "sector": companies.iloc[idx]["sector"],
                        "country": companies.iloc[idx]["country"],
                        "primary_esg_risk": str(erow.get("primary_esg_risk", "") ) ,
                        "risk_category": str(erow.get("risk_category", "") ),
                        "summary": str(erow.get("summary", "") ),
                    }
                    completed_indices.add(idx)
            if completed_indices:
                logging.info("Resuming run: %d rows already completed", len(completed_indices))
        except Exception as exc:
            logging.warning("Could not parse existing output for resume: %s", exc)
    else:
        # initial save to create header
        save_result([], OUTPUT_CSV)

    if args.workers <= 1:
        for index, row in companies.iterrows():
            company_name = str(row["company_name"]).strip()
            sector = str(row["sector"]).strip()
            country = str(row["country"]).strip()

            logging.info("Processing %s/%s: %s", index + 1, total_companies, company_name)
            logging.info("First request details: model=%s, company=%s, sector=%s, country=%s", MODEL_NAME, company_name, sector, country)

            analysis = {"primary_esg_risk": "", "risk_category": "", "summary": ""}
            try:
                analysis = analyze_company(model, company_name, sector, country)
            except Exception as exc:
                logging.exception("Failed to analyze %s: %s", company_name, exc)

            results[index] = {
                "company_name": company_name,
                "sector": sector,
                "country": country,
                "primary_esg_risk": analysis["primary_esg_risk"],
                "risk_category": analysis["risk_category"],
                "summary": analysis["summary"],
            }
            # save partial results so progress is durable
            save_result([r for r in results if r is not None], OUTPUT_CSV)

    else:
        workers = max(1, args.workers)
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for index, row in companies.iterrows():
                company_name = str(row["company_name"]).strip()
                sector = str(row["sector"]).strip()
                country = str(row["country"]).strip()
                logging.info("Queued %s/%s: %s", index + 1, total_companies, company_name)
                futures.append((index, ex.submit(_worker_wrapper, model, index, company_name, sector, country)))

            completed = 0
            for idx, fut in futures:
                try:
                    analysis = fut.result()
                    results[idx] = {
                        "company_name": companies.iloc[idx]["company_name"],
                        "sector": companies.iloc[idx]["sector"],
                        "country": companies.iloc[idx]["country"],
                        "primary_esg_risk": analysis.get("primary_esg_risk", ""),
                        "risk_category": analysis.get("risk_category", ""),
                        "summary": analysis.get("summary", ""),
                    }
                except Exception as exc:
                    logging.exception("Failed to analyze %s: %s", companies.iloc[idx]["company_name"], exc)
                    results[idx] = {
                        "company_name": companies.iloc[idx]["company_name"],
                        "sector": companies.iloc[idx]["sector"],
                        "country": companies.iloc[idx]["country"],
                        "primary_esg_risk": "",
                        "risk_category": "",
                        "summary": "",
                    }
                completed += 1
                logging.info("Completed %d/%d", completed, total_companies)
                save_result([r for r in results if r is not None], OUTPUT_CSV)

    logging.info("Finished processing %s companies. Output saved to %s", total_companies, OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())