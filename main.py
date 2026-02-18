import os
import sys
import json
from typing import List, Dict

import pandas as pd
import numpy as np
import openai  # pip install openai

from dotenv import load_dotenv
load_dotenv()

# =============================
# Config
# =============================

REQUIRED_COLUMNS = ["date", "amount", "description"]

CATEGORY_LIST = [
    "Travel",
    "Meals",
    "Software",
    "Utilities",
    "Office Supplies",
    "Salary",
    "Miscellaneous",
    "Other"
]

BATCH_SIZE = 20


# =============================
# Groq setup
# =============================

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


# =============================
# CSV ingestion & cleaning
# =============================

def load_and_clean_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.dropna(how="all")

    for col in ["merchant", "description", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# =============================
# Rule-based anomaly detection
# =============================

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reasons: List[List[str]] = [[] for _ in range(len(df))]

    if df["amount"].notna().sum() > 0:
        high_thresh = df["amount"].quantile(0.95)
    else:
        high_thresh = np.nan

    for i, row in df.iterrows():
        if pd.notna(row["amount"]) and pd.notna(high_thresh) and row["amount"] > high_thresh:
            reasons[i].append(f"Unusually high amount (> {high_thresh:.2f})")

        if pd.notna(row["amount"]) and row["amount"] < 0:
            reasons[i].append("Negative amount (possible refund or data issue)")

    subset_cols = [c for c in ["date", "amount", "merchant", "description"] if c in df.columns]
    if subset_cols:
        dup_mask = df.duplicated(subset=subset_cols, keep=False)
        for i, is_dup in enumerate(dup_mask):
            if is_dup:
                reasons[i].append("Possible duplicate transaction")

    df["anomaly_reasons_rule"] = [", ".join(r) if r else "" for r in reasons]
    df["is_anomaly_rule"] = df["anomaly_reasons_rule"] != ""
    return df


# =============================
# LLM categorization helpers
# =============================

def build_batch_payload(df_batch: pd.DataFrame) -> List[Dict]:
    records = []
    for idx, row in df_batch.iterrows():
        records.append(
            {
                "row_index": int(idx),
                "date": str(row.get("date", "")),
                "amount": float(row["amount"]) if pd.notna(row.get("amount")) else None,
                "merchant": str(row.get("merchant", "")),
                "description": str(row.get("description", "")),
                "currency": str(row.get("currency", "")),
            }
        )
    return records


def call_groq_for_batch(client, df_batch: pd.DataFrame) -> Dict[int, Dict]:
    payload = build_batch_payload(df_batch)
    payload_json = json.dumps(payload, ensure_ascii=False)

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant helping a finance team categorize business expenses.

For EACH transaction, you MUST:
- Assign exactly one category from this list: {CATEGORY_LIST}
- Decide whether it looks suspicious (true/false)
- Provide a short reason
- Provide a confidence score between 0 and 1

Return ONLY a JSON object with a "results" array containing objects:
{{
  "results": [
    {{
      "row_index": <int>,
      "category": <string from the allowed list>,
      "suspicious": <true or false>,
      "suspicious_reason": <string>,
      "confidence": <float between 0 and 1>
    }}
  ]
}}"""
        },
        {"role": "user", "content": f"Transactions:\n```json\n{payload_json}\n```"}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # NEW - current version
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1
    )

    text = response.choices[0].message.content
    parsed = json.loads(text)
    
    result = {}
    results_list = parsed.get("results", [])
    for item in results_list:
        row_index = int(item["row_index"])
        result[row_index] = {
            "category": item.get("category", "Other"),
            "suspicious_llm": bool(item.get("suspicious", False)),
            "suspicious_reason_llm": item.get("suspicious_reason", ""),
            "confidence_llm": float(item.get("confidence", 0.0)),
        }
    return result


def apply_llm_categorization(df: pd.DataFrame, client) -> pd.DataFrame:
    df = df.copy()
    df["category_llm"] = ""
    df["suspicious_llm"] = False
    df["suspicious_reason_llm"] = ""
    df["confidence_llm"] = np.nan

    n = len(df)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = df.iloc[start:end]
        mapping = call_groq_for_batch(client, batch)

        for row_index, info in mapping.items():
            if row_index in df.index:
                df.at[row_index, "category_llm"] = info["category"]
                df.at[row_index, "suspicious_llm"] = info["suspicious_llm"]
                df.at[row_index, "suspicious_reason_llm"] = info["suspicious_reason_llm"]
                df.at[row_index, "confidence_llm"] = info["confidence_llm"]

    return df


# =============================
# Summary printing
# =============================

def print_summary(df: pd.DataFrame):
    print("\n=== Basic Summary ===")
    print(f"Total transactions: {len(df)}")

    if df["amount"].notna().sum() > 0:
        total_spend = df["amount"].sum()
        print(f"Total amount: {total_spend:.2f}")

    if "category_llm" in df.columns:
        print("\nSpending by LLM category:")
        cat_summary = (
            df.groupby("category_llm")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        total = df["amount"].sum()
        for cat, amt in cat_summary.items():
            pct = (amt / total * 100) if total != 0 else 0
            print(f"  - {cat}: {amt:.2f} ({pct:.1f}%)")

    if "is_anomaly_rule" in df.columns:
        n_rule = df["is_anomaly_rule"].sum()
        print(f"\nRule-based anomalies flagged: {n_rule}")

    if "suspicious_llm" in df.columns:
        n_llm = df["suspicious_llm"].sum()
        print(f"LLM-suspicious transactions: {n_llm}")


# =============================
# Main
# =============================

def main(csv_path: str):
    print(f"Loading CSV from: {csv_path}")
    df = load_and_clean_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    print("First 5 rows:")
    print(df.head())

    df = detect_anomalies(df)

    client = get_groq_client()
    df = apply_llm_categorization(df, client)

    print_summary(df)

    out_path = "categorized_with_anomalies.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved enriched CSV to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_expenses_csv>")
        sys.exit(1)
    main(sys.argv[1])
