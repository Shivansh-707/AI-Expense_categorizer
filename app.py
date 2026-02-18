import os
import json

import pandas as pd
import numpy as np
import streamlit as st
import openai  # pip install openai
import plotly.express as px  # pip install plotly
from fpdf import FPDF       # pip install fpdf2

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
    "Other",
]

BATCH_SIZE = 20


# =============================
# Groq setup
# =============================

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY environment variable is not set.")
        st.stop()
    return openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


# =============================
# Backend helpers
# =============================

def load_and_clean_df(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    df = df.dropna(how="all")

    for col in ["merchant", "description", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reasons = [[] for _ in range(len(df))]

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


def build_batch_payload(df_batch: pd.DataFrame):
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


def call_groq_for_batch(client, df_batch: pd.DataFrame):
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
        model="llama-3.3-70b-versatile",
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
# LLM Q&A over dataset
# =============================

def build_dataset_summary_for_qa(df: pd.DataFrame) -> str:
    total_tx = len(df)
    total_amount = float(df["amount"].sum(skipna=True)) if "amount" in df.columns else 0.0

    cat_summary = (
        df.groupby("category_llm")["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    anomalies = df[df["is_anomaly_rule"] | df.get("suspicious_llm", False)]
    top_anomalies = anomalies.head(10)[
        ["date", "amount", "merchant", "description", "anomaly_reasons_rule", "suspicious_reason_llm"]
    ].to_dict(orient="records")

    context = {
        "total_transactions": total_tx,
        "total_amount": total_amount,
        "category_summary": cat_summary.to_dict(orient="records"),
        "num_anomalies": int(len(anomalies)),
        "sample_anomalies": top_anomalies,
    }
    return json.dumps(context, ensure_ascii=False)


def ask_groq_about_data(client, df: pd.DataFrame, question: str) -> str:
    context_json = build_dataset_summary_for_qa(df)

    messages = [
        {
            "role": "system",
            "content": """You are a senior financial analyst AI.
You are given a JSON summary of an expense dataset and user questions.
Explain clearly and concisely, referencing categories, amounts and anomalies.
Do NOT invent data beyond what is in the JSON summary."""
        },
        {
            "role": "user",
            "content": f"Here is the dataset summary (JSON):\n```json\n{context_json}\n```"
        },
        {
            "role": "user",
            "content": f"Question: {question}"
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content


# =============================
# PDF export
# =============================

def generate_pdf_report(df: pd.DataFrame) -> bytes:
    total_tx = len(df)
    total_amount = float(df["amount"].sum(skipna=True))
    anomalies = df[df["is_anomaly_rule"] | df.get("suspicious_llm", False)]
    num_anomalies = len(anomalies)

    cat_summary = (
        df.groupby("category_llm")["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Expense Categorizer Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Total transactions: {total_tx}", ln=True)
    pdf.cell(0, 8, f"Total amount: {total_amount:,.2f}", ln=True)
    pdf.cell(0, 8, f"Total anomalies: {num_anomalies}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Spending by Category:", ln=True)
    pdf.set_font("Arial", "", 12)
    for _, row in cat_summary.iterrows():
        cat = row["category_llm"]
        amt = row["amount"]
        pdf.cell(0, 8, f"- {cat}: {amt:,.2f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Sample Anomalies:", ln=True)
    pdf.set_font("Arial", "", 11)

    for _, row in anomalies.head(10).iterrows():
        line = f"{row.get('date')} | {row.get('amount'):,.2f} | {row.get('merchant', '')} | {row.get('anomaly_reasons_rule', '')}"
        pdf.multi_cell(0, 6, line)
        pdf.ln(1)

    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    else:
        return output.encode("latin1")


# =============================
# Streamlit UI
# =============================

def main():
    st.set_page_config(page_title="AI Expense Categorizer", layout="wide")

    st.title("AI Expense Categorizer")
    st.caption("Upload a CSV of expenses, categorize them with AI, flag anomalies, and ask questions about the data.")

    st.sidebar.header("Upload & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload expense CSV", type="csv")

    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None

    run_clicked = st.sidebar.button("Run Analysis", type="primary")

    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        return

    st.sidebar.write(f"**File:** {uploaded_file.name}")

    df = load_and_clean_df(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Initial run or re-run from sidebar button
    if run_clicked:
        with st.spinner("Running categorization and anomaly detection..."):
            client = get_groq_client()
            df_processed = detect_anomalies(df)
            df_processed = apply_llm_categorization(df_processed, client)
            st.session_state["processed_df"] = df_processed

    df_processed = st.session_state["processed_df"]

    if df_processed is None:
        st.warning("Click 'Run Analysis' in the sidebar to categorize and detect anomalies.")
        return

    # =============================
    # Review & Edit Categories
    # =============================
    st.subheader("Review & Edit Categories")

    display_cols = [
        "date", "amount", "merchant", "description",
        "category_llm", "anomaly_reasons_rule",
        "suspicious_llm", "confidence_llm",
    ]
    existing_cols = [c for c in display_cols if c in df_processed.columns]

    edited_view = st.data_editor(
        df_processed[existing_cols],
        key="editor",
        use_container_width=True,
    )

    col_apply, col_info = st.columns([1, 3])
    with col_apply:
        apply_edits = st.button("Apply edits & update analysis (no extra LLM)")

    if apply_edits:
        updated = df_processed.copy()
        for col in edited_view.columns:
            updated[col] = edited_view[col]
        updated = detect_anomalies(updated)
        st.session_state["processed_df"] = updated
        df_processed = updated
        st.success("Edits applied and rule-based analysis updated.")

    with col_info:
        st.caption(
            "Tip: Edit categories or fields above. "
            "Click 'Apply edits & update analysis' to refresh rules/charts without re-calling the LLM. "
            "If you just want the original enriched file, download it directly below."
        )

    # =============================
    # Summary metrics
    # =============================
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(df_processed))
    with col2:
        total_amount = df_processed["amount"].sum(skipna=True)
        st.metric("Total Amount", f"{total_amount:,.2f}")
    with col3:
        st.metric("Rule-based Anomalies", int(df_processed["is_anomaly_rule"].sum()))

    # =============================
    # Interactive charts (Plotly)
    # =============================
    if "category_llm" in df_processed.columns:
        st.markdown("#### Spending by Category (LLM)")

        cat_summary = (
            df_processed.groupby("category_llm")["amount"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        total = df_processed["amount"].sum()
        if total != 0:
            cat_summary["Percent"] = (cat_summary["amount"] / total * 100).round(1)
        else:
            cat_summary["Percent"] = 0.0

        fig_bar = px.bar(
            cat_summary,
            x="category_llm",
            y="amount",
            title="Spending by Category",
            labels={"category_llm": "Category", "amount": "Total Amount"},
            text="Percent",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            cat_summary,
            names="category_llm",
            values="amount",
            title="Category Share of Spend",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # =============================
    # Monthly trends
    # =============================
    st.subheader("Monthly Trends")

    if "date" in df_processed.columns:
        df_processed["month"] = pd.to_datetime(df_processed["date"], errors="coerce").dt.to_period("M").astype(str)
        trend_df = (
            df_processed.groupby(["month", "category_llm"])["amount"]
            .sum()
            .reset_index()
        )
        if not trend_df.empty:
            fig_trend = px.line(
                trend_df,
                x="month",
                y="amount",
                color="category_llm",
                markers=True,
                title="Monthly Spend by Category",
                labels={"amount": "Total Amount", "month": "Month", "category_llm": "Category"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.write("Not enough date information for trend analysis.")
    else:
        st.write("No 'date' column available for trend analysis.")

    # =============================
    # Anomalies table
    # =============================
    st.subheader("Flagged Anomalies")

    anomaly_mask = (df_processed["is_anomaly_rule"]) | (df_processed.get("suspicious_llm", False))
    anomalies = df_processed[anomaly_mask]

    if not anomalies.empty:
        anomaly_cols = [
            "date", "amount", "merchant", "description",
            "anomaly_reasons_rule",
            "suspicious_llm", "suspicious_reason_llm",
            "category_llm", "confidence_llm",
        ]
        existing_anomaly_cols = [c for c in anomaly_cols if c in anomalies.columns]
        st.dataframe(anomalies[existing_anomaly_cols], use_container_width=True)
    else:
        st.write("No anomalies flagged.")

    # =============================
    # Enriched / Edited data + Downloads
    # =============================
    st.subheader("Download Data")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        csv_enriched = df_processed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download enriched CSV (current analysis)",
            data=csv_enriched,
            file_name="categorized_with_anomalies_streamlit.csv",
            mime="text/csv",
        )

    with col_dl2:
        csv_edited = edited_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download edited table CSV",
            data=csv_edited,
            file_name="edited_view_only.csv",
            mime="text/csv",
        )

    with col_dl3:
        pdf_bytes = generate_pdf_report(df_processed)
        st.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name="expense_report.pdf",
            mime="application/pdf",
        )

    # =============================
    # LLM Q&A about dataset (FIXED ORDER)
    # =============================
    st.subheader("Ask the AI about your expenses")

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    # Form FIRST, handle submit, THEN render history
    with st.form("qa_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask a question about this dataset (anomalies, categories, trends, etc.)",
            key="qa_input",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and user_q.strip():
        with st.spinner("Thinking..."):
            client = get_groq_client()
            answer = ask_groq_about_data(client, df_processed, user_q.strip())
        st.session_state["qa_history"].append({"q": user_q.strip(), "a": answer})
        st.rerun()  # Rerun so new Q&A appears immediately[web:100]

    # Now render full history
    for qa in st.session_state["qa_history"]:
        st.markdown(f"**You:** {qa['q']}")
        st.markdown(f"**AI:** {qa['a']}")


if __name__ == "__main__":
    main()
