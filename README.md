# AI Expense Categorizer

An end-to-end AI assistant for finance teams that takes a raw CSV of expenses, cleans it, auto‑categorizes each transaction, flags anomalies, and gives you interactive reports — plus an “Ask the AI” panel to query the dataset in plain English.[file:116][file:112]

Live app: `<< add your Streamlit Cloud URL here >>`

---

## 1. What this project does

This project is built to mirror a real finance/ops workflow: you export a CSV from your bank or accounting tool, upload it, and in a single place you get:

- Cleaned and validated data.
- Consistent, LLM‑powered categories (Travel, Software, Meals, etc.).[file:116][file:112]
- Automatic anomaly flags (high amounts, duplicates, out‑of‑pattern transactions).[file:116][file:112]
- Summary metrics and charts that are ready for review or reporting.
- Optional exports (CSV + PDF) for sharing with stakeholders.[file:112]

It is implemented in **Python + Pandas + Streamlit**, and uses **Groq’s Llama 3.3 model via the OpenAI-compatible API** for categorization and analysis, matching the stack requested in the assessment brief.[file:116][file:112]

---

## 2. How it matches the assignment

The original brief asked for four core capabilities plus a set of “advanced (optional)” features.[file:116]  
This implementation intentionally hits **all** core requirements and **all** advanced ones, plus one extra bonus capability.

### ✅ Core requirements (100%)

1. **Expense data ingestion**[file:116][file:112]  
   - Uploads CSV via Streamlit (`st.file_uploader`).  
   - Validates that `date`, `amount`, and `description` are present.  
   - Drops completely empty rows, normalizes string fields, and parses amounts and dates safely.

2. **Expense categorization**[file:116][file:112]  
   - Uses a fixed, standard category list: `Travel`, `Meals`, `Software`, `Utilities`, `Office Supplies`, `Salary`, `Miscellaneous`, `Other`.  
   - Sends batched transactions to the LLM with a structured JSON schema (`row_index`, `category`, `suspicious`, `suspicious_reason`, `confidence`).  
   - Ensures each row gets exactly one category from the allowed list; unknown merchants fall back to `Other`.

3. **Anomaly detection**[file:116][file:112]  
   - Rule‑based:
     - Flags unusually high amounts using the 95th percentile of `amount`.  
     - Flags negative amounts.  
     - Detects possible duplicates using a combination of `date + amount + merchant + description`.  
   - LLM‑based:
     - The model also returns whether a transaction looks suspicious and why.  
   - Combined flags are exposed as `is_anomaly_rule`, `anomaly_reasons_rule`, `suspicious_llm`, and `suspicious_reason_llm`.

4. **Summary report generation**[file:116][file:112]  
   - Aggregates spend by LLM category, calculates totals and percentages.  
   - Shows overall metrics (total transactions, total amount, number of anomalies).  
   - Highlights flagged anomalies in a dedicated table.  
   - Presents everything in a clean Streamlit UI with charts and tables.

5. **Error handling & reliability**[file:116][file:112]  
   - Wraps CSV reading and schema validation in `try/except` with user‑friendly error messages.  
   - Validates and parses LLM JSON output defensively.  
   - Stops the app gracefully with explanations if critical configuration (like the API key) is missing.

### ✅ Advanced capabilities (100%)

These were listed as “optional” in the brief; all are implemented here.[file:116][file:112]

1. **Monthly trend analysis**  
   - Derives a `month` column from `date` and aggregates `amount` by `month x category_llm`.  
   - Visualized with a Plotly line chart (markers on each month) so trends by category are easy to spot.

2. **Editable categories**  
   - Uses `st.data_editor` to show a live, editable table of key columns, including `category_llm`.  
   - “Apply edits & update analysis (no extra LLM)” button:
     - Applies your edits back into the main dataframe.
     - Re-runs rule‑based anomaly detection and refreshes charts, **without** re‑calling the LLM.

3. **Exportable reports (CSV / PDF)**  
   - Enriched CSV: full dataset with LLM categories, anomaly flags, and confidence scores.  
   - Edited‑view CSV: exactly what you see in the editor (useful for manual override workflows).  
   - PDF report: high‑level summary (totals, category breakdown, sample anomalies) generated via `fpdf2`.[file:112]

4. **Rule‑based + AI hybrid categorization**  
   - LLM handles semantic understanding (merchant/description → category + suspicious flag).  
   - Rule engine handles numeric heuristics (thresholds, duplicates, negatives).  
   - The UI exposes both views so a human reviewer can understand *why* something was flagged.

5. **Confidence scores per classification**  
   - Each LLM prediction includes a `confidence` between 0 and 1, surfaced as `confidence_llm` and shown in the tables/exports.[file:112]

### ⭐ Bonus: LLM “Ask the AI” interface

On top of the required features, the app includes an **interactive Q&A panel**:

- Automatically builds a compact JSON summary of the dataset (totals, top categories, sample anomalies).[file:112]  
- You can type natural‑language questions like:
  - “Which category is driving the most spend this month?”
  - “How many anomalies did you find and in which categories?”
- The LLM answers using only that summary, in 3–5 sentences, making it feel like a mini financial analyst layered on top of the structured reporting.[file:112]

This isn’t required in the assessment, but it demonstrates how the same underlying LLM can be reused for both **row‑level classification** and **dataset‑level analytics**, which is exactly the kind of pattern modern AI tools use in production.

---

## 3. Architecture at a glance

High‑level flow:

1. **Streamlit frontend (`app.py`)**[file:112]  
   - Handles file upload, user interactions, and visualizations.  
   - Manages session state for:
     - `processed_df` (the latest enriched dataframe).  
     - Q&A chat history.

2. **Data processing (Pandas)**[file:112]  
   - `load_and_clean_df`: ingestion + validation + normalization.  
   - `detect_anomalies`: rule‑based anomaly logic.  
   - Groupby operations for summaries and trends.

3. **LLM integration (Groq via OpenAI client)**[file:112]  
   - `call_groq_for_batch`: sends batched JSON payloads and parses structured JSON responses.  
   - `apply_llm_categorization`: loops through the dataframe in batches and fills in LLM columns.  
   - `ask_groq_about_data`: powers the Q&A panel over a compact dataset summary.

4. **Reporting & exports**[file:112]  
   - Plotly for interactive bar, pie, and line charts.  
   - `generate_pdf_report` for a compact, printable PDF summary.  
   - Streamlit download buttons for CSV and PDF deliverables.

---

## 4. Setup & installation

### Prerequisites

- Python 3.10+  
- A Groq API key (used via the OpenAI-compatible client)  
- Git + virtual environment (recommended)

### 1. Clone the repo

```bash
git clone https://github.com/Shivansh-707/AI-Expense_categorizer.git
cd AI-Expense_categorizer
2. Create & activate a virtual environment
bash
python -m venv .venv
source .venv/bin/activate      # on macOS / Linux
# .venv\Scripts\activate       # on Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Configure your API key
Create a .env file in the project root:

bash
GROQ_API_KEY=your_groq_api_key_here
On Streamlit Cloud, the app instead reads GROQ_API_KEY from Secrets, so the code works both locally and in deployment without changes.[file:112]

5. Running the app locally
bash
streamlit run app.py
Then open the URL shown in the terminal (usually http://localhost:8501).

6. How to use the app
Upload your CSV

Required columns: date, amount, description.

Optional columns like merchant, notes, currency are handled if present.

Run analysis

Click “Run Analysis” in the sidebar.

The app will:

Clean and validate the data.

Run rule‑based anomaly checks.

Call the LLM in batches to assign categories, suspicious flags, and confidence scores.

Review & edit

Use the “Review & Edit Categories” table to spot check and adjust categories or fields.

Click “Apply edits & update analysis (no extra LLM)” to refresh anomalies and charts based on your edits.

Explore the reports

Check the Summary section for top‑level metrics.

Use the Spending by Category and Monthly Trends charts to understand patterns.

Inspect Flagged Anomalies for outliers and potential issues.

Ask questions with AI

Scroll to “Ask the AI about your expenses”.

Type a question and hit Ask.

The app shows a running Q&A history so you can drill down with follow‑ups.

Download outputs

Download the enriched CSV, edited table CSV, and the PDF report from the Download Data section.

7. Files of interest
app.py — main Streamlit app with UI, LLM integration, anomaly detection, charts, and exports.[file:112]

main.py — utility / local runner script (if present in the repo setup).[file:115]

comprehensive_expenses.csv — sample dataset used for local testing and demo flows.[file:113]

dataset_generator.py — helper script to generate synthetic expense data for experimentation.[file:114]

AI-Expense-Categorizer.docx — original assessment brief from E2M Solutions (for reference).[file:116]
