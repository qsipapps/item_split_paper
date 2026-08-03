import io
import os
import re
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

# Optional: use the same font registration logic as your project if you want PDF export later.
# This demo focuses on upload -> paper split -> dataframe display -> Excel export.

st.set_page_config(page_title="Paper Split Demo", layout="wide")
st.title("PDF Paper Split Demo")
st.write(
    "Upload a PDF to test whether the app can detect paper boundaries and split the item table into separate DataFrames."
)


# -----------------------------
# Helpers
# -----------------------------

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def safe_int(value, default=None):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value, default=None):
    try:
        return float(str(value).strip().replace(',', '').replace('%', ''))
    except Exception:
        return default


def detect_paper_name_from_text(text: str):
    if not text:
        return None
    normalized = normalize_text(text)
    patterns = [
        r"(?:卷\s*)?(paper\s*\d+[A-Za-z]?)",
        r"(?:試卷\s*)?(paper\s*\d+[A-Za-z]?)",
        r"(?:卷\s*)?(Paper\s*\d+[A-Za-z]?)",
        r"(?:卷\s*)?(PAPER\s*\d+[A-Za-z]?)",
        r"(?:卷\s*)?(第\s*\d+\s*卷)",
    ]
    for pat in patterns:
        m = re.search(pat, normalized, flags=re.IGNORECASE)
        if m:
            paper = normalize_text(m.group(1))
            paper = paper.replace("Paper", "paper").replace("PAPER", "paper")
            return paper
    return None


def extract_paper_anchor_from_page(page):
    try:
        words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
        if words:
            top_words = [w for w in words if w.get("top", 10**9) < 140]
            top_text = " ".join(w.get("text", "") for w in top_words)
            paper = detect_paper_name_from_text(top_text)
            if paper:
                return paper
    except Exception:
        pass

    try:
        text = page.extract_text() or ""
    except Exception:
        text = ""
    return detect_paper_name_from_text(text)


ITEM_ROW_PATTERN = re.compile(
    r"^(\d+|[IVX]+)\s+(.+?)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+%?)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+%?)\s+([\d\.\-]+)\s+([\d\.\-]+)$"
)


def parse_item_row(line):
    clean = " ".join(str(line).split())
    m = ITEM_ROW_PATTERN.search(clean)
    if not m:
        return None
    return m.groups()


def extract_item_analysis_by_paper(filebytes):
    paper_rows = OrderedDict()
    current_paper = None

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_paper = extract_paper_anchor_from_page(page)
            if page_paper:
                current_paper = page_paper
            elif current_paper is None:
                current_paper = "Unknown Paper"

            for raw_line in page_text.splitlines():
                row = parse_item_row(raw_line)
                if not row:
                    continue
                if current_paper not in paper_rows:
                    paper_rows[current_paper] = []
                paper_rows[current_paper].append(
                    {
                        "paper": current_paper,
                        "item": row[1].strip(),
                        "max_mark": safe_int(row[2], 0),
                        "your_school_attm_no": safe_int(row[3], 0),
                        "your_school_attem": safe_int(row[4], 0),
                        "your_school_mean": safe_float(row[5], None),
                        "your_school_sd": safe_float(row[6], 0),
                        "day_schools_attm_no": safe_int(row[7], 0),
                        "day_schools_attem": safe_int(row[8], 0),
                        "day_schools_mean": safe_float(row[9], None),
                        "day_schools_sd": safe_float(row[10], 0),
                        "source_page": page_no,
                    }
                )

    output = OrderedDict()
    for paper_name, rows in paper_rows.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df.insert(0, "rowindex", range(1, len(df) + 1))
        output[paper_name] = df

    return output


def merge_paper_dfs(paper_map: OrderedDict):
    if not paper_map:
        return pd.DataFrame()
    return pd.concat(paper_map.values(), ignore_index=True)


def to_excel_bytes_from_paper_map(paper_map: OrderedDict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for paper_name, df in paper_map.items():
            sheet_name = re.sub(r"[\\/:*?\[\]]", "_", str(paper_name))[:31] or "Paper"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


# -----------------------------
# UI
# -----------------------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    filebytes = uploaded_file.getvalue()

    try:
        paper_map = extract_item_analysis_by_paper(filebytes)
        merged_df = merge_paper_dfs(paper_map)

        st.success(f"Detected {len(paper_map)} paper(s).")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Merged Item Table")
            st.dataframe(merged_df, use_container_width=True)
            if not merged_df.empty:
                st.download_button(
                    "Download merged Excel",
                    data=merged_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="item_merged.csv",
                    mime="text/csv",
                )
        with col2:
            st.subheader("Paper Summary")
            summary_rows = []
            for paper_name, df in paper_map.items():
                summary_rows.append({
                    "paper": paper_name,
                    "rows": len(df),
                    "pages": ", ".join(map(str, sorted(set(df.get("source_page", []))))),
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            if paper_map:
                st.download_button(
                    "Download multi-sheet Excel",
                    data=to_excel_bytes_from_paper_map(paper_map),
                    file_name="item_by_paper.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        st.divider()
        st.subheader("Per-paper DataFrames")
        for paper_name, df in paper_map.items():
            with st.expander(f"{paper_name} ({len(df)} rows)", expanded=False):
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    label=f"Download {paper_name}",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{paper_name}.csv",
                    mime="text/csv",
                    key=f"csv_{paper_name}",
                )

    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")
else:
    st.info("請先上載 PDF。")
