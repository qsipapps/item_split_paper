import io
import re
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Paper Split Demo v2", layout="wide")
st.title("PDF Paper Split Demo v2")
st.caption("Improved splitting logic based on the three sample PDFs you provided.")

# -----------------------------
# Text helpers
# -----------------------------

def norm_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def safe_int(value, default=None):
    try:
        return int(float(str(value).strip().replace(",", "")))
    except Exception:
        return default


def safe_float(value, default=None):
    try:
        return float(str(value).strip().replace(",", "").replace("%", ""))
    except Exception:
        return default


def sheet_name(text: str) -> str:
    name = re.sub(r"[\\/:*?\[\]]", "_", norm_text(text))
    return (name[:31] or "Sheet")


# -----------------------------
# Section / paper detection
# -----------------------------

ITEM_SECTION_PATTERNS = [
    r"^3\.?\s*項目分析",
    r"Item analysis",
]
MCQ_SECTION_PATTERNS = [
    r"^4\.?\s*多項選擇題分析",
    r"Multiple choice question analysis",
]
CATEGORY_SECTION_PATTERNS = [
    r"^2\.?\s*甲類學科成績",
    r"Category A subject results",
]


def detect_section(text: str):
    t = norm_text(text)
    for pat in ITEM_SECTION_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "item"
    for pat in MCQ_SECTION_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "mcq"
    for pat in CATEGORY_SECTION_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "category"
    return None


def detect_paper_label(text: str):
    """Detect paper labels robustly.

    Supports:
    - 卷 Paper: 1
    - 卷 Paper: 2
    - 卷 Paper: 1A
    - 地理 卷1A / Geography Paper 1A
    - Chinese Language Paper 101
    """
    if not text:
        return None
    t = norm_text(text)

    patterns = [
        r"(卷\s*Paper\s*:\s*[0-9]+[A-Za-z]?)",
        r"(Paper\s*[:]?\s*[0-9]+[A-Za-z]?)",
        r"(卷\s*[0-9]+[A-Za-z]?)",
        r"(Paper\s*1A)",
        r"(卷\s*1A)",
        r"(卷\s*101)",
        r"(Paper\s*101)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            label = norm_text(m.group(1))
            label = re.sub(r"\s+", " ", label)
            label = label.replace("PAPER", "Paper").replace("paper", "Paper")
            label = label.replace("卷Paper", "卷 Paper")
            label = label.replace("卷 1A", "卷1A")
            label = label.replace("卷 101", "卷101")
            return label

    # Fallbacks for titles like "地理 卷1A" or "Geography Paper 1A"
    m = re.search(r"([\u4e00-\u9fffA-Za-z\s]+?\s+(?:卷\s*)?[0-9]+[A-Za-z]?)", t)
    if m:
        cand = norm_text(m.group(1))
        if re.search(r"\b(?:卷\s*)?[0-9]+[A-Za-z]?\b", cand):
            return cand
    return None


def is_item_row(line: str):
    s = norm_text(line)
    if not s:
        return False
    # Item rows often start with item number or Q labels and contain the diff column at the end.
    return bool(
        re.match(r"^(\d+|Q\d+|Q\d+\.\d+|Q\d+\([a-z]\)|Q\d+\([a-z]\)\([ivx]+\)|Q\d+\([ivx]+\))\b", s)
        or re.match(r"^\d+\s+Q\d+", s)
    )


def parse_item_row(line: str):
    """Return dict-like tuple from a line when it matches the item table pattern.

    This parser is intentionally conservative and is meant to work with the three
    sample PDFs first, not every possible SSR variant.
    """
    s = norm_text(line)
    if not s:
        return None

    # Common layout in Chinese/Geography item pages:
    # item label, item text, max mark, your attempted, your mean, your sd, day attempted,
    # day mean, day sd, diff, diff%
    # We'll use a flexible regex that captures the last 9 numeric columns.
    m = re.match(
        r"^(?P<itemcode>(?:Q\d+(?:\.\d+)?|Q\d+\([^)]+\)|\d+))\s+(?P<label>.+?)\s+"
        r"(?P<maxmark>[\d\.\-]+)\s+(?P<your_attm>[\d\.\-]+)\s+(?P<your_mean>[\d\.\-]+)\s+"
        r"(?P<your_sd>[\d\.\-]+)\s+(?P<day_attm>[\d\.\-]+)\s+(?P<day_mean>[\d\.\-]+)\s+"
        r"(?P<day_sd>[\d\.\-]+)\s+(?P<diff>[\+\-]?[\d\.\-]+)\s+(?P<diffpct>[\+\-]?[\d\.\-]+)%?$",
        s,
    )
    if not m:
        return None
    return m.groupdict()


# -----------------------------
# Extraction
# -----------------------------

def _extract_page_top_text(page):
    try:
        words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
        if not words:
            return page.extract_text() or ""
        top_words = [w for w in words if w.get("top", 9999) < 150]
        top_text = " ".join(w.get("text", "") for w in top_words)
        return top_text or (page.extract_text() or "")
    except Exception:
        return page.extract_text() or ""


@st.cache_data
def extract_item_analysis_by_paper(filebytes):
    paper_rows = OrderedDict()
    current_section = None
    current_paper = None

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            top_text = _extract_page_top_text(page)
            combined_text = f"{top_text}\n{page_text}"

            # Section detection first
            sec = detect_section(top_text) or detect_section(page_text) or current_section
            if sec in {"item", "mcq", "category"}:
                current_section = sec

            # Paper label detection depends on section context.
            label = detect_paper_label(top_text) or detect_paper_label(page_text)

            if current_section == "item":
                # For item section, every new paper label becomes a new DF.
                if label and not re.search(r"Paper\s*101\b|卷\s*101\b", label, flags=re.IGNORECASE):
                    current_paper = label
                elif current_paper is None:
                    # If a page starts in item section but paper label is not detected,
                    # keep Unknown Item Paper so rows still get captured.
                    current_paper = "Unknown Item Paper"

            elif current_section == "mcq":
                # MCQ section gets its own paper labels like Paper 1A / Paper 101.
                if label:
                    current_paper = label
                elif current_paper is None:
                    current_paper = "Unknown MCQ Paper"

            else:
                current_paper = None

            if current_section != "item":
                continue

            for raw_line in page_text.splitlines():
                row = parse_item_row(raw_line)
                if not row:
                    continue
                paper_rows.setdefault(current_paper, []).append(
                    {
                        "paper": current_paper,
                        "itemcode": row["itemcode"],
                        "label": row["label"],
                        "max_mark": safe_int(row["maxmark"], 0),
                        "your_attempted": safe_int(row["your_attm"], None),
                        "your_mean": safe_float(row["your_mean"], None),
                        "your_sd": safe_float(row["your_sd"], None),
                        "day_attempted": safe_int(row["day_attm"], None),
                        "day_mean": safe_float(row["day_mean"], None),
                        "day_sd": safe_float(row["day_sd"], None),
                        "diff": safe_float(row["diff"], None),
                        "diffpct": safe_float(row["diffpct"], None),
                        "source_page": page_no,
                    }
                )

    out = OrderedDict()
    for paper, rows in paper_rows.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df.insert(0, "rowindex", range(1, len(df) + 1))
        out[paper] = df
    return out


def merge_paper_dfs(paper_map: OrderedDict):
    if not paper_map:
        return pd.DataFrame()
    return pd.concat(paper_map.values(), ignore_index=True)


def paper_map_summary(paper_map: OrderedDict):
    rows = []
    for paper, df in paper_map.items():
        pages = sorted(set(df["source_page"].tolist())) if "source_page" in df.columns and not df.empty else []
        rows.append({"paper": paper, "rows": len(df), "pages": ", ".join(map(str, pages))})
    return pd.DataFrame(rows)


def to_excel_bytes(paper_map: OrderedDict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary sheet
        summary_df = paper_map_summary(paper_map)
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        # Paper sheets
        for paper, df in paper_map.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name(paper))
    output.seek(0)
    return output.getvalue()


# -----------------------------
# UI
# -----------------------------

uploaded = st.file_uploader("Upload SSR PDF", type=["pdf"])

with st.expander("Splitting rules used in this demo", expanded=False):
    st.markdown(
        """
- **Item analysis** is split only inside the item section (`3. 項目分析 / Item analysis`).
- Paper labels are detected from the page header, e.g. `卷 Paper: 1`, `卷 Paper: 2`, `卷1A`.
- A new paper label starts a new DataFrame.
- Same paper across multiple pages is merged into one DataFrame.
- MCQ labels like `Paper 1A` / `Paper 101` are treated separately and are **not** mixed into item split.
- The demo keeps rows even when paper label is missing, under `Unknown Item Paper` or `Unknown MCQ Paper`.
        """
    )

if uploaded is None:
    st.info("Please upload a PDF to test paper splitting.")
    st.stop()

try:
    paper_map = extract_item_analysis_by_paper(uploaded.getvalue())
    merged = merge_paper_dfs(paper_map)
    summary = paper_map_summary(paper_map)

    st.success(f"Detected {len(paper_map)} item paper(s).")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Paper Summary")
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "Download multi-sheet Excel",
            data=to_excel_bytes(paper_map),
            file_name="paper_split_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        st.subheader("Merged Item DataFrame")
        st.dataframe(merged, use_container_width=True)
        st.download_button(
            "Download merged CSV",
            data=merged.to_csv(index=False).encode("utf-8-sig"),
            file_name="item_merged.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Per-paper DataFrames")

    for paper, df in paper_map.items():
        with st.expander(f"{paper} ({len(df)} rows)", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.download_button(
                f"Download {paper} CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sheet_name(paper)}.csv",
                mime="text/csv",
                key=f"csv_{sheet_name(paper)}",
            )

except Exception as e:
    st.error(f"❌ 發生錯誤: {e}")
