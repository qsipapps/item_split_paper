import io
import re
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Paper Split Demo v4", layout="wide")
st.title("PDF Paper Split Demo v4")
st.caption("Line-level paper switching for same-page multiple papers. Item rows are kept as DataFrames.")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def norm_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def compact_text(s):
    return re.sub(r"\s+", "", norm_text(s))


def safe_int(v, default=None):
    try:
        return int(float(str(v).strip().replace(",", "")))
    except Exception:
        return default


def safe_float(v, default=None):
    try:
        return float(str(v).strip().replace(",", "").replace("%", ""))
    except Exception:
        return default


def clean_sheet_name(name):
    name = re.sub(r"[\\/:*?\[\]]", "_", norm_text(name))
    return name[:31] or "Sheet"


def page_top_text(page, y_max=180):
    try:
        words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
        if not words:
            return page.extract_text() or ""
        top_words = [w for w in words if w.get("top", 9999) < y_max]
        return norm_text(" ".join(w.get("text", "") for w in top_words))
    except Exception:
        return page.extract_text() or ""


def line_tokens(line):
    return norm_text(line).split()


# ------------------------------------------------------------
# Section / paper detection
# ------------------------------------------------------------

def detect_section(text):
    t = norm_text(text)
    c = compact_text(t).lower()
    if "項目分析" in t or "itemanalysis" in c:
        return "item"
    if "多項選擇題分析" in t or "multiplechoicequestionanalysis" in c:
        return "mcq"
    if "甲類學科成績" in t or "categoryasubjectresults" in c:
        return "category"
    return None


def detect_paper_marker(text):
    """Detect paper marker anywhere in the text.

    Supports same-page switching, e.g.:
    - 卷 Paper: 1
    - 卷 Paper: 2
    - 地理 卷1A
    - Geography Paper 1A
    - Chinese Language Paper 101
    - Paper 1
    - Paper 2
    """
    if not text:
        return None
    t = norm_text(text)

    patterns = [
        r"卷\s*Paper\s*:\s*([0-9]+[A-Za-z]?)",
        r"Paper\s*:\s*([0-9]+[A-Za-z]?)",
        r"Paper\s*([0-9]+[A-Za-z]?)",
        r"卷\s*([0-9]+[A-Za-z]?)",
        r"卷\s*(1A)",
        r"Paper\s*(1A)",
        r"卷\s*(101)",
        r"Paper\s*(101)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = m.group(1)
            if str(val).upper() == "1A":
                return "Paper 1A"
            if str(val) == "101":
                return "Paper 101"
            return f"Paper {val}"

    # broader fallback, but only when explicit paper-ish token exists
    m = re.search(r"(?:卷|Paper)\s*([0-9]{1,3}[A-Za-z]?)", t, flags=re.IGNORECASE)
    if m:
        val = m.group(1)
        if str(val).upper() == "1A":
            return "Paper 1A"
        if str(val) == "101":
            return "Paper 101"
        return f"Paper {val}"

    return None


def is_item_header_line(line):
    s = norm_text(line)
    c = compact_text(s).lower()
    return ("項目分析" in s or "itemanalysis" in c)


def line_is_rowish(line):
    """Permissive row detection.

    We deliberately accept more lines and parse later.
    """
    s = norm_text(line)
    if not s:
        return False
    if is_item_header_line(s):
        return False
    if s.startswith("卷 Paper:") or s.startswith("Paper ") or s.startswith("卷"):
        return False
    if any(k in s for k in ["Your school", "Day schools", "Difference", "Answer marked", "Chart of difference"]):
        return False
    # row-like if it starts with a question/item label or contains a lot of numbers
    return bool(
        re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\)|\d+\.\d+)\b", s)
        or len(re.findall(r"\d+(?:\.\d+)?%?", s)) >= 6
    )


# ------------------------------------------------------------
# Item row parsing
# ------------------------------------------------------------

def parse_item_row(line):
    """Parse item row permissively.

    Returns dict with raw_line and best-effort columns.
    The parser is designed to work with varied SSR layouts.
    """
    s = norm_text(line)
    tokens = line_tokens(s)
    if len(tokens) < 5:
        return None

    # Common item code forms
    if re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\))$", tokens[0]):
        itemcode = tokens[0]
        rest = tokens[1:]
    elif len(tokens) > 1 and re.match(r"^Q\d+(?:\.\d+)?|Q\d+\([^)]+\)$", tokens[1]):
        itemcode = tokens[1]
        rest = tokens[2:]
    else:
        # fallback: if the line is numeric-heavy and has a first token that's not an item code,
        # keep the first token as itemcode so we do not lose rows.
        itemcode = tokens[0]
        rest = tokens[1:]

    # Try to separate label from numeric tail by locating the last 8 numeric-like tokens.
    numeric_positions = [i for i, tok in enumerate(rest) if re.fullmatch(r"[\+\-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?%?", tok)]
    if len(numeric_positions) >= 8:
        tail_start = numeric_positions[-8]
        label = " ".join(rest[:tail_start])
        nums = rest[tail_start:tail_start + 8]
    else:
        # If we cannot find enough numbers, still keep the row.
        label = " ".join(rest[:-1]) if len(rest) > 1 else ""
        nums = []

    row = {
        "itemcode": itemcode,
        "label": label,
        "raw_line": s,
        "max_mark": None,
        "your_attempted": None,
        "your_mean": None,
        "your_sd": None,
        "day_attempted": None,
        "day_mean": None,
        "day_sd": None,
        "diff": None,
        "diffpct": None,
    }

    # Best-effort field mapping
    if len(nums) >= 8:
        row["max_mark"] = safe_float(nums[0], None)
        row["your_attempted"] = safe_float(nums[1], None)
        row["your_mean"] = safe_float(nums[2], None)
        row["your_sd"] = safe_float(nums[3], None)
        row["day_attempted"] = safe_float(nums[4], None)
        row["day_mean"] = safe_float(nums[5], None)
        row["day_sd"] = safe_float(nums[6], None)
        row["diffpct"] = safe_float(nums[7], None)
        # Some PDFs place Diff before Diff%; we approximate diff from the percentage when direct diff absent.
        row["diff"] = row["diffpct"]

    return row


# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------

@st.cache_data
def extract_item_analysis_by_paper(filebytes):
    """Split item analysis into papers line-by-line.

    Key idea for same-page multi-paper PDFs:
    - Scan each line in order.
    - If a line contains a paper marker, switch current paper immediately.
    - Any subsequent row-like line belongs to that current paper until another marker appears.
    """
    paper_rows = OrderedDict()
    current_section = None
    current_paper = None
    item_mode_started = False

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            top_text = page_top_text(page, y_max=180)
            combined = f"{top_text}\n{page_text}"

            sec = detect_section(top_text) or detect_section(page_text) or current_section
            if sec:
                current_section = sec

            # Reset when entering other sections
            if current_section != "item":
                if current_section in {"mcq", "category"}:
                    current_paper = None
                    item_mode_started = False
                continue

            lines = [norm_text(x) for x in page_text.splitlines() if norm_text(x)]
            if not lines:
                continue

            # item section line-by-line scan
            for line in lines:
                # Update paper on paper-marker lines, anywhere in line.
                marker = detect_paper_marker(line)
                if marker:
                    current_paper = marker
                    item_mode_started = True
                    continue

                # Skip header / explanatory lines
                if is_item_header_line(line):
                    continue

                # If no paper yet, keep Unknown but do not drop rows.
                if current_paper is None:
                    current_paper = "Unknown Item Paper"

                if not line_is_rowish(line):
                    continue

                parsed = parse_item_row(line)
                row = {
                    "paper": current_paper,
                    "source_page": page_no,
                    "raw_line": line,
                }
                if parsed:
                    row.update(parsed)
                paper_rows.setdefault(current_paper, []).append(row)

    out = OrderedDict()
    for paper, rows in paper_rows.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df.insert(0, "rowindex", range(1, len(df) + 1))
        out[paper] = df
    return out


def merge_paper_dfs(paper_map):
    if not paper_map:
        return pd.DataFrame()
    return pd.concat(paper_map.values(), ignore_index=True)


def summary_df(paper_map):
    rows = []
    for paper, df in paper_map.items():
        pages = sorted(set(df["source_page"].tolist())) if "source_page" in df.columns else []
        rows.append({
            "paper": paper,
            "rows": len(df),
            "pages": ", ".join(map(str, pages)),
        })
    return pd.DataFrame(rows)


def to_excel_bytes(paper_map):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df(paper_map).to_excel(writer, index=False, sheet_name="Summary")
        for paper, df in paper_map.items():
            df.to_excel(writer, index=False, sheet_name=clean_sheet_name(paper))
    output.seek(0)
    return output.getvalue()


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

uploaded = st.file_uploader("Upload SSR PDF", type=["pdf"])

with st.expander("Splitting logic", expanded=True):
    st.markdown(
        """
- This version uses **line-level paper switching**.
- A line containing a paper marker like `卷 Paper: 1`, `卷 Paper: 2`, `Paper 1A`, `Paper 101` changes the current paper immediately.
- All row-like lines after that belong to the current paper until the next marker.
- Item rows are parsed best-effort into a DataFrame; raw lines are preserved.
- This is intended to handle **multiple papers on the same page**.
        """
    )

if uploaded is None:
    st.info("Upload a PDF to start.")
    st.stop()

try:
    paper_map = extract_item_analysis_by_paper(uploaded.getvalue())
    merged_df = merge_paper_dfs(paper_map)
    summary = summary_df(paper_map)

    st.success(f"Detected {len(paper_map)} item paper(s).")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Paper Summary")
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "Download Excel (multi-sheet)",
            data=to_excel_bytes(paper_map),
            file_name="paper_split_v4.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        st.subheader("Merged Item DataFrame")
        st.dataframe(merged_df, use_container_width=True)
        st.download_button(
            "Download merged CSV",
            data=merged_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="merged_item_rows.csv",
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
                file_name=f"{clean_sheet_name(paper)}.csv",
                mime="text/csv",
                key=f"csv_{clean_sheet_name(paper)}",
            )
            if "raw_line" in df.columns:
                st.caption(f"Raw lines kept: {int(df['raw_line'].notna().sum())}")

except Exception as e:
    st.error(f"❌ Error: {e}")
