import io
import re
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Paper Split Demo v5", layout="wide")
st.title("PDF Paper Split Demo v5")
st.caption("Improved paper detection for 1B1 / 1B2 / 3B1 / 3B2 and same-page paper switches.")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Section / paper detection
# -------------------------------------------------------------------

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


def _normalize_paper_code(code):
    code = norm_text(code)
    code = code.replace("：", ":").replace("；", ";")
    code = code.strip()
    if not code:
        return None
    # Keep full paper code, including 1B1 / 1B2 / 3B1 / 3B2.
    code = re.sub(r"^Paper\s*:?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"^卷\s*Paper\s*:?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"^卷\s*", "", code)
    code = code.strip()
    if re.fullmatch(r"1A|1B1|1B2|3A|3B1|3B2|101|\d{1,3}[A-Za-z]?[0-9]?", code, flags=re.IGNORECASE):
        return f"Paper {code.upper()}"
    return f"Paper {code}"


def detect_paper_marker(text):
    """Detect paper marker robustly.

    Supports:
    - 卷 Paper: 1A
    - 卷 Paper: 1B1
    - 卷 Paper: 1B2
    - Paper 1A / Paper 1B1 / Paper 3B2
    - 卷1A / 卷1B1
    - Geography Paper 1A
    """
    if not text:
        return None
    t = norm_text(text)
    tokens = t.split()

    # 1) Token-based extraction: find the word Paper and grab the next token(s)
    #    This is the key fix for 1B1 / 1B2.
    for i, tok in enumerate(tokens):
        if tok.lower().rstrip(":") == "paper":
            # Candidate tokens after 'Paper'
            for j in range(i + 1, min(i + 4, len(tokens))):
                cand = tokens[j].strip().rstrip(":")
                cand = cand.replace("（", "").replace("）", "")
                cand = cand.replace("(", "").replace(")", "")
                # Allow forms like 1A, 1B1, 3B2, 101
                if re.fullmatch(r"[0-9]{1,3}[A-Za-z]?[0-9]?", cand, flags=re.IGNORECASE):
                    return f"Paper {cand.upper()}"
                # Sometimes the candidate might be split with punctuation in the next token.
                cand2 = re.sub(r"[^0-9A-Za-z]", "", cand)
                if re.fullmatch(r"[0-9]{1,3}[A-Za-z]?[0-9]?", cand2, flags=re.IGNORECASE):
                    return f"Paper {cand2.upper()}"

    # 2) Regex-based extraction around "Paper" / "卷 Paper:"
    patterns = [
        r"卷\s*Paper\s*:\s*([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"Paper\s*:\s*([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"Paper\s+([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"卷\s*([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"([0-9]{1,3}[A-Za-z]?[0-9]?)\s*Paper",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return _normalize_paper_code(m.group(1))

    return None


def is_item_header_line(line):
    s = norm_text(line)
    c = compact_text(s).lower()
    return ("項目分析" in s or "itemanalysis" in c)


def line_is_rowish(line):
    s = norm_text(line)
    if not s:
        return False
    if is_item_header_line(s):
        return False
    if s.startswith("卷 Paper") or s.startswith("Paper ") or s.startswith("卷"):
        return False
    if any(k in s for k in ["Your school", "Day schools", "Difference", "Answer marked", "Chart of difference"]):
        return False
    return bool(
        re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\)|\d+\.\d+)\b", s)
        or len(re.findall(r"\d+(?:\.\d+)?%?", s)) >= 6
    )


# -------------------------------------------------------------------
# Item row parsing
# -------------------------------------------------------------------

def parse_item_row(line):
    s = norm_text(line)
    tokens = line_tokens(s)
    if len(tokens) < 5:
        return None

    if re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\))$", tokens[0], flags=re.IGNORECASE):
        itemcode = tokens[0]
        rest = tokens[1:]
    elif len(tokens) > 1 and re.match(r"^Q\d+(?:\.\d+)?$|^Q\d+\([^)]+\)$", tokens[1], flags=re.IGNORECASE):
        itemcode = tokens[1]
        rest = tokens[2:]
    else:
        itemcode = tokens[0]
        rest = tokens[1:]

    numeric_positions = [i for i, tok in enumerate(rest) if re.fullmatch(r"[\+\-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?%?", tok)]
    if len(numeric_positions) >= 8:
        tail_start = numeric_positions[-8]
        label = " ".join(rest[:tail_start])
        nums = rest[tail_start:tail_start + 8]
    else:
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

    if len(nums) >= 8:
        row["max_mark"] = safe_float(nums[0], None)
        row["your_attempted"] = safe_float(nums[1], None)
        row["your_mean"] = safe_float(nums[2], None)
        row["your_sd"] = safe_float(nums[3], None)
        row["day_attempted"] = safe_float(nums[4], None)
        row["day_mean"] = safe_float(nums[5], None)
        row["day_sd"] = safe_float(nums[6], None)
        row["diffpct"] = safe_float(nums[7], None)
        row["diff"] = row["diffpct"]

    return row


# -------------------------------------------------------------------
# Main extractor
# -------------------------------------------------------------------

@st.cache_data
def extract_item_analysis_by_paper(filebytes):
    paper_rows = OrderedDict()
    current_section = None
    current_paper = None

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            top_text = page_top_text(page, y_max=180)
            sec = detect_section(top_text) or detect_section(page_text) or current_section
            if sec:
                current_section = sec

            if current_section != "item":
                if current_section in {"mcq", "category"}:
                    current_paper = None
                continue

            lines = [norm_text(x) for x in page_text.splitlines() if norm_text(x)]
            if not lines:
                continue

            for line in lines:
                marker = detect_paper_marker(line)
                if marker:
                    current_paper = marker
                    continue

                if is_item_header_line(line):
                    continue

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
        rows.append({"paper": paper, "rows": len(df), "pages": ", ".join(map(str, pages))})
    return pd.DataFrame(rows)


def to_excel_bytes(paper_map):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df(paper_map).to_excel(writer, index=False, sheet_name="Summary")
        for paper, df in paper_map.items():
            df.to_excel(writer, index=False, sheet_name=clean_sheet_name(paper))
    output.seek(0)
    return output.getvalue()


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

uploaded = st.file_uploader("Upload SSR PDF", type=["pdf"])

with st.expander("Splitting logic", expanded=True):
    st.markdown(
        """
- Item pages are scanned **line by line**.
- A line containing `Paper 1B1`, `Paper 1B2`, `Paper 3B1`, `Paper 3B2`, etc. will switch the current paper.
- The code now preserves the full paper code after `Paper`, not only the first 1-2 characters.
- Rows are grouped by the current paper into DataFrames.
- Raw lines are preserved to make debugging easier.
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
            file_name="paper_split_v5.xlsx",
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
