import io
import re
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Paper Split Demo v3", layout="wide")
st.title("PDF Paper Split Demo v3")
st.caption("Goal: reliably split papers first, then parse rows. Uses a permissive approach inspired by your pdf utils.")

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


def extract_words_text(page, y_max=180):
    try:
        words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
        if not words:
            return page.extract_text() or ""
        top_words = [w for w in words if w.get("top", 9999) < y_max]
        return norm_text(" ".join(w.get("text", "") for w in top_words))
    except Exception:
        return page.extract_text() or ""


# -------------------------------------------------------------------
# Section and paper detection
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


def detect_paper_label(text):
    """Detect paper label from header text.

    This is intentionally permissive.
    Examples handled:
    - 卷 Paper: 1
    - 卷 Paper: 2
    - 卷 Paper: 1A
    - 地理 卷1A
    - Geography Paper 1A
    - Chinese Language Paper 101
    """
    if not text:
        return None
    t = norm_text(text)
    c = compact_text(t)
    cl = c.lower()

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
            num = m.group(1)
            # normalize case and spacing
            if str(num).upper() == "1A":
                return "Paper 1A"
            if str(num) == "101":
                return "Paper 101"
            return f"Paper {num}"

    # fallback for compact headers like "地理卷1A", "ChineseLanguagePaper101"
    m = re.search(r"(?:卷|Paper)?\s*([0-9]{1,3}[A-Za-z]?)", c, flags=re.IGNORECASE)
    if m:
        num = m.group(1)
        if str(num).upper() == "1A":
            return "Paper 1A"
        if str(num) == "101":
            return "Paper 101"
        return f"Paper {num}"

    return None


def is_item_header_line(line):
    s = norm_text(line)
    c = compact_text(s).lower()
    return ("項目分析" in s or "itemanalysis" in c)


def is_likely_item_row(line):
    s = norm_text(line)
    if not s:
        return False
    # very permissive: lines starting with item/question identifiers or total rows
    return bool(
        re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\)|[A-Z]\d+|\d+\.\d+)\b", s)
        or "總分 Total" in s
        or "Total" in s and re.search(r"Q\d+", s)
    )


# -------------------------------------------------------------------
# Row parsing
# -------------------------------------------------------------------

def split_numeric_tail(parts, min_count=8):
    """Try to split a line into [lead text..., numeric tail...] using the tail of numbers.

    For item tables, the important thing is to not fail completely.
    We search for the last min_count numeric-like tokens and keep the prefix as item label.
    """
    numeric_idx = []
    for i, p in enumerate(parts):
        if re.fullmatch(r"[\+\-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?%?", p):
            numeric_idx.append(i)
    if len(numeric_idx) < min_count:
        return None
    start = numeric_idx[-min_count]
    return parts[:start], parts[start:]


def parse_item_like_line(line):
    s = norm_text(line)
    if not s:
        return None

    # Use whitespace tokenization first.
    parts = s.split()
    tail = split_numeric_tail(parts, min_count=8)
    if tail is None:
        return None
    lead, nums = tail
    if len(lead) < 2:
        return None

    # The first token of lead is usually item code.
    itemcode = lead[0]
    label = " ".join(lead[1:])

    # Common 8/9 tail layout; we keep what we can.
    # Try best effort to map from the end.
    # Some rows have 8 numeric fields, some 9 or more.
    data = {
        "itemcode": itemcode,
        "label": label,
        "raw": s,
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

    # If there are exactly 8 numeric tail tokens, map them conservatively.
    # If there are more, we use the last 8 and ignore extras.
    nums = nums[-8:]
    if len(nums) < 8:
        return data

    data["max_mark"] = safe_float(nums[0], None)
    data["your_attempted"] = safe_int(nums[1], None)
    data["your_mean"] = safe_float(nums[2], None)
    data["your_sd"] = safe_float(nums[3], None)
    data["day_attempted"] = safe_float(nums[4], None)
    data["day_mean"] = safe_float(nums[5], None)
    data["day_sd"] = safe_float(nums[6], None)
    data["diff"] = safe_float(nums[6], None)  # placeholder fallback; will improve if needed
    data["diffpct"] = safe_float(nums[7], None)

    return data


# -------------------------------------------------------------------
# Main extractor
# -------------------------------------------------------------------

@st.cache_data
def extract_item_analysis_by_paper(filebytes):
    """Permissive extractor.

    Strategy:
    1) Detect item section.
    2) Detect paper label from header area.
    3) Keep rows that look like item rows.
    4) Group rows by paper label.

    This is designed to work even when row parsing is imperfect, so you can at
    least confirm paper splitting first.
    """
    paper_rows = OrderedDict()
    current_section = None
    current_paper = None
    item_started = False

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            top_text = extract_words_text(page, y_max=180)
            full_text = f"{top_text}\n{page_text}"

            # Section switching
            sec = detect_section(top_text) or detect_section(page_text) or current_section
            if sec:
                current_section = sec

            # Paper detection from top area first, then whole page.
            label = detect_paper_label(top_text) or detect_paper_label(page_text)

            # We only care about item section here.
            if current_section != "item":
                # reset paper when leaving item section
                if current_section in {"mcq", "category"}:
                    current_paper = None
                    item_started = False
                continue

            # Once item section is entered, paper label can appear on the page.
            if label:
                # Avoid treating section headers like "Paper 101" from MCQ if we are in item section.
                # This demo assumes item paper labels are the ones visible in the item section.
                current_paper = label
                item_started = True
            else:
                # If we are in item section but paper isn't found yet, keep the previous one.
                if current_paper is None:
                    current_paper = "Unknown Item Paper"

            # Extract rows line-by-line. We also allow a page to trigger row capture once item started.
            for raw_line in page_text.splitlines():
                s = norm_text(raw_line)
                if not s:
                    continue

                # Skip obvious headers/instructions, but keep anything row-like.
                if is_item_header_line(s):
                    continue

                # Very permissive acceptance: if line looks like an item row, keep it.
                if not is_likely_item_row(s):
                    continue

                parsed = parse_item_like_line(s)
                row = {
                    "paper": current_paper,
                    "source_page": page_no,
                    "raw_line": s,
                }
                if parsed:
                    row.update(parsed)

                paper_rows.setdefault(current_paper, []).append(row)

    # Build DataFrames.
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

with st.expander("What this demo does", expanded=True):
    st.markdown(
        """
- It first tries to detect **item analysis** pages.
- It then tries to detect a paper label from the top of the page.
- Every row-like line inside item section is assigned to the current paper.
- If row parsing is imperfect, the row is still kept as a raw line so you can inspect the split.
- This is intentionally permissive because your PDFs have varying layouts.
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
            file_name="paper_split_v3.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        st.subheader("Merged Rows")
        st.dataframe(merged_df, use_container_width=True)
        st.download_button(
            "Download merged CSV",
            data=merged_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="merged_rows.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Per-paper data")
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
            raw_count = int(df["raw_line"].notna().sum()) if "raw_line" in df.columns else 0
            st.caption(f"Raw lines kept: {raw_count}")

except Exception as e:
    st.error(f"❌ Error: {e}")
