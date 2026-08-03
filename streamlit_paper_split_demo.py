import io
import re
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Paper Split Demo v7", layout="wide")
st.title("PDF Paper Split Demo v7")
st.caption("Auto-detects repeated paper codes with different section headers and splits by group + paper when needed.")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def norm_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def compact_text(s):
    return re.sub(r"\s+", "", norm_text(s))


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


def is_item_header_line(line):
    s = norm_text(line)
    c = compact_text(s).lower()
    return ("項目分析" in s or "itemanalysis" in c)


# -------------------------------------------------------------------
# Section / paper / group detection
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


def detect_paper_marker(text):
    if not text:
        return None
    t = norm_text(text)
    tokens = t.split()
    for i, tok in enumerate(tokens):
        if tok.lower().rstrip(":") == "paper":
            for j in range(i + 1, min(i + 4, len(tokens))):
                cand = re.sub(r"[^0-9A-Za-z]", "", tokens[j])
                if re.fullmatch(r"[0-9]{1,3}[A-Za-z]?[0-9]?", cand, flags=re.IGNORECASE):
                    return f"Paper {cand.upper()}"
    patterns = [
        r"卷\s*Paper\s*:\s*([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"Paper\s*:\s*([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"Paper\s+([0-9]{1,3}[A-Za-z]?[0-9]?)",
        r"卷\s*([0-9]{1,3}[A-Za-z]?[0-9]?)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return f"Paper {m.group(1).upper()}"
    return None


def detect_group_marker(text):
    t = norm_text(text)
    if not t:
        return None
    patterns = [
        r"(數學必修部分)",
        r"(數學延伸部分[（(]微積分與統計[）)])",
        r"(數學延伸部分[（(]代數與微積分[）)])",
        r"(Maths\s*Core)",
        r"(Extended\s*Maths\s*[\-–—]\s*Calculus\s*and\s*Statistics)",
        r"(Extended\s*Maths\s*[\-–—]\s*Algebra\s*and\s*Calculus)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return norm_text(m.group(1))
    return None


def default_group_for_paper(paper):
    p = norm_text(paper).upper()
    if p in {"PAPER 1", "PAPER 1A", "PAPER 1B1", "PAPER 1B2"}:
        return "Paper 1"
    return ""


# -------------------------------------------------------------------
# Intelligent mode detection
# -------------------------------------------------------------------

def detect_group_mode(pdf):
    section_headers = []
    paper_keys = set()
    paper_to_groups = {}

    for page in pdf.pages[: min(6, len(pdf.pages))]:
        page_text = page.extract_text() or ""
        top_text = page_top_text(page, y_max=180)
        texts = [top_text, page_text]
        current_group = None
        for txt in texts:
            grp = detect_group_marker(txt)
            if grp:
                current_group = grp
                section_headers.append(grp)
            p = detect_paper_marker(txt)
            if p:
                paper_keys.add(p)
                if current_group:
                    paper_to_groups.setdefault(p, set()).add(current_group)

    repeated_papers = any(len(groups) > 1 for groups in paper_to_groups.values())
    explicit_groups = len(set(section_headers)) >= 2
    repeated_same_paper = len(paper_keys) < len(section_headers) and len(paper_keys) > 0
    return bool(explicit_groups or repeated_papers or repeated_same_paper), sorted(set(section_headers))


# -------------------------------------------------------------------
# Row parsing
# -------------------------------------------------------------------

def line_is_rowish(line):
    s = norm_text(line)
    if not s or is_item_header_line(s):
        return False
    if s.startswith("卷 Paper") or s.startswith("Paper ") or s.startswith("卷"):
        return False
    if any(k in s for k in ["Your school", "Day schools", "Difference", "Chart of difference"]):
        return False
    return bool(
        re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\)|\d+\.\d+)\b", s)
        or len(re.findall(r"\d+(?:\.\d+)?%?", s)) >= 6
    )


def parse_item_row(line):
    s = norm_text(line)
    tokens = s.split()
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

    return row


# -------------------------------------------------------------------
# Main extractor
# -------------------------------------------------------------------

@st.cache_data
def extract_item_analysis(filebytes):
    rows_by_key = OrderedDict()
    current_section = None
    current_group = None
    current_paper = None
    group_mode = False

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        group_mode, detected_groups = detect_group_mode(pdf)
        st.session_state["detected_groups"] = detected_groups

        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            top_text = page_top_text(page, y_max=180)
            sec = detect_section(top_text) or detect_section(page_text) or current_section
            if sec:
                current_section = sec

            if current_section != "item":
                if current_section in {"mcq", "category"}:
                    current_group = None
                    current_paper = None
                continue

            lines = [norm_text(x) for x in page_text.splitlines() if norm_text(x)]
            if not lines:
                continue

            if group_mode:
                page_group = detect_group_marker(top_text) or detect_group_marker(page_text)
                if page_group:
                    current_group = page_group

            for line in lines:
                if group_mode:
                    marker_group = detect_group_marker(line)
                    if marker_group:
                        current_group = marker_group
                        continue

                marker_paper = detect_paper_marker(line)
                if marker_paper:
                    current_paper = marker_paper
                    continue

                if is_item_header_line(line):
                    continue

                if current_paper is None:
                    current_paper = "Unknown Paper"
                if current_group is None:
                    current_group = default_group_for_paper(current_paper) or "Unknown Group"

                if not line_is_rowish(line):
                    continue

                parsed = parse_item_row(line)
                key = f"{current_group} | {current_paper}" if group_mode else current_paper
                row = {
                    "group_mode": group_mode,
                    "group": current_group if group_mode else "",
                    "paper": current_paper,
                    "paper_key": key,
                    "source_page": page_no,
                    "raw_line": line,
                }
                if parsed:
                    row.update(parsed)
                rows_by_key.setdefault(key, []).append(row)

    out = OrderedDict()
    for key, rows in rows_by_key.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df.insert(0, "rowindex", range(1, len(df) + 1))
        out[key] = df
    return out, bool(group_mode), st.session_state.get("detected_groups", [])


def summary_df(paper_map):
    rows = []
    for key, df in paper_map.items():
        pages = sorted(set(df["source_page"].tolist())) if "source_page" in df.columns else []
        rows.append({"paper_key": key, "rows": len(df), "pages": ", ".join(map(str, pages))})
    return pd.DataFrame(rows)


def to_excel_bytes(paper_map):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df(paper_map).to_excel(writer, index=False, sheet_name="Summary")
        for key, df in paper_map.items():
            df.to_excel(writer, index=False, sheet_name=clean_sheet_name(key))
    output.seek(0)
    return output.getvalue()


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

uploaded = st.file_uploader("Upload SSR PDF", type=["pdf"])

with st.expander("Splitting logic", expanded=True):
    st.markdown(
        """
- The app first scans early pages to decide whether **group mode** should be used.
- Group mode is enabled when it finds repeated paper codes with different section/group headers.
- If group mode is enabled, the key becomes `group | paper`.
- If not, it falls back to `paper only`.
        """
    )

if uploaded is None:
    st.info("Upload a PDF to start.")
    st.stop()

try:
    paper_map, group_mode, detected_groups = extract_item_analysis(uploaded.getvalue())
    summary = summary_df(paper_map)

    st.success(f"Detected {len(paper_map)} paper group(s). Group mode: {'ON' if group_mode else 'OFF'}")
    if detected_groups:
        st.caption("Detected group headers: " + ", ".join(detected_groups))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "Download Excel (multi-sheet)",
            data=to_excel_bytes(paper_map),
            file_name="paper_split_v7.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        merged_df = pd.concat(paper_map.values(), ignore_index=True) if paper_map else pd.DataFrame()
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
    for key, df in paper_map.items():
        with st.expander(f"{key} ({len(df)} rows)", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.download_button(
                f"Download {key} CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{clean_sheet_name(key)}.csv",
                mime="text/csv",
                key=f"csv_{clean_sheet_name(key)}",
            )
            if "raw_line" in df.columns:
                st.caption(f"Raw lines kept: {int(df['raw_line'].notna().sum())}")

except Exception as e:
    st.error(f"❌ Error: {e}")
