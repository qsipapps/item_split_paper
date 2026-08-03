import io
import re
import unicodedata
from collections import OrderedDict

import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Paper Split Demo v12", layout="wide")
st.title("PDF Paper Split Demo v12")
st.caption("Fixes state leakage so Unknown Paper / group values do not inherit stale subjects.")

# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------

def norm_text(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def fuzzy_text(s):
    s = norm_text(s).lower()
    s = s.replace("＆", "and")
    s = s.replace("和", "")
    s = s.replace("與", "")
    s = s.replace("、", "")
    s = s.replace("，", "")
    s = s.replace(",", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("・", "")
    s = re.sub(r"[\s\-_/()]+", "", s)
    return s


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
    c = fuzzy_text(s)
    return ("項目分析" in s or "itemanalysis" in c)


# -------------------------------------------------------------------
# Group dictionary
# -------------------------------------------------------------------

GROUP_LABELS = [
    "生物 Biology",
    "企業、會計與財務概論會計 Business, Accounting and Financial Studies Accounting",
    "企業、會計與財務概論商業管理 Business, Accounting and Financial Studies Business Management",
    "化學 Chemistry",
    "中國歷史 Chinese History",
    "中國語文 Chinese Language",
    "中國文學 Chinese Literature",
    "設計與應用科技 Design and Applied Technology",
    "經濟 Economics",
    "英國語文 English Language",
    "倫理與宗教 Ethics and Religious Studies",
    "地理 Geography",
    "健康管理與社會關懷 Health Management and Social Care",
    "歷史 History",
    "資訊及通訊科技 Information and Communication Technology",
    "英語文學 Literature in English",
    "數學必修部分 Mathematics Compulsory Part",
    "數學延伸部分(微積分與統計) Mathematics Extended Part(Calculus and Statistics)",
    "數學延伸部分(代數與微積分) Mathematics Extended Part(Algebra and Calculus)",
    "音樂 Music",
    "體育 Physical Education",
    "物理 Physics",
    "科技與生活-食品科學與科技 Food Science and Technology",
    "科技與生活-服裝、成衣與紡織 Fashion,Clothing and Textiles",
    "旅遊與款待 Tourism and Hospitality Studies",
    "視覺藝術 Visual Arts",
]


def group_aliases(label):
    alias_map = {
        "生物 Biology": ["biology", "生物"],
        "化學 Chemistry": ["chemistry", "化學"],
        "中國歷史 Chinese History": ["chinesehistory", "中國歷史"],
        "中國語文 Chinese Language": ["chineselanguage", "中國語文"],
        "中國文學 Chinese Literature": ["chineseliterature", "中國文學"],
        "設計與應用科技 Design and Applied Technology": ["designandappliedtechnology", "設計與應用科技", "dat"],
        "經濟 Economics": ["economics", "經濟"],
        "英國語文 English Language": ["englishlanguage", "英國語文"],
        "倫理與宗教 Ethics and Religious Studies": ["ethicsandreligiousstudies", "倫理與宗教"],
        "地理 Geography": ["geography", "地理"],
        "健康管理與社會關懷 Health Management and Social Care": ["healthmanagementandsocialcare", "健康管理與社會關懷"],
        "歷史 History": ["history", "歷史"],
        "資訊及通訊科技 Information and Communication Technology": ["informationandcommunicationtechnology", "資訊及通訊科技", "ict"],
        "英語文學 Literature in English": ["literatureinenglish", "英語文學"],
        "數學必修部分 Mathematics Compulsory Part": ["mathematicscompulsorypart", "數學必修部分", "mathscompulsorypart", "mathcompulsorypart"],
        "數學延伸部分(微積分與統計) Mathematics Extended Part(Calculus and Statistics)": [
            "mathematicsextendedpartcalculusandstatistics",
            "數學延伸部分微積分與統計",
            "mathsextendedpartcalculusandstatistics",
            "extendedpartcalculusandstatistics",
            "calculusandstatistics",
        ],
        "數學延伸部分(代數與微積分) Mathematics Extended Part(Algebra and Calculus)": [
            "mathematicsextendedpartalgebraandcalculus",
            "數學延伸部分代數與微積分",
            "mathsextendedpartalgebraandcalculus",
            "extendedpartalgebraandcalculus",
            "algebraandcalculus",
        ],
        "音樂 Music": ["music", "音樂"],
        "體育 Physical Education": ["physicaleducation", "體育"],
        "物理 Physics": ["physics", "物理"],
        "科技與生活-食品科學與科技 Food Science and Technology": ["foodscienceandtechnology", "科技與生活食品科學與科技", "foodscience"],
        "科技與生活-服裝、成衣與紡織 Fashion,Clothing and Textiles": ["fashionclothingandtextiles", "科技與生活服裝成衣與紡織", "clothingandtextiles"],
        "旅遊與款待 Tourism and Hospitality Studies": ["tourismandhospitalitystudies", "旅遊與款待"],
        "視覺藝術 Visual Arts": ["visualarts", "視覺藝術"],
        "企業、會計與財務概論會計 Business, Accounting and Financial Studies Accounting": ["businessaccountingandfinancialstudiesaccounting", "企業會計與財務概論會計", "bafsaccounting"],
        "企業、會計與財務概論商業管理 Business, Accounting and Financial Studies Business Management": ["businessaccountingandfinancialstudiesbusinessmanagement", "企業會計與財務概論商業管理", "bafsbusinessmanagement"],
    }
    return list(dict.fromkeys(alias_map.get(label, [label.split(" ")[0], label])))


GROUP_ALIASES = [(label, [fuzzy_text(a) for a in group_aliases(label)]) for label in GROUP_LABELS]


def detect_group_marker(text):
    t = norm_text(text)
    if not t:
        return None
    ft = fuzzy_text(t)
    for label, aliases in GROUP_ALIASES:
        for alias in aliases:
            if alias and alias in ft:
                return label
    return None


def detect_section(text):
    t = norm_text(text)
    ft = fuzzy_text(t)
    if "項目分析" in t or "itemanalysis" in ft:
        return "item"
    if "多項選擇題分析" in t or "multiplechoicequestionanalysis" in ft:
        return "mcq"
    if "甲類學科成績" in t or "categoryasubjectresults" in ft:
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
    for pat in [r"卷\s*Paper\s*:\s*([0-9]{1,3}[A-Za-z]?[0-9]?)", r"Paper\s*:\s*([0-9]{1,3}[A-Za-z]?[0-9]?)", r"Paper\s+([0-9]{1,3}[A-Za-z]?[0-9]?)", r"卷\s*([0-9]{1,3}[A-Za-z]?[0-9]?)"]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return f"Paper {m.group(1).upper()}"
    return None


def derive_group_from_text(page_text):
    ft = fuzzy_text(page_text)
    for label, aliases in GROUP_ALIASES:
        if any(alias in ft for alias in aliases if alias):
            return label
    return None


def choose_group(page_text, current_group, reset=False):
    if reset:
        current_group = None
    detected = detect_group_marker(page_text)
    if detected:
        return detected
    if current_group:
        return current_group
    return derive_group_from_text(page_text) or "Unknown Group"


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
    return bool(re.match(r"^(?:\d+|Q\d+(?:\.\d+)?|Q\d+\([^)]+\)|\d+\.\d+)\b", s) or len(re.findall(r"\d+(?:\.\d+)?%?", s)) >= 6)


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

    row = {"itemcode": itemcode, "label": label, "raw_line": s, "max_mark": None, "your_attempted": None, "your_mean": None, "your_sd": None, "day_attempted": None, "day_mean": None, "day_sd": None, "diffpct": None}
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

    with pdfplumber.open(io.BytesIO(filebytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            top_text = page_top_text(page, y_max=180)
            page_blob = top_text + " " + page_text

            sec = detect_section(top_text) or detect_section(page_text) or current_section
            if sec and sec != current_section:
                current_section = sec
                if current_section != "item":
                    current_group = None
                    current_paper = None

            if current_section != "item":
                continue

            lines = [norm_text(x) for x in page_text.splitlines() if norm_text(x)]
            if not lines:
                continue

            page_group = detect_group_marker(page_blob)
            if page_group:
                current_group = page_group

            page_has_paper_marker = any(detect_paper_marker(line) for line in lines)
            if page_has_paper_marker and current_group is None:
                current_group = choose_group(page_blob, current_group, reset=True)

            for line in lines:
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
                    current_group = choose_group(page_blob, current_group, reset=True)
                else:
                    # If the current group is stale and page blob contains another explicit group, refresh it.
                    detected = detect_group_marker(page_blob)
                    if detected and detected != current_group:
                        current_group = detected

                if not line_is_rowish(line):
                    continue

                parsed = parse_item_row(line)
                key = f"{current_group} | {current_paper}"
                row = {"group": current_group, "paper": current_paper, "paper_key": key, "source_page": page_no, "raw_line": line}
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
    return out


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
- v12 resets group state when entering item analysis sections.
- Unknown Paper no longer inherits stale subjects.
- The page blob is rescanned to refresh the group when the header appears on the same page.
- Final key remains `group | paper`.
        """
    )

if uploaded is None:
    st.info("Upload a PDF to start.")
    st.stop()

try:
    paper_map = extract_item_analysis(uploaded.getvalue())
    summary = summary_df(paper_map)

    st.success(f"Detected {len(paper_map)} paper group(s).")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)
        st.download_button("Download Excel (multi-sheet)", data=to_excel_bytes(paper_map), file_name="paper_split_v12.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        merged_df = pd.concat(paper_map.values(), ignore_index=True) if paper_map else pd.DataFrame()
        st.subheader("Merged Item DataFrame")
        st.dataframe(merged_df, use_container_width=True)
        st.download_button("Download merged CSV", data=merged_df.to_csv(index=False).encode("utf-8-sig"), file_name="merged_item_rows.csv", mime="text/csv")

    st.divider()
    st.subheader("Per-paper DataFrames")
    for key, df in paper_map.items():
        with st.expander(f"{key} ({len(df)} rows)", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.download_button(f"Download {key} CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{clean_sheet_name(key)}.csv", mime="text/csv", key=f"csv_{clean_sheet_name(key)}")
            if "raw_line" in df.columns:
                st.caption(f"Raw lines kept: {int(df['raw_line'].notna().sum())}")

except Exception as e:
    st.error(f"❌ Error: {e}")
