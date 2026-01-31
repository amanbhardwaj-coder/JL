# app.py
# Run: streamlit run app.py

import io
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# Helpers
# =========================================================
def slugify(text):
    if pd.isna(text) or text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def safe_get(row, col):
    if col is None or col == "" or col == "__BLANK__":
        return np.nan
    return row.get(col, np.nan)

def is_valid_image(url):
    if not isinstance(url, str):
        return False
    u = url.strip().lower()
    if not u.startswith("http"):
        return False
    return re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", u) is not None

def normalize_metal_for_option(metal: str) -> str:
    m = metal.strip()
    m = re.sub(r"\b(\d{2})\s*KT\b", r"\1Kt", m, flags=re.IGNORECASE)
    m = re.sub(r"\b(\d{2})\s*K\b", r"\1K", m, flags=re.IGNORECASE)
    return m

def normalize_metal_for_tag(metal: str) -> str:
    m = metal.strip()
    m = re.sub(r"\b(\d{2})\s*KT\b", r"\1 Kt", m, flags=re.IGNORECASE)
    m = re.sub(r"\b(\d{2})\s*K\b", r"\1 K", m, flags=re.IGNORECASE)
    return m

# --- category tag cleanup ---
CATEGORY_TAGS = {
    "ring","rings","engagement ring","engagement rings",
    "necklace","necklaces",
    "earring","earrings",
    "bracelet","bracelets",
    "pendant","pendants",
    "bangle","bangles",
}

def type_to_allowed_categories(prod_type: str):
    t = (prod_type or "").lower()
    allowed = set()
    if "ring" in t:
        allowed.update(["ring","rings"])
        if "engagement" in t:
            allowed.update(["engagement ring","engagement rings"])
    if "necklace" in t:
        allowed.update(["necklace","necklaces"])
    if "earring" in t:
        allowed.update(["earring","earrings"])
    if "bracelet" in t:
        allowed.update(["bracelet","bracelets"])
    if "pendant" in t:
        allowed.update(["pendant","pendants"])
    if "bangle" in t:
        allowed.update(["bangle","bangles"])
    return allowed

# =========================================================
# Styles from SAME file (NEW)
# =========================================================
STYLE_COLS = ["Style", "Style1", "Style2", "Style3"]

def extract_styles_from_row(row) -> list[str]:
    styles = []
    for c in STYLE_COLS:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            styles.append(v.strip())
    # de-dupe keep order
    out = []
    seen = set()
    for s in styles:
        k = s.lower()
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out

def style_multiline(styles: list[str]):
    return "\n".join(styles) if styles else np.nan

# =========================================================
# Shopify output builder
# =========================================================
def pick_type(master_row):
    jc = master_row.get("Jewelry Classification")
    if isinstance(jc, str) and jc.strip():
        return jc.strip()
    jt = master_row.get("Jewelry Type")
    if isinstance(jt, str) and jt.strip():
        return jt.strip()
    return "Jewelry"

def build_body_html(master_row, styles: list[str]):
    desc = master_row.get("Description")
    desc_txt = str(desc).strip() if isinstance(desc, str) and desc.strip() else ""

    side_clarity = master_row.get("Side Clarity")
    side_color = master_row.get("Side Color")

    details = []

    # styles in Details (only if present)
    for s in styles:
        details.append(f"<p><strong>Style</strong> - {s}</p>")

    if isinstance(side_clarity, str) and side_clarity.strip():
        details.append(f"<p><strong>Side Stone Clarity</strong> - {side_clarity.strip()}</p>")

    if isinstance(side_color, str) and side_color.strip():
        details.append(f"<p><strong>Side Stone Color</strong> - {side_color.strip()}</p>")

    out = []
    if desc_txt:
        out.append(f"<p>{desc_txt}</p>")
    if details:
        out.append("<hr><h3>Details</h3>" + "".join(details))

    return "".join(out) if out else np.nan

def build_tags(master_row, prod_type: str, styles: list[str], item_location="United States"):
    tags = []

    # keep existing tags in file
    orig = master_row.get("Tags")
    if isinstance(orig, str) and orig.strip():
        tags.extend([t.strip() for t in orig.split(",") if t.strip()])

    # --- remove WRONG category tags based on Type ---
    allowed = type_to_allowed_categories(prod_type)
    cleaned = []
    for t in tags:
        tl = t.lower().strip()
        if tl in CATEGORY_TAGS and tl not in allowed:
            continue
        cleaned.append(t)
    tags = cleaned

    # --- add correct category tags (so collections work) ---
    for cat in sorted(allowed):
        if not any(x.lower().strip() == cat for x in tags):
            tags.append(cat.title())

    # ensure standard tags
    if not any(t.lower().startswith("item location_") for t in tags):
        tags.append(f"Item Location_{item_location}")

    metal = master_row.get("Metal")
    if isinstance(metal, str) and metal.strip():
        if not any(t.lower().startswith("metal_") for t in tags):
            tags.append(f"Metal_{normalize_metal_for_tag(metal)}")

    sc = master_row.get("Side Clarity")
    if isinstance(sc, str) and sc.strip():
        if not any(t.lower().startswith("side stone clarity_") for t in tags):
            tags.append(f"Side Stone Clarity_{sc.strip()}")

    scol = master_row.get("Side Color")
    if isinstance(scol, str) and scol.strip():
        if not any(t.lower().startswith("side stone color_") for t in tags):
            tags.append(f"Side Stone Color_{scol.strip()}")

    jt = master_row.get("Jewelry Type")
    if isinstance(jt, str) and jt.strip():
        tags.append(f"Type_{jt.strip()}")

    jc = master_row.get("Jewelry Classification")
    if isinstance(jc, str) and jc.strip():
        tags.append(f"Type_{jc.strip()}")

    # add style tags only if present in file
    for s in styles:
        if not any(t.lower().strip() == s.lower() for t in tags):
            tags.append(s)

    # ensure VDBJL
    if "VDBJL" not in {t.upper() for t in tags}:
        tags.append("VDBJL")

    # de-dupe (case-insensitive) keep order
    out = []
    seen = set()
    for t in tags:
        k = t.lower().strip()
        if k not in seen:
            out.append(t.strip())
            seen.add(k)

    return ", ".join(out) if out else np.nan

def to_shopify_df(df: pd.DataFrame, vendor_name: str, item_location: str = "United States") -> pd.DataFrame:
    # minimal Shopify columns (your exact base + Style metafield)
    shopify_cols = [
        "Handle","Title","Body (HTML)","Vendor","Product Category","Type","Tags","Published",
        "Option1 Name","Option1 Value","Option1 Linked To",
        "Option2 Name","Option2 Value","Option2 Linked To",
        "Option3 Name","Option3 Value","Option3 Linked To",
        "Variant SKU","Variant Grams","Variant Inventory Tracker","Variant Inventory Qty",
        "Variant Inventory Policy","Variant Fulfillment Service",
        "Variant Price","Variant Compare At Price","Variant Requires Shipping","Variant Taxable",
        "Unit Price Total Measure","Unit Price Total Measure Unit","Unit Price Base Measure","Unit Price Base Measure Unit",
        "Variant Barcode","Image Src","Image Position","Image Alt Text","Gift Card","SEO Title","SEO Description",
        "Style (product.metafields.custom.style)",
        "Complementary products (product.metafields.shopify--discovery--product_recommendation.complementary_products)",
        "Related products (product.metafields.shopify--discovery--product_recommendation.related_products)",
        "Related products settings (product.metafields.shopify--discovery--product_recommendation.related_products_display)",
        "Search product boosts (product.metafields.shopify--discovery--product_search_boost.queries)",
        "Variant Image","Variant Weight Unit","Variant Tax Code","Cost per item","Status",
    ]

    required = ["Master Stock Number", "Stock Number", "Metal", "Image URL 1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows_out = []

    for msn, group in df.groupby("Master Stock Number", dropna=False):
        group = group.copy()

        # master row
        if "is_master_product" in group.columns and group["is_master_product"].fillna(False).any():
            master = group.loc[group["is_master_product"].fillna(False)].iloc[0]
            variants = group.loc[~group["is_master_product"].fillna(False)].copy()
        else:
            master = group.iloc[0]
            variants = group.copy()

        if len(variants) == 0:
            variants = pd.DataFrame([master])

        # keep first SKU row as the “main” row
        variants = variants.drop_duplicates(subset=["Stock Number"], keep="first").reset_index(drop=True)
        first_variant = variants.iloc[0]

        # remove product if no valid primary image
        if not is_valid_image(first_variant.get("Image URL 1")):
            continue

        title = str(master.get("Master Stock Number")).strip()  # SKU as title
        handle = slugify(title)

        prod_type = pick_type(master)
        styles = extract_styles_from_row(master)  # NEW: from same file
        tags = build_tags(master, prod_type=prod_type, styles=styles, item_location=item_location)
        body_html = build_body_html(master, styles)

        metal = master.get("Metal")
        metal_option = normalize_metal_for_option(metal) if isinstance(metal, str) and metal.strip() else np.nan

        # MAIN ROW
        out = {c: np.nan for c in shopify_cols}
        out["Handle"] = handle
        out["Title"] = title
        out["Body (HTML)"] = body_html
        out["Vendor"] = vendor_name
        out["Type"] = prod_type
        out["Tags"] = tags
        out["Published"] = True

        # Metal filter option
        out["Option1 Name"] = "Metal Type"
        out["Option1 Value"] = metal_option

        out["Variant SKU"] = str(first_variant.get("Stock Number")).strip()
        out["Variant Grams"] = 0
        out["Variant Inventory Tracker"] = "shopify"
        out["Variant Inventory Qty"] = 1
        out["Variant Inventory Policy"] = "deny"
        out["Variant Fulfillment Service"] = "manual"

        price = first_variant.get("Price")
        out["Variant Price"] = price
        out["Cost per item"] = price
        out["Variant Requires Shipping"] = True
        out["Variant Taxable"] = True
        out["Variant Weight Unit"] = "lb"
        out["Gift Card"] = False
        out["Status"] = "active"

        img1 = first_variant.get("Image URL 1")
        out["Image Src"] = img1
        out["Variant Image"] = img1
        out["Image Position"] = 1
        out["Image Alt Text"] = handle

        # Style metafield ONLY if present in file
        out["Style (product.metafields.custom.style)"] = style_multiline(styles)

        rows_out.append(out)

        # extra images as additional rows
        pos = 2
        for col in ["Image URL 2", "Image URL 3", "Image URL 4"]:
            u = first_variant.get(col)
            if is_valid_image(u):
                rr = {c: np.nan for c in shopify_cols}
                rr["Handle"] = handle
                rr["Image Src"] = u
                rr["Image Position"] = pos
                rr["Image Alt Text"] = handle
                rows_out.append(rr)
                pos += 1

    return pd.DataFrame(rows_out, columns=shopify_cols)

# =========================================================
# VDB internal format converter (unchanged)
# =========================================================
DEFAULT_VDB_OUTPUT_COLS = [
    "master_stock_number","stock_number","short_title","jewelry_type","metal","diamond_spread","price",
    "image_url_1","side_color","side_clarity","shape","weight","customizable","vendor_name",
]

DEFAULT_VDB_MAPPING = {
    "master_stock_number": "Master Stock Number",
    "stock_number": "Stock Number",
    "short_title": "Short Title",
    "jewelry_type": "Jewelry Type",
    "metal": "Metal",
    "diamond_spread": "Diamond Spread",
    "price": "Price",
    "image_url_1": "Image URL 1",
    "side_color": "Side Color",
    "side_clarity": "Side Clarity",
    "shape": "Shape",
    "weight": "Weight",
    "customizable": "Customizable",
    "vendor_name": "__STATIC_VENDOR__",
}

def to_vdb_df(df: pd.DataFrame, vendor_name: str, output_cols: list[str], mapping: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        out = {}
        for out_col in output_cols:
            in_col = mapping.get(out_col, "__BLANK__")
            if in_col == "__STATIC_VENDOR__":
                out[out_col] = vendor_name
            else:
                out[out_col] = safe_get(row, in_col)
        rows.append(out)
    return pd.DataFrame(rows, columns=output_cols)

# =========================================================
# Streamlit App
# =========================================================
st.set_page_config(page_title="Inventory Converter", layout="wide")
st.title("Inventory Converter (One App)")
st.caption("Upload your inventory CSV → choose output format (Shopify or VDB) → download converted CSV.")

with st.sidebar:
    st.header("Vendor Info")
    vendor_name = st.text_input("Vendor Name", value="Perfect Love Inventory")
    item_location = st.text_input("Item Location", value="United States")
    st.markdown("---")
    output_mode = st.selectbox("Select Output Format", ["Shopify format", "VDB format"], index=0)

uploaded_file = st.file_uploader("Upload Combined Inventory CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df_in = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Input Preview")
st.write(f"Rows: {len(df_in):,} | Columns: {len(df_in.columns):,}")
st.dataframe(df_in.head(25), use_container_width=True)

if not vendor_name or not vendor_name.strip():
    st.warning("Please enter a Vendor Name in the sidebar.")
    st.stop()

if output_mode == "Shopify format":
    if st.button("Convert", type="primary"):
        try:
            out_df = to_shopify_df(df_in, vendor_name=vendor_name.strip(), item_location=item_location.strip() or "United States")
            out_bytes = df_to_csv_bytes(out_df)

            st.success(f"✅ Shopify CSV generated: {len(out_df):,} rows.")
            st.subheader("Output Preview (Shopify)")
            st.dataframe(out_df.head(25), use_container_width=True)

            base_name = uploaded_file.name.rsplit(".", 1)[0]
            out_name = f"{base_name}_SHOPIFY.csv"

            st.download_button("⬇️ Download Shopify CSV", data=out_bytes, file_name=out_name, mime="text/csv")
        except Exception as e:
            st.error(f"Conversion failed: {e}")

else:
    st.subheader("VDB Format Configuration")

    cols = list(df_in.columns)
    selectable_cols = ["__BLANK__", "__STATIC_VENDOR__"] + cols

    st.markdown("### Output columns (your internal schema)")
    out_cols_text = st.text_area("One column per line", value="\n".join(DEFAULT_VDB_OUTPUT_COLS), height=220)
    output_cols = [c.strip() for c in out_cols_text.splitlines() if c.strip()]

    st.markdown("### Mapping (output → input)")
    st.caption("Map each VDB output column to a source column, or choose __STATIC_VENDOR__ / __BLANK__.")

    mapping = {}
    left, right = st.columns(2)
    for i, out_col in enumerate(output_cols):
        default_in = DEFAULT_VDB_MAPPING.get(out_col, "__BLANK__")
        box = left if i % 2 == 0 else right
        with box:
            mapping[out_col] = st.selectbox(
                out_col,
                options=selectable_cols,
                index=selectable_cols.index(default_in) if default_in in selectable_cols else 0,
                key=f"map_{out_col}",
            )

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("Convert", type="primary"):
            try:
                out_df = to_vdb_df(df_in, vendor_name=vendor_name.strip(), output_cols=output_cols, mapping=mapping)
                out_bytes = df_to_csv_bytes(out_df)

                st.success(f"✅ VDB CSV generated: {len(out_df):,} rows.")
                st.subheader("Output Preview (VDB)")
                st.dataframe(out_df.head(25), use_container_width=True)

                base_name = uploaded_file.name.rsplit(".", 1)[0]
                out_name = f"{base_name}_VDB_FORMAT.csv"

                st.download_button("⬇️ Download VDB CSV", data=out_bytes, file_name=out_name, mime="text/csv")
            except Exception as e:
                st.error(f"VDB conversion failed: {e}")

    with c2:
        st.markdown("#### Export mapping JSON")
        mapping_json = json.dumps({"output_cols": output_cols, "mapping": mapping}, indent=2)
        st.download_button("Download mapping JSON", data=mapping_json.encode("utf-8"), file_name="vdb_mapping.json", mime="application/json")

    with c3:
        st.markdown("#### Import mapping JSON")
        uploaded_mapping = st.file_uploader("Upload mapping JSON", type=["json"], key="mapping_upload")
        if uploaded_mapping is not None:
            try:
                loaded = json.loads(uploaded_mapping.read().decode("utf-8"))
                st.info("Loaded mapping JSON (shown below). For now, paste its output_cols into the text box and re-select mappings.")
                st.code(json.dumps(loaded, indent=2), language="json")
            except Exception as e:
                st.error(f"Could not read mapping JSON: {e}")
