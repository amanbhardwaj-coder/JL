# app.py
# One Streamlit app: Convert inventory CSV to (1) Shopify format OR (2) Your internal VDB format
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

# =========================================================
# Shopify converter (your existing logic)
# =========================================================
def build_tags_from_master(master_row):
    orig_tags = ""
    if "Tags.1" in master_row and isinstance(master_row.get("Tags.1"), str):
        orig_tags = master_row.get("Tags.1") or ""
    elif "Tags" in master_row and isinstance(master_row.get("Tags"), str):
        orig_tags = master_row.get("Tags") or ""

    orig_tags = orig_tags.strip()
    base_list = orig_tags.split(",") if orig_tags else []
    base_list = [t for t in base_list if t != ""]

    first_tag = base_list[0].strip() if base_list else ""
    existing = {t.strip() for t in base_list}
    extras = []

    if first_tag:
        pf_tag = f"Product Family_{first_tag.replace(' ', '_')}"
        if pf_tag not in existing:
            extras.append(pf_tag)

    il_tag = "Item Location_United States"
    if il_tag not in existing:
        extras.append(il_tag)

    master_stock_num = master_row.get("Stock Number")
    if isinstance(master_stock_num, str) and master_stock_num:
        vdb_tag = f"vdb_stock_num_{master_stock_num}"
        if vdb_tag not in existing:
            extras.append(vdb_tag)

    combined = base_list + extras
    if "vdbjl" not in {t.strip() for t in combined}:
        combined.append("vdbjl")

    return ",".join(combined) if combined else np.nan

def to_shopify_df(df: pd.DataFrame, vendor_name: str, item_location: str = "United States") -> pd.DataFrame:
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

    meta_base = "metafields_global_namespace_key[single_line_text].vdbjl."
    metafield_cols = [
        meta_base + "vdb_stock_id",
        meta_base + "vdb_stock_num",
        meta_base + "type",
        meta_base + "metal",
        meta_base + "item_location",
        meta_base + "side_stone_color",
        meta_base + "side_stone_clarity",
        meta_base + "jewelry_classification",
        meta_base + "shape",
        meta_base + "weight",
        meta_base + "available_diamond_spread",
        meta_base + "available_metal_type",
        meta_base + "available_shape",
        meta_base + "customizable",
    ]

    all_cols = shopify_cols + metafield_cols
    rows_out = []

    required = ["Master Stock Number", "Stock Number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for msn, group in df.groupby("Master Stock Number"):
        group = group.copy()

        if "is_master_product" in group.columns and group["is_master_product"].any():
            group = group.sort_values("is_master_product", ascending=False).reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)

        master = group.iloc[0]

        short_title = master.get("Short Title")
        stock_number_master = master.get("Stock Number")

        base_for_handle = short_title if isinstance(short_title, str) and short_title else stock_number_master
        handle = slugify(base_for_handle)

        title = short_title if isinstance(short_title, str) and short_title else stock_number_master

        description = master.get("Description")
        body_html = f"<p>{description}</p>" if isinstance(description, str) and description else np.nan

        prod_type = "Jewelry"
        tags_str = build_tags_from_master(master)

        for idx, (_, row) in enumerate(group.iterrows()):
            out = {c: np.nan for c in all_cols}
            out["Handle"] = handle

            if idx == 0:
                out["Title"] = title
                out["Body (HTML)"] = body_html
                out["Vendor"] = vendor_name
                out["Type"] = prod_type
                out["Tags"] = tags_str
                out["Published"] = True
                out["Option1 Name"] = "Metal Type"
                out["Option2 Name"] = "Available Diamond Spread"
                out["Gift Card"] = False
                out["Status"] = "active"
            else:
                out["Title"] = np.nan
                out["Body (HTML)"] = np.nan
                out["Published"] = np.nan
                out["Status"] = np.nan

            out["Option3 Name"] = np.nan
            out["Option3 Value"] = np.nan
            out["Option3 Linked To"] = np.nan

            out["Option1 Value"] = row.get("Metal")
            out["Option2 Value"] = row.get("Diamond Spread")

            out["Variant SKU"] = row.get("Stock Number")
            out["Variant Grams"] = 0
            out["Variant Inventory Tracker"] = "shopify"
            out["Variant Inventory Qty"] = 1
            out["Variant Inventory Policy"] = "deny"
            out["Variant Fulfillment Service"] = "manual"

            price = row.get("Price")
            out["Variant Price"] = price
            out["Cost per item"] = price

            out["Variant Requires Shipping"] = True
            out["Variant Taxable"] = True
            out["Variant Weight Unit"] = "lb"

            img = row.get("Image URL 1")
            if isinstance(img, str) and img:
                out["Image Src"] = img
                out["Variant Image"] = img
                out["Image Position"] = idx + 1

                metal_slug = str(row.get("Metal") or "").strip().lower().replace(" ", "-")
                spread = str(row.get("Diamond Spread") or "").strip()
                alt_parts = []
                if metal_slug:
                    alt_parts.append(metal_slug)
                if spread:
                    alt_parts.append(spread)
                if alt_parts:
                    out["Image Alt Text"] = "-".join(alt_parts)

            if idx == 0:
                out[meta_base + "vdb_stock_id"] = np.nan
                out[meta_base + "vdb_stock_num"] = stock_number_master
                out[meta_base + "type"] = master.get("Jewelry Type")
                out[meta_base + "metal"] = master.get("Metal")
                out[meta_base + "item_location"] = item_location
                out[meta_base + "side_stone_color"] = master.get("Side Color")
                out[meta_base + "side_stone_clarity"] = master.get("Side Clarity")

                out[meta_base + "jewelry_classification"] = master.get("Jewelry Classification")
                out[meta_base + "shape"] = master.get("Shape")
                out[meta_base + "weight"] = master.get("Weight")
                out[meta_base + "available_diamond_spread"] = master.get("Available Diamond Spread")
                out[meta_base + "available_metal_type"] = master.get("Available Metal Type")
                out[meta_base + "available_shape"] = master.get("Available Shape")
                out[meta_base + "customizable"] = master.get("Customizable")

            rows_out.append(out)

    return pd.DataFrame(rows_out, columns=all_cols)

# =========================================================
# VDB internal format converter (customizable mapping)
# =========================================================
DEFAULT_VDB_OUTPUT_COLS = [
    # Replace these with your true internal schema columns
    "master_stock_number",
    "stock_number",
    "short_title",
    "jewelry_type",
    "metal",
    "diamond_spread",
    "price",
    "image_url_1",
    "side_color",
    "side_clarity",
    "shape",
    "weight",
    "customizable",
    "vendor_name",
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
    item_location = st.text_input("Item Location (Shopify metafield)", value="United States")
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

# -----------------------------
# Shopify Mode
# -----------------------------
if output_mode == "Shopify format":
    if st.button("Convert", type="primary"):
        try:
            out_df = to_shopify_df(
                df_in,
                vendor_name=vendor_name.strip(),
                item_location=item_location.strip() or "United States",
            )
            out_bytes = df_to_csv_bytes(out_df)

            st.success(f"✅ Shopify CSV generated: {len(out_df):,} rows.")
            st.subheader("Output Preview (Shopify)")
            st.dataframe(out_df.head(25), use_container_width=True)

            base_name = uploaded_file.name.rsplit(".", 1)[0]
            out_name = f"{base_name}_SHOPIFY_WITH_METAFIELDS.csv"

            st.download_button(
                "⬇️ Download Shopify CSV",
                data=out_bytes,
                file_name=out_name,
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Conversion failed: {e}")

# -----------------------------
# VDB Mode
# -----------------------------
else:
    st.subheader("VDB Format Configuration")

    cols = list(df_in.columns)
    selectable_cols = ["__BLANK__", "__STATIC_VENDOR__"] + cols

    st.markdown("### Output columns (your internal schema)")
    out_cols_text = st.text_area(
        "One column per line",
        value="\n".join(DEFAULT_VDB_OUTPUT_COLS),
        height=220,
    )
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
                out_df = to_vdb_df(
                    df_in,
                    vendor_name=vendor_name.strip(),
                    output_cols=output_cols,
                    mapping=mapping,
                )
                out_bytes = df_to_csv_bytes(out_df)

                st.success(f"✅ VDB CSV generated: {len(out_df):,} rows.")
                st.subheader("Output Preview (VDB)")
                st.dataframe(out_df.head(25), use_container_width=True)

                base_name = uploaded_file.name.rsplit(".", 1)[0]
                out_name = f"{base_name}_VDB_FORMAT.csv"

                st.download_button(
                    "⬇️ Download VDB CSV",
                    data=out_bytes,
                    file_name=out_name,
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"VDB conversion failed: {e}")

    with c2:
        st.markdown("#### Export mapping JSON")
        mapping_json = json.dumps({"output_cols": output_cols, "mapping": mapping}, indent=2)
        st.download_button(
            "Download mapping JSON",
            data=mapping_json.encode("utf-8"),
            file_name="vdb_mapping.json",
            mime="application/json",
        )

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
