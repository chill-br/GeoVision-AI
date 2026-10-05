from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FINAL_VIZ_DIR = OUTPUT_DIR / "final_visualizations"

st.set_page_config(
    page_title="GeoVision-AI Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }

        .metric-label {
            color: #777;
            font-size: 0.9rem;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 650;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        .insight {
            padding: 1rem;
            border-left: 4px solid #4c78a8;
            background-color: rgba(76,120,168,0.08);
            border-radius: 6px;
            margin: 0.7rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

CLASS_NAMES = [
    "Vegetation",
    "Built-up",
    "Water",
    "Agriculture",
    "Bare land",
]

CLASS_COLORS = {
    "Vegetation": "#4daf4a",
    "Built-up": "#e41a1c",
    "Water": "#377eb8",
    "Agriculture": "#984ea3",
    "Bare land": "#ff7f00",
}


def find_file(filename):
    """Return a file path if it exists."""
    path = OUTPUT_DIR / filename

    if path.exists():
        return path

    return None


def load_csv(filename):
    path = find_file(filename)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_image(filename):
    path = FINAL_VIZ_DIR / filename

    if path.exists():
        return str(path)

    return None


def safe_mean(series):
    if series is None or len(series) == 0:
        return 0.0

    return float(series.mean())


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ GeoVision-AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Multispectral Satellite Image Semantic Segmentation Analytics
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    GeoVision-AI uses six Sentinel-2 spectral bands with a PyTorch U-Net
    to classify every pixel into five land-cover classes:
    vegetation, built-up, water, agriculture, and bare land.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Class Performance",
        "Bare-land Analysis",
        "Failure Analysis",
        "Visual Results",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Model**

    U-Net

    **Input**

    6 Sentinel-2 bands

    **Classes**

    5 land-cover classes

    **Tile size**

    256 × 256 pixels
    """
)


# ============================================================
# LOAD DATA
# ============================================================

area_df = load_csv("bareland_area_vs_iou.csv")
area_bins_df = load_csv("bareland_area_bins_summary.csv")

# Optional future files
test_analysis_df = load_csv("test_tile_bareland_analysis.csv")
training_city_df = load_csv("training_bareland_by_city.csv")
training_stats_df = load_csv("training_bareland_global_stats.csv")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Model Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pixel Accuracy", "84.51%")

    with col2:
        st.metric("Mean IoU", "68.55%")

    with col3:
        st.metric("Macro F1", "79.45%")

    with col4:
        st.metric("Test Tiles", "219")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Configuration")

        config_df = pd.DataFrame(
            {
                "Parameter": [
                    "Architecture",
                    "Input bands",
                    "Input size",
                    "Output classes",
                    "Training tiles",
                    "Test tiles",
                ],
                "Value": [
                    "U-Net",
                    "B2, B3, B4, B8, B11, B12",
                    "256 × 256",
                    "5",
                    "1,018+",
                    "219",
                ],
            }
        )

        st.dataframe(
            config_df,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("Class IoU")

        class_iou = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "IoU": [
                    0.6621,
                    0.8042,
                    0.9496,
                    0.6733,
                    0.3381,
                ],
            }
        )

        class_iou["IoU (%)"] = class_iou["IoU"] * 100

        st.bar_chart(
            class_iou.set_index("Class")["IoU (%)"],
            y_label="IoU (%)",
        )

    st.markdown("---")

    st.subheader("Key Finding")

    st.markdown(
        """
        <div class="insight">
        <b>Bare land is the primary segmentation challenge.</b><br><br>

        While water achieves approximately 95% IoU and built-up areas
        achieve approximately 80% IoU, bare land achieves only about
        34% IoU. Additional spectral and spatial analysis was therefore
        performed to understand the failure modes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Pipeline")

    st.code(
        """
Sentinel-2 imagery
        ↓
6 spectral bands
(B2, B3, B4, B8, B11, B12)
        ↓
256 × 256 tiles
        ↓
Normalization / preprocessing
        ↓
PyTorch U-Net
        ↓
5-class segmentation
        ↓
Evaluation + failure analysis
        """,
        language="text",
    )


# ============================================================
# CLASS PERFORMANCE
# ============================================================

elif page == "Class Performance":

    st.markdown(
        '<div class="section-title">Class-wise Performance</div>',
        unsafe_allow_html=True,
    )

    metrics_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Precision": [
                0.8057,
                0.9156,
                0.9582,
                0.8088,
                0.3735,
            ],
            "Recall": [
                0.7880,
                0.8686,
                0.9906,
                0.8007,
                0.7808,
            ],
            "IoU": [
                0.6621,
                0.8042,
                0.9496,
                0.6733,
                0.3381,
            ],
            "F1": [
                0.7967,
                0.8915,
                0.9741,
                0.8047,
                0.5053,
            ],
        }
    )

    display_df = metrics_df.copy()

    for column in ["Precision", "Recall", "IoU", "F1"]:
        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("IoU Comparison")

    chart_df = metrics_df.set_index("Class")["IoU"] * 100

    st.bar_chart(
        chart_df,
        y_label="IoU (%)",
    )

    st.subheader("Confusion Matrix")

    confusion = np.array(
        [
            [3567187, 232758, 60241, 574007, 92768],
            [252384, 3234746, 1856, 68450, 166486],
            [7863, 845, 2301585, 6518, 6650],
            [593498, 32947, 30231, 2758169, 29834],
            [6748, 31636, 7976, 3129, 176313],
        ]
    )

    # Row-normalized confusion matrix
    row_sums = confusion.sum(axis=1, keepdims=True)

    confusion_pct = confusion / row_sums * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(confusion_pct)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))

    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{confusion_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Percentage")

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "Bare land has high recall (78.1%) but low precision (37.4%), "
        "indicating substantial false-positive predictions."
    )


# ============================================================
# BARE LAND ANALYSIS
# ============================================================

elif page == "Bare-land Analysis":

    st.markdown(
        '<div class="section-title">Bare-land Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Bare-land IoU", "33.81%")

    with col2:
        st.metric("Bare-land Precision", "37.35%")

    with col3:
        st.metric("Bare-land Recall", "78.08%")

    with col4:
        st.metric("Bare-land F1", "50.53%")

    st.markdown("---")

    st.subheader("Bare-land Area vs IoU")

    if area_df is not None:

        required_columns = {
            "gt_bare_pixels",
            "bare_iou",
        }

        if required_columns.issubset(area_df.columns):

            plot_df = area_df[
                area_df["gt_bare_pixels"] > 0
            ].copy()

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(
                plot_df["gt_bare_pixels"],
                plot_df["bare_iou"],
                alpha=0.65,
            )

            ax.set_xscale("log")

            ax.set_xlabel(
                "Ground-truth bare-land pixels (log scale)"
            )

            ax.set_ylabel("Bare-land IoU")

            ax.set_title(
                "Relationship Between Bare-land Area and IoU"
            )

            ax.grid(alpha=0.25)

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "Larger bare-land regions generally achieve better "
                "segmentation performance."
            )

    else:

        st.warning(
            "bareland_area_vs_iou.csv was not found."
        )

    st.subheader("Area-bin Performance")

    if area_bins_df is not None:

        st.dataframe(
            area_bins_df,
            use_container_width=True,
            hide_index=True,
        )

        if "area_bin" in area_bins_df.columns and "mean_iou" in area_bins_df.columns:

            chart_df = area_bins_df[
                ["area_bin", "mean_iou"]
            ].copy()

            chart_df["mean_iou"] *= 100

            st.bar_chart(
                chart_df.set_index("area_bin")["mean_iou"],
                y_label="Mean IoU (%)",
            )

    st.markdown(
        """
        <div class="insight">
        <b>Spatial finding:</b><br>
        Bare-land area shows a moderate positive relationship with
        segmentation quality. The analysis found approximately
        <b>0.59 Spearman correlation</b> between bare-land area and IoU.
        This suggests that small or fragmented bare-land regions are
        substantially harder for the current model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Spectral Interpretation")

    spectral_df = pd.DataFrame(
        {
            "Band": [
                "B2",
                "B3",
                "B4",
                "B8",
                "B11",
                "B12",
            ],
            "Failed bare land": [
                0.094446,
                0.127867,
                0.125567,
                0.161123,
                0.166276,
                0.141276,
            ],
            "Successful bare land": [
                0.159387,
                0.213486,
                0.258617,
                0.289980,
                0.370034,
                0.375146,
            ],
        }
    )

    st.dataframe(
        spectral_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Successful bare-land regions show substantially stronger "
        "reflectance, especially in the SWIR bands B11 and B12."
    )


# ============================================================
# FAILURE ANALYSIS
# ============================================================

elif page == "Failure Analysis":

    st.markdown(
        '<div class="section-title">Failure-case Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The model does not fail uniformly. Some bare-land regions are
        segmented extremely well, while small or spectrally difficult
        regions can be missed almost completely.
        """
    )

    failure_data = pd.DataFrame(
        {
            "Test index": [
                213,
                172,
                190,
                81,
                205,
            ],
            "Tile": [
                "Chennai 19",
                "Ahmedabad 178",
                "Kolkata 39",
                "Unknown",
                "Unknown",
            ],
            "GT bare pixels": [
                102,
                301,
                184,
                145,
                330,
            ],
            "Bare IoU": [
                0.0000,
                0.0033,
                0.0045,
                0.0241,
                0.0254,
            ],
        }
    )

    display_failure = failure_data.copy()

    display_failure["Bare IoU"] = (
        display_failure["Bare IoU"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        display_failure,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Failure Cases")

    failure_images = [
        (
            "final_bareland_failure_cases.png",
            "Worst bare-land examples",
        ),
        (
            "final_bareland_success_cases.png",
            "Successful bare-land examples",
        ),
    ]

    for filename, caption in failure_images:

        image_path = load_image(filename)

        if image_path:

            st.image(
                image_path,
                caption=caption,
                use_container_width=True,
            )

    st.subheader("What the Analysis Suggests")

    st.markdown(
        """
        **1. Small regions are harder**

        Very small bare-land regions often have poor IoU.

        **2. Spectral variability matters**

        Failed and successful bare-land regions can have substantially
        different spectral characteristics.

        **3. Built-up confusion is important**

        Some bare-land regions are strongly confused with built-up areas.

        **4. Water confusion can also occur**

        In some difficult tiles, bare-land pixels are predicted as water.

        **5. Thresholding is not the main solution**

        Several failed bare-land regions have very low predicted
        bare-land probabilities, meaning the model itself is uncertain
        rather than simply using the wrong probability threshold.
        """
    )


# ============================================================
# VISUAL RESULTS
# ============================================================

elif page == "Visual Results":

    st.markdown(
        '<div class="section-title">Visual Results</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The following visualizations compare the model's predictions
        against ground-truth land-cover labels and highlight successful
        and unsuccessful bare-land segmentation.
        """
    )

    images = [
        (
            "final_area_effect_examples.png",
            "Effect of bare-land area",
        ),
        (
            "final_bareland_examples.png",
            "Bare-land examples",
        ),
        (
            "final_bareland_failure_cases.png",
            "Bare-land failure cases",
        ),
        (
            "final_bareland_success_cases.png",
            "Bare-land success cases",
        ),
    ]

    for filename, caption in images:

        image_path = load_image(filename)

        if image_path:

            st.subheader(caption)

            st.image(
                image_path,
                use_container_width=True,
            )

        else:

            st.warning(
                f"{filename} was not found in "
                f"{FINAL_VIZ_DIR}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "GeoVision-AI • Sentinel-2 Multispectral Semantic Segmentation "
    "• PyTorch U-Net"
)