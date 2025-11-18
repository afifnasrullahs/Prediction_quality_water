from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.config import AppConfig
from app.core.container import ApplicationContainer
from app.core.recommendation import RecommendationResult


CACHE_VERSION = "groq-v1"


@st.cache_resource(show_spinner=False)
def get_container(_cache_buster: str = CACHE_VERSION) -> ApplicationContainer:
    return ApplicationContainer.build()


def list_panels() -> List[str]:
    return get_container().data_repository.list_panels()


@st.cache_data(show_spinner=False)
def get_summary(panel: str) -> Dict[str, object]:
    dataset = get_container().data_repository.get_dataset(panel)
    summary = dataset.get_summary()
    total_rows = len(dataset.df)
    filtered_rows = len(dataset.filtered_df())
    return {
        "feature_summary": summary.feature_summary,
        "sample_rows": summary.sample_rows,
        "label_distribution_before": summary.label_distribution_before,
        "label_distribution_after": summary.label_distribution_after,
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
    }


@st.cache_resource(show_spinner=False)
def get_best_panel() -> str:
    return get_container().get_best_model().panel


def run_prediction(payload: Dict[str, float]) -> Dict[str, object]:
    model = get_container().get_best_model()
    label = model.predict_label(payload)
    accuracy = model.evaluate().accuracy
    return {"panel": model.panel, "label": label, "accuracy": accuracy}


def run_recommendation(payload: Dict[str, float], label: str) -> RecommendationResult:
    return get_container().recommender.generate(payload, label)


def parse_recommendation_text(text: str) -> Dict[str, str]:
    interpretation = ""
    recommendation = ""
    lines = text.splitlines()

    def _next_non_empty(start_idx: int) -> str:
        for idx in range(start_idx, len(lines)):
            candidate = lines[idx].strip()
            if candidate:
                return candidate
        return ""

    for idx, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("interpretasi:"):
            value = stripped.split(":", 1)[1].strip()
            if not value:
                value = _next_non_empty(idx + 1)
            interpretation = value
        elif lower.startswith("rekomendasi:"):
            value = stripped.split(":", 1)[1].strip()
            if not value:
                value = _next_non_empty(idx + 1)
            recommendation = value

    return {
        "interpretasi": interpretation,
        "rekomendasi": recommendation,
        "raw": text,
    }


def render_data_page():
    st.header("Data & Info")
    all_panels = list_panels()
    panels = [panel for panel in all_panels if panel.lower().startswith("panel")]
    if not panels:
        panels = all_panels
    summaries = {panel: get_summary(panel) for panel in panels}

    st.subheader("Ringkasan Panel")
    metric_cols = st.columns(len(panels))
    for col, panel in zip(metric_cols, panels):
        summary = summaries[panel]
        filtered = summary["filtered_rows"]
        col.metric(
            label=f"Panel {panel.upper()}",
            value=f"{filtered:,}",
        )

    st.subheader("Ringkasan Fitur")
    feature_cols = st.columns(len(panels))
    for col, panel in zip(feature_cols, panels):
        with col:
            st.markdown(f"**Panel {panel.upper()}**")
            st.dataframe(
                summaries[panel]["feature_summary"],
                use_container_width=True,
            )

    st.subheader("Contoh Data")
    sample_cols = st.columns(len(panels))
    for col, panel in zip(sample_cols, panels):
        with col:
            st.markdown(f"**Panel {panel.upper()}**")
            st.dataframe(
                summaries[panel]["sample_rows"],
                use_container_width=True,
            )

    st.divider()
    st.subheader("Distribusi Label: Sebelum vs Sesudah")
    panel_before = panels[0]
    panel_after = panels[-1]

    combined_records = []
    for label, count in summaries[panel_before]["label_distribution_before"].items():
        combined_records.append(
            {"panel": f"{panel_before.upper()} - Sebelum", "label": label, "count": count}
        )
    for label, count in summaries[panel_after]["label_distribution_after"].items():
        combined_records.append(
            {"panel": f"{panel_after.upper()} - Sesudah", "label": label, "count": count}
        )

    if combined_records:
        combined_df = pd.DataFrame(combined_records)
        fig_distribution = px.bar(
            combined_df,
            x="label",
            y="count",
            color="panel",
            barmode="group",
        )
        st.plotly_chart(fig_distribution, use_container_width=True)


def render_model_page():
    st.header("Model & Evaluasi")
    performances = AppConfig.MODEL_PERFORMANCE

    if not performances:
        st.info("Belum ada data performa model.")
        return

    panel_tabs = st.tabs([f"Panel {panel.upper()}" for panel in performances])
    for tab, (panel, models) in zip(panel_tabs, performances.items()):
        with tab:
            st.subheader(f"Kinerja Model Panel {panel.upper()}")
            rows = []
            for model_name, scores in models.items():
                rows.append(
                    {
                        "Model": model_name,
                        "Train Accuracy": f"{scores['train']*100:.2f}%",
                        "Test Accuracy": f"{scores['test']*100:.2f}%",
                    }
                )
            df = pd.DataFrame(rows).set_index("Model")
            st.table(df)

            best_model_name, best_scores = max(
                models.items(), key=lambda item: item[1]["test"]
            )
            st.success(
                f"Model terbaik berdasarkan Test Accuracy: {best_model_name} "
                f"({best_scores['test']*100:.2f}%)"
            )

            st.divider()
            st.subheader("Confusion Matrix")
            matrices = AppConfig.CONFUSION_MATRIX_IMAGES.get(panel, {})
            if not matrices:
                st.info("Belum ada confusion matrix untuk panel ini.")
            else:
                cols = st.columns(len(matrices))
                for col, (model_name, path) in zip(cols, matrices.items()):
                    with col:
                        st.caption(model_name)
                        if path.exists():
                            st.image(
                                str(path),
                                caption=model_name,
                                use_column_width=True,
                            )
                        else:
                            st.warning(f"Gambar tidak ditemukan: {path}")


def render_prediction_page():
    st.header("Simulasi Prediksi")
    st.caption("Masukkan pembacaan sensor terbaru untuk melihat kualitas air dan rekomendasi AI.")

    with st.form("prediction-form"):
        col1, col2 = st.columns(2)
        with col1:
            flow1 = st.number_input("Flow1", value=0.0, step=0.1)
            flow2 = st.number_input("Flow2", value=0.0, step=0.1)
            turbidity = st.number_input("Turbidity", value=5.0, step=0.1)
        with col2:
            tds = st.number_input("TDS", value=500.0, step=1.0)
            ph = st.number_input("pH", value=7.0, step=0.1)
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        payload = {
            "flow1": flow1,
            "flow2": flow2,
            "turbidity": turbidity,
            "tds": tds,
            "ph": ph,
        }
        try:
            prediction = run_prediction(payload)
        except Exception as exc:
            st.error(f"Gagal melakukan prediksi: {exc}")
            return

        st.success(
            f"Prediksi kualitas air: **{prediction['label']}** "
            f"(model panel {prediction['panel'].upper()}, accuracy {prediction['accuracy']*100:.2f}%)"
        )

        try:
            with st.spinner("Menghubungi Recommendation AI..."):
                recommendation = run_recommendation(payload, prediction["label"])
        except Exception as exc:
            st.warning(f"Rekomendasi tidak tersedia: {exc}")
            return

        st.subheader("Recommendation AI")
        parsed = parse_recommendation_text(recommendation.response)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Interpretasi")
                st.write(parsed["interpretasi"] or "-")
            with col2:
                st.markdown("##### Rekomendasi")
                st.write(parsed["rekomendasi"] or "-")
        if not (parsed["interpretasi"] or parsed["rekomendasi"]):
            st.caption("Teks asli:")
            st.info(parsed["raw"])


def main():
    st.set_page_config(page_title="Water Quality Monitoring", layout="wide")
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman", ("Data & Info", "Model & Evaluasi", "Simulasi Prediksi")
    )

    if page == "Data & Info":
        render_data_page()
    elif page == "Model & Evaluasi":
        render_model_page()
    else:
        render_prediction_page()


if __name__ == "__main__":
    main()


