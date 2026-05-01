"""
AI User Interview Analyzer — Streamlit Dashboard
Extracts themes, pain points, and opportunity signals from user interview transcripts.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import InterviewAnalyzer

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
      page_title="AI User Interview Analyzer",
      page_icon="🎙️",
      layout="wide"
)

st.title("🎙️ AI User Interview Analyzer")
st.markdown(
      "*Turn raw interview transcripts into product insights — automatically.*"
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
      st.header("⚙️ Settings")
      n_themes = st.slider("Number of themes to extract", 3, 8, 5)
      min_sentiment_threshold = st.slider(
          "Pain point threshold (sentiment)", -1.0, 0.0, -0.2, step=0.1
      )
      st.markdown("---")
      st.markdown("**How to use:**")
      st.markdown("1. Upload a `.txt` transcript or paste text below")
      st.markdown("2. Click **Analyze** to extract themes")
      st.markdown("3. Review themes, pain points & opportunities")

# ─── Input ─────────────────────────────────────────────────────────────────────
st.header("📄 Input Transcript")

input_method = st.radio(
      "Input method", ["Paste text", "Upload .txt file"], horizontal=True
)

transcript_text = ""

if input_method == "Paste text":
      transcript_text = st.text_area(
                "Paste your interview transcript here",
                height=200,
                placeholder=(
                              "I: How do you currently manage your projects?\n"
                              "P: Mostly in spreadsheets. We tried the app but onboarding was confusing...\n"
                ),
      )
else:
      uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])
      if uploaded_file:
                transcript_text = uploaded_file.read().decode("utf-8")
                st.success(f"✅ Loaded: {uploaded_file.name} ({len(transcript_text)} characters)")

  # ─── Analyze ───────────────────────────────────────────────────────────────────
  if st.button("🔍 Analyze Interview", type="primary", disabled=not transcript_text):
        analyzer = InterviewAnalyzer(n_themes=n_themes, pain_threshold=min_sentiment_threshold)

    with st.spinner("Analyzing transcript..."):
              results = analyzer.analyze(transcript_text)

    st.success(
              f"✅ Analysis complete — {results['sentence_count']} sentences processed, "
              f"{results['theme_count']} themes extracted"
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Themes", "🔥 Pain Points", "💡 Opportunities"])

    with tab1:
              st.subheader("🏷️ Recurring Themes")
              theme_df = pd.DataFrame(results["themes"])
              st.dataframe(theme_df, use_container_width=True)

        # Sentiment bar chart
              fig, ax = plt.subplots(figsize=(8, 4))
              colors = [
                  "#e74c3c" if s < -0.2 else "#2ecc71" if s > 0.2 else "#f39c12"
                  for s in theme_df["avg_sentiment"]
              ]
              ax.barh(theme_df["theme_label"], theme_df["avg_sentiment"], color=colors)
              ax.axvline(0, color="white", linewidth=0.5)
              ax.set_xlabel("Average Sentiment Score")
              ax.set_title("Theme Sentiment Distribution")
              ax.set_facecolor("#0e1117")
              fig.patch.set_facecolor("#0e1117")
              ax.tick_params(colors="white")
              ax.xaxis.label.set_color("white")
              ax.title.set_color("white")
              st.pyplot(fig)

    with tab2:
              st.subheader("🔥 Pain Points Detected")
              if results["pain_points"]:
                            for i, pain in enumerate(results["pain_points"], 1):
                                              with st.expander(f"Pain #{i} — Sentiment: {pain['sentiment']:.2f}"):
                                                                    st.write(f"**Quote:** *\"{pain['sentence']}\"*")
                                                                    st.write(f"**Theme:** {pain['theme']}")
              else:
                            st.info("No strong pain points detected with current threshold.")

          with tab3:
                    st.subheader("💡 Positive Signals & Opportunities")
                    if results["opportunities"]:
                                  for i, opp in enumerate(results["opportunities"], 1):
                                                    with st.expander(f"Opportunity #{i} — Sentiment: {opp['sentiment']:.2f}"):
                                                                          st.write(f"**Quote:** *\"{opp['sentence']}\"*")
                                                                          st.write(f"**Theme:** {opp['theme']}")
                    else:
                                  st.info("No strong positive signals detected.")

                # ── PM Summary ───────────────────────────────────────────────────────────
                st.markdown("---")
    st.subheader("📋 PM Summary Report")
    st.code(results["pm_summary"], language=None)

elif transcript_text:
    st.info("👆 Click **Analyze Interview** to process the transcript.")
else:
    st.info("👆 Paste a transcript or upload a file to get started.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
      "Built by [Pooja Hegde](https://github.com/Poojaahegde) · "
      "[Portfolio](https://github.com/Poojaahegde) · "
      "Part of the AI PM Portfolio"
)
