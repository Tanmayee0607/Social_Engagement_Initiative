import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Social Engagement Initiative",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"


@st.cache_data
def load_data():
    posts = pd.read_csv(DATA / "content_posts.csv", parse_dates=["date"])
    comments = pd.read_csv(DATA / "comments.csv")
    ideas = pd.read_csv(DATA / "content_series.csv")
    return posts, comments, ideas


posts, comments, ideas = load_data()

st.title("📊 Data-Driven Social Engagement Initiative")
st.caption(
    "A beginner-friendly demonstration dashboard. Included data is synthetic "
    "and is not collected from real social media accounts."
)

with st.sidebar:
    st.header("Filters")
    topic_options = ["All"] + sorted(posts["topic"].dropna().unique().tolist())
    chosen_topic = st.selectbox("Topic", topic_options)
    format_options = ["All"] + sorted(posts["format"].dropna().unique().tolist())
    chosen_format = st.selectbox("Content format", format_options)
    st.markdown("---")
    st.caption("Data source: included synthetic CSV files")

filtered = posts.copy()
if chosen_topic != "All":
    filtered = filtered[filtered["topic"] == chosen_topic]
if chosen_format != "All":
    filtered = filtered[filtered["format"] == chosen_format]

# Explainable engagement and virality metrics
filtered["engagement_rate"] = (
    (
        filtered["likes"]
        + filtered["comments"]
        + filtered["shares"]
        + filtered["saves"]
    )
    / filtered["reach"].replace(0, np.nan)
    * 100
)
filtered["viral_score"] = (
    3 * filtered["shares"]
    + 2 * filtered["saves"]
    + filtered["likes"]
    + filtered["comments"]
) / filtered["reach"].replace(0, np.nan) * 100

total_posts = len(filtered)
avg_engagement = filtered["engagement_rate"].mean() if total_posts else 0
avg_shares = filtered["shares"].mean() if total_posts else 0
avg_saves = filtered["saves"].mean() if total_posts else 0
relatable_share = (
    comments["label"].eq("Relatable").mean() * 100 if len(comments) else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts in view", f"{total_posts}")
c2.metric("Average engagement rate", f"{avg_engagement:.2f}%")
c3.metric("Average shares", f"{avg_shares:.1f}")
c4.metric("Sample comments labelled relatable", f"{relatable_share:.1f}%")

st.info(
    "Interpretation: engagement rate is total likes, comments, shares and saves "
    "divided by reach. The viral score is a simple project-defined ranking score, "
    "not a validated probability of going viral."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Virality ranking",
    "Comment NLP",
    "A/B comparison",
    "Recommendations & content",
    "Trend Forecasting"
])

# 1. OVERVIEW
with tab1:
    st.subheader("Performance by topic")

    if filtered.empty:
        st.warning("No posts match the selected filters.")
    else:
        by_topic = (
            filtered.groupby("topic", as_index=False)
            .agg(
                average_engagement=("engagement_rate", "mean"),
                average_shares=("shares", "mean"),
                average_saves=("saves", "mean"),
                posts=("post_id", "count")
            )
            .sort_values("average_engagement", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(
            by_topic["topic"],
            by_topic["average_engagement"]
        )
        ax.set_ylabel("Average engagement rate (%)")
        ax.set_xlabel("Topic")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.dataframe(
            by_topic.round(2),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Performance over time")
    if filtered.empty:
        st.info("Select a broader filter to display the weekly trend.")
    else:
        weekly_overview = filtered[["date", "engagement_rate"]].dropna().copy()
        weekly_overview["week"] = (
            weekly_overview["date"].dt.to_period("W").astype(str)
        )
        weekly_overview = (
            weekly_overview.groupby("week", as_index=False)
            .agg(average_engagement=("engagement_rate", "mean"))
        )

        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.plot(
            range(len(weekly_overview)),
            weekly_overview["average_engagement"],
            marker="o"
        )
        ax2.set_ylabel("Average engagement rate (%)")
        ax2.set_xlabel("Week")
        step = max(1, len(weekly_overview) // 6)
        tick_positions = list(range(0, len(weekly_overview), step))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(
            weekly_overview.iloc[::step]["week"],
            rotation=35,
            ha="right"
        )
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=False)
        plt.close(fig2)

# 2. VIRALITY RANKING
with tab2:
    st.subheader("Top posts by project-defined viral score")
    if filtered.empty:
        st.warning("No posts match the selected filters.")
    else:
        ranking = filtered.sort_values("viral_score", ascending=False)[
            [
                "post_id", "date", "topic", "format", "reach", "likes",
                "shares", "saves", "comments", "retention_rate", "viral_score"
            ]
        ].head(15)
        st.dataframe(
            ranking.round(2),
            use_container_width=True,
            hide_index=True
        )
    st.caption(
        "Shares receive weight 3, saves weight 2, and likes/comments weight 1. "
        "These weights are illustrative and can be changed after discussion with your faculty."
    )

# 3. COMMENT NLP
with tab3:
    st.subheader("Audience comment classifier")
    st.write(
        "A TF-IDF + Logistic Regression baseline is trained on the included labelled "
        "example comments. Because this dataset is synthetic and small, the model is "
        "only a demonstration and should not be treated as a reliable real-world classifier."
    )

    X = comments["comment_text"].astype(str)
    y = comments["label"].astype(str)

    if len(comments) >= 10 and y.nunique() > 1:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_vec, y_train)
            preds = model.predict(X_test_vec)

            st.metric(
                "Hold-out accuracy (synthetic examples)",
                f"{accuracy_score(y_test, preds) * 100:.1f}%"
            )
            st.code(
                classification_report(y_test, preds, zero_division=0),
                language="text"
            )

            new_comment = st.text_area(
                "Try a comment",
                "I thought I was the only one feeling this way."
            )
            if st.button("Classify comment"):
                prediction = model.predict(
                    vectorizer.transform([new_comment])
                )[0]
                st.success(f"Predicted label: {prediction}")
        except Exception as exc:
            st.error(f"Could not train the sample classifier: {exc}")
    else:
        st.warning("Need at least 10 labelled comments and two label classes.")

# 4. A/B COMPARISON
with tab4:
    st.subheader("Descriptive format comparison")
    st.write(
        "This compares historical sample posts by format. It is not a controlled "
        "experiment unless posts were randomly assigned and other factors were controlled."
    )

    comparison = (
        filtered.groupby("format", as_index=False)
        .agg(
            posts=("post_id", "count"),
            average_engagement=("engagement_rate", "mean"),
            average_retention=("retention_rate", "mean"),
            average_shares=("shares", "mean")
        )
        .sort_values("average_engagement", ascending=False)
    )
    st.dataframe(
        comparison.round(3),
        use_container_width=True,
        hide_index=True
    )

    if not comparison.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.bar(comparison["format"], comparison["average_engagement"])
        ax3.set_ylabel("Average engagement rate (%)")
        ax3.set_xlabel("Format")
        ax3.tick_params(axis="x", rotation=15)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=False)
        plt.close(fig3)

    st.caption(
        "For a genuine A/B test, publish comparable versions under a defined protocol "
        "and record exposure, timing and audience conditions."
    )

# 5. RECOMMENDATIONS AND CONTENT
with tab5:
    st.subheader("Recommendations based on the selected data")

    if filtered.empty:
        st.warning("Select a broader filter to generate recommendations.")
    else:
        topic_summary = (
            filtered.groupby("topic")["engagement_rate"]
            .mean()
            .sort_values(ascending=False)
        )
        format_summary = (
            filtered.groupby("format")["engagement_rate"]
            .mean()
            .sort_values(ascending=False)
        )

        best_topic = topic_summary.index[0]
        best_format = format_summary.index[0]

        st.markdown(
            f"- **Topic to investigate further:** {best_topic} "
            "(highest average engagement rate in the current filtered sample)."
        )
        st.markdown(
            f"- **Format to test next:** {best_format} "
            "(highest average engagement rate in the current filtered sample)."
        )
        st.markdown(
            "- **Next experiment:** publish two comparable posts with different hooks, "
            "keeping timing and topic as similar as practical."
        )
        st.markdown(
            "- **Content quality:** review the actual comments before finalising the next "
            "post; numeric metrics alone do not explain why people respond."
        )

    st.subheader("Draft content series")
    st.dataframe(ideas, use_container_width=True, hide_index=True)

# 6. TREND FORECASTING
with tab6:
    st.subheader("Trend Forecasting Module")
    st.write(
        "This module analyses weekly engagement and estimates the next four weeks "
        "using a simple linear trend. It also summarises frequently occurring hashtags "
        "in the selected sample."
    )

    if filtered.empty:
        st.warning("No posts match the selected filters.")
    else:
        trend_data = filtered[["date", "engagement_rate"]].dropna().copy()
        trend_data["week"] = (
            trend_data["date"].dt.to_period("W").dt.start_time
        )

        weekly = (
            trend_data.groupby("week", as_index=False)["engagement_rate"]
            .mean()
            .sort_values("week")
        )

        st.subheader("Weekly engagement trend")
        st.dataframe(
            weekly.round(2),
            use_container_width=True,
            hide_index=True
        )

        if len(weekly) >= 2:
            x = np.arange(len(weekly))
            y_values = weekly["engagement_rate"].to_numpy()
            slope, intercept = np.polyfit(x, y_values, 1)

            future_x = np.arange(len(weekly), len(weekly) + 4)
            forecast_values = slope * future_x + intercept
            future_dates = pd.date_range(
                start=weekly["week"].max() + pd.Timedelta(weeks=1),
                periods=4,
                freq="W-MON"
            )

            forecast = pd.DataFrame({
                "week": future_dates,
                "forecast_engagement_rate": np.maximum(forecast_values, 0)
            })

            fig4, ax4 = plt.subplots(figsize=(9, 4))
            ax4.plot(
                weekly["week"],
                weekly["engagement_rate"],
                marker="o",
                label="Observed weekly average"
            )
            ax4.plot(
                forecast["week"],
                forecast["forecast_engagement_rate"],
                marker="o",
                linestyle="--",
                label="Simple trend estimate"
            )
            ax4.set_xlabel("Week")
            ax4.set_ylabel("Engagement rate (%)")
            ax4.set_title("Engagement Trend and Four-Week Estimate")
            ax4.legend()
            fig4.autofmt_xdate()
            fig4.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

            st.subheader("Four-week forecast")
            st.dataframe(
                forecast.round(2),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "At least two different weeks of data are needed to calculate a trend estimate."
            )

        st.subheader("Frequently occurring hashtags")
        hashtag_counts = (
            filtered["hashtags"]
            .fillna("")
            .str.lower()
            .str.findall(r"#\w+")
            .explode()
            .dropna()
            .value_counts()
            .head(10)
        )

        if not hashtag_counts.empty:
            hashtag_table = (
                hashtag_counts.rename("post_count")
                .rename_axis("hashtag")
                .reset_index()
            )
            st.dataframe(
                hashtag_table,
                use_container_width=True,
                hide_index=True
            )
            st.bar_chart(hashtag_counts)
        else:
            st.info("No hashtags are available in the selected posts.")

        st.info(
            "Limitation: this is a simple linear trend estimate, not a validated forecasting "
            "model. Results use synthetic sample data and should not be treated as actual "
            "future performance or proof that a hashtag is trending on a social platform."
        )

st.markdown("---")
st.caption(
    "Project prototype • Synthetic demonstration data • Replace with approved real/public "
    "data before making real-world claims."
)
