import streamlit as st
import pandas as pd
import io
from preprocessing.text_cleaning import clean_text
from model.similarity_model import calculate_similarity

# Try to import matplotlib for pie chart rendering; fallback to bar chart if unavailable
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📊", layout="centered")

st.title("🧠 AI Resume Analyzer")
st.write("Match your skills with suitable job roles.")


# ---------------- USER INPUT ----------------
name = st.text_input("👤 Enter your name", key="name_input")
skills = st.text_area(
    "🛠️ Enter your skills (comma separated)",
    placeholder="python, machine learning, sql",
    key="skills_area",
)


# ---------------- LOAD DATA ----------------
try:
    jobs = pd.read_csv("data/jobs.csv")
except Exception:
    st.error("Could not load data/jobs.csv — please ensure the file exists.")
    st.stop()

if "required_skills" not in jobs.columns:
    st.error("data/jobs.csv missing required column 'required_skills'.")
    st.stop()

jobs["required_skills"] = jobs["required_skills"].fillna("").apply(clean_text)

# Now that we know how many jobs exist, create the slider with min=3 and max=32
total_jobs = len(jobs)
if total_jobs > 0:
    slider_min = 3 if total_jobs >= 3 else 1
    slider_max = 32 if total_jobs >= 32 else total_jobs
    default_value = slider_max
    # Streamlit requires min < max for slider; handle equal-case by showing number_input
    if slider_max > slider_min:
        top_n = st.slider("Top matches to show", min_value=slider_min, max_value=slider_max, value=default_value, key="top_n_slider")
    else:
        top_n = slider_max
        st.write(f"Top matches to show: {top_n}")
else:
    top_n = 1

# Display mode: by count (Top N) or by minimum percentage threshold
display_mode = st.radio("Display results by", ("Top Priority", "Minimum Percentage"), index=0, key="display_mode")
min_pct = None
if display_mode == "Minimum Percentage":
    min_pct = st.slider("Minimum match percentage", min_value=0, max_value=100, value=20, key="min_pct")


# ---------------- ANALYZE ----------------
if st.button("🔍 Analyze Resume", key="analyze_btn"):

    if not name or not skills:
        st.warning("⚠️ Please enter both name and skills.")
    else:
        user_skills = clean_text(skills)

        # compute similarity scores (robust function handles empty job lists)
        scores = calculate_similarity(user_skills, jobs["required_skills"].tolist())
        try:
            jobs["Match Score (%)"] = (scores * 100).round(2)
        except Exception:
            # fallback if calculate_similarity returned list-like
            jobs["Match Score (%)"] = pd.Series(scores).astype(float).fillna(0).round(2)

        # Sort by highest score
        jobs_sorted = jobs.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        # Exclude zero-percent matches from display
        jobs_nonzero = jobs_sorted[jobs_sorted["Match Score (%)"] > 0].reset_index(drop=True)

        if jobs_nonzero.empty:
            st.error("❌ No suitable job match found (all matches are 0%). Try adding more relevant skills.")
            csv = jobs_sorted.to_csv(index=False)
            st.download_button("📥 Download full results (CSV)", data=csv, file_name="match_results.csv", mime="text/csv")
            st.stop()

        st.subheader("📊 Match Results")

        # Select matches according to display mode (operate on non-zero set)
        if display_mode == "Top Priority":
            top_n = min(int(top_n), len(jobs_nonzero))
            if top_n <= 0:
                top_matches = jobs_nonzero.head(0)
            else:
                cutoff = jobs_nonzero["Match Score (%)"].nlargest(top_n).min()
                top_matches = jobs_nonzero[jobs_nonzero["Match Score (%)"] >= cutoff].reset_index(drop=True)
        else:
            top_matches = jobs_nonzero[jobs_nonzero["Match Score (%)"] >= (min_pct or 0)].reset_index(drop=True)

        st.write(f"Showing {len(top_matches)} of {len(jobs_nonzero)} matching jobs (0% matches hidden)")

        if top_matches.empty:
            st.info("No matches meet the selected criteria.")
        else:
            # Visualization: pie chart preferred, bar fallback
            if _HAS_MATPLOTLIB:
                fig, ax = plt.subplots(figsize=(6, 6))
                sizes = top_matches["Match Score (%)"].values.astype(float)
                total = sizes.sum()
                if total <= 0:
                    st.bar_chart(top_matches.set_index("job_title")["Match Score (%)"])
                else:
                    labels = [f"{t} ({s:.1f}%)" for t, s in zip(top_matches["job_title"].values, sizes)]
                    ax.pie(sizes / total, labels=labels, autopct="%1.1f%%", startangle=90)
                    ax.axis("equal")
                    st.pyplot(fig)
            else:
                st.bar_chart(top_matches.set_index("job_title")["Match Score (%)"])

            # List matches and shared/missing skills
            user_set = set(user_skills.split())
            for _, row in top_matches.iterrows():
                st.markdown(f"**{row['job_title']}** → {row['Match Score (%)']}%")
                job_set = set(row["required_skills"].split())
                shared = sorted(job_set & user_set)
                missing = sorted(job_set - user_set)
                if shared:
                    st.write("Shared skills:", ", ".join(shared))
                if missing:
                    st.write("Missing skills:", ", ".join(missing))
                st.write("---")

            # Best match (first of nonzero sorted)
            best_job = jobs_nonzero.iloc[0]["job_title"]
            best_score = jobs_nonzero.iloc[0]["Match Score (%)"]
            st.success(f"✅ Best Match: **{best_job} ({best_score}%)**")

            # Offer CSV download of full ranked results (including zeros)
            csv = jobs_sorted.to_csv(index=False)
            st.download_button("📥 Download full results (CSV)", data=csv, file_name="match_results.csv", mime="text/csv")

            st.info("💡 Tip: Add more relevant skills to improve accuracy.")
