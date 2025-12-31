import streamlit as st
import pandas as pd
import io
import os
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

# Small header image to make the interface more friendly
HEADER_IMAGE = "https://images.unsplash.com/photo-1522071820081-009f0129c71c"

st.image(HEADER_IMAGE, width=900)
st.title("🧠 AI Resume Analyzer")
st.write("Match your skills with suitable job roles.")

# Sidebar: vertical navigation (radio menu)
st.sidebar.title("Navigation")
nav_choice = st.sidebar.radio("", ["🧠 Analyzer", "📤 Upload Resume", "ℹ️ About"], index=0)
# debug toggle
show_debug = st.sidebar.checkbox("Show debug info", value=False)
# map emoji labels back to simple page keys
page = {
    "🧠 Analyzer": "Analyzer",
    "📤 Upload Resume": "Upload Resume",
    "ℹ️ About": "About",
}[nav_choice]


# ---------------- PDF SUPPORT ----------------
try:
    import PyPDF2
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False


def show_header_small():
    st.markdown("---")


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

# Optional: allow filtering by job category if dataset has it
if "category" in jobs.columns:
    categories = ["All"] + sorted(jobs["category"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Job Category", options=categories, index=0)
else:
    selected_category = "All"


# ---------------- PAGES ----------------
def analyzer_page():
    show_header_small()

    # ---------------- USER INPUT ----------------
    name = st.text_input("👤 Enter your name", key="name_input")
    skills = st.text_area(
        "🛠️ Enter your skills (comma separated)",
        placeholder="python, machine learning, sql",
        key="skills_area",
    )

    # Filter jobs by selected category (if any)
    local_jobs = jobs if selected_category == "All" else jobs[jobs["category"] == selected_category].reset_index(drop=True)

    # Now that we know how many jobs exist, create the slider with min=3 and max=32
    total_jobs = len(local_jobs)
    if total_jobs > 0:
        slider_min = 3 if total_jobs >= 3 else 1
        slider_max = 32 if total_jobs >= 32 else total_jobs
        default_value = slider_max
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

    if st.button("🔍 Analyze Resume", key="analyze_btn"):
        if not name or not skills:
            st.warning("⚠️ Please enter both name and skills.")
            return

        # show a small illustration
        st.image("https://images.unsplash.com/photo-1559526324-593bc073d938", width=600)

        user_skills = clean_text(skills)
        if show_debug:
            st.info(f"Cleaned input: {user_skills}")

        # compute similarity scores against the filtered job set (robust function handles empty job lists)
        scores = calculate_similarity(user_skills, local_jobs["required_skills"].tolist())

        # Attach scores to the local copy so indices align correctly
        jobs_sorted = local_jobs.copy()
        try:
            jobs_sorted["Match Score (%)"] = (scores * 100).round(2)
        except Exception:
            jobs_sorted["Match Score (%)"] = pd.Series(scores).astype(float).fillna(0).round(2)

        if show_debug:
            st.write("Sample similarity scores (first 10):", list(scores[:10]))

        # Sort by highest score
        jobs_sorted = jobs_sorted.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        # Exclude zero-percent matches from display
        jobs_nonzero = jobs_sorted[jobs_sorted["Match Score (%)"] > 0].reset_index(drop=True)

        if jobs_nonzero.empty:
            st.error("❌ No suitable job match found (all matches are 0%). Try adding more relevant skills.")
            csv = jobs_sorted.to_csv(index=False)
            st.download_button("📥 Download full results (CSV)", data=csv, file_name="match_results.csv", mime="text/csv")
            return

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
            # Show pie chart (percentages) if matplotlib is available, else bar chart
            try:
                sizes = top_matches["Match Score (%)"].astype(float).values
            except Exception:
                sizes = pd.to_numeric(top_matches["Match Score (%)"], errors="coerce").fillna(0).values

            total = float(sizes.sum()) if len(sizes) else 0.0
            if total > 0 and _HAS_MATPLOTLIB:
                fig, ax = plt.subplots(figsize=(6, 6))
                labels = [f"{t} ({s:.1f}%)" for t, s in zip(top_matches["job_title"].values, sizes)]
                ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.bar_chart(top_matches.set_index("job_title")["Match Score (%)"])

            # compact list with progress bars and colored indicators
            for _, row in top_matches.iterrows():
                pct = float(row["Match Score (%)"])
                if pct >= 75:
                    icon = "🟢"
                elif pct >= 40:
                    icon = "🟡"
                else:
                    icon = "🔴"

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"{icon} **{row['job_title']}** — {pct:.1f}%")
                    if row.get("company"):
                        st.write(row.get("company"))
                with col2:
                    st.progress(min(max(int(pct), 0), 100))

                job_set = set(row["required_skills"].split())
                user_set = set(user_skills.split())
                shared = sorted(job_set & user_set)
                missing = sorted(job_set - user_set)
                if shared:
                    st.write("Shared:", ", ".join(shared))
                if missing:
                    st.write("Missing:", ", ".join(missing))
                st.write("---")

            best_job = jobs_nonzero.iloc[0]["job_title"]
            best_score = jobs_nonzero.iloc[0]["Match Score (%)"]
            st.success(f"✅ Best Match: **{best_job} ({best_score}%)**")

            csv = jobs_sorted.to_csv(index=False)
            st.download_button("📥 Download full results (CSV)", data=csv, file_name="match_results.csv", mime="text/csv")

            st.info("💡 Tip: Add more relevant skills to improve accuracy.")


def upload_resume_page():
    show_header_small()
    st.header("📄 Upload Resume")
    st.write("Upload a plain text file to extract skills or paste your resume text.")
    uploaded = st.file_uploader("Upload resume (.txt or .pdf)", type=["txt", "pdf"], key="uploader")
    if uploaded is not None:
        fname = uploaded.name
        if fname.lower().endswith(".txt"):
            raw = uploaded.read().decode(errors="ignore")
            st.subheader("Extracted Text")
            st.code(raw[:10000])
            cleaned = clean_text(raw)
            st.subheader("Cleaned / tokenized skills preview")
            st.write(cleaned)
        elif fname.lower().endswith(".pdf"):
            if not _HAS_PYPDF2:
                st.error("PDF parsing not available. Install PyPDF2 to enable this feature.")
            else:
                try:
                    reader = PyPDF2.PdfReader(uploaded)
                    text = []
                    for p in reader.pages:
                        text.append(p.extract_text() or "")
                    raw = "\n".join(text)
                    st.subheader("Extracted Text (first 10k chars)")
                    st.code(raw[:10000])
                    cleaned = clean_text(raw)
                    st.subheader("Cleaned / tokenized skills preview")
                    st.write(cleaned)
                except Exception as e:
                    st.error(f"Could not parse PDF: {e}")


def about_page():
    show_header_small()
    st.header("About this app")
    st.write("This small app matches your skills to job descriptions using a simple similarity model.")
    st.image("https://images.unsplash.com/photo-1542744173-8e7e53415bb0", caption="Find roles that fit your skills", width=700)

    st.markdown("**Version:** 0.1 — lightweight prototype")

    st.subheader("What this app does")
    st.write(
        "- Matches your listed skills to job descriptions from the included dataset (data/jobs.csv).\n"
        "- Ranks jobs by a simple similarity score and visualizes top matches.\n"
        "- Lets you upload a resume (text or PDF) to extract skills and compare with job requirements."
    )

    st.subheader("How it works")
    st.write(
        "1. Text cleaning: resume text and job-required skills are normalized and tokenized.\n"
        "2. Similarity: a lightweight model computes similarity between the user's skill tokens and each job's required skills.\n"
        "3. Presentation: matches are sorted and visualized as percentages with charts and progress indicators."
    )

    st.subheader("Data & Privacy")
    st.write(
        "- Job data is loaded from the local file `data/jobs.csv`.\n"
        "- Uploaded resume files are processed locally and not sent to any external server by this app.\n"
        "- If you deploy this app publicly, review hosting privacy settings and any optional telemetry."
    )

    st.subheader("Tips to get better matches")
    st.write(
        "- Provide a focused list of skills (comma separated) rather than a long paragraph.\n"
        "- Add relevant keywords used in job descriptions (framework names, tools, certifications).\n"
        "- Try different `Top matches` or `Minimum Percentage` thresholds to tune results."
    )

    st.subheader("Contact & Source")
    st.write(
        "- Repo / Source: Check the project files in this workspace.\n"
        "- Contact: Add an email or link here if you want users to reach out."
    )


if page == "Analyzer":
    analyzer_page()
elif page == "Upload Resume":
    upload_resume_page()
else:
    about_page()
