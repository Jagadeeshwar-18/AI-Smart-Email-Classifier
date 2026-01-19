import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="AI Smart Email Classifier", layout="wide")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "classified_emails" not in st.session_state:
    st.session_state.classified_emails = []

if "bulk_done" not in st.session_state:
    st.session_state.bulk_done = False

# ---------------- SIDEBAR ----------------
st.sidebar.title("🤖 AI Smart Email Classifier")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"

if st.sidebar.button("✉️ Single Email"):
    st.session_state.page = "single"

if st.sidebar.button("📂 Bulk Email"):
    st.session_state.page = "bulk"

if st.session_state.bulk_done:
    if st.sidebar.button("📊 Dashboard"):
        st.session_state.page = "dashboard"
else:
    st.sidebar.caption("🔒 Dashboard locked")

# =================================================
# 🏠 HOME
# =================================================
if st.session_state.page == "home":
    st.title("AI Smart Email Classifier")
    st.subheader("Choose a classification mode")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✉️ Single Email Classification")
        if st.button("Go to Single Email"):
            st.session_state.page = "single"

    with col2:
        st.markdown("### 📂 Bulk / Gmail Classification")
        if st.button("Go to Bulk Email"):
            st.session_state.page = "bulk"

# =================================================
# ✉️ SINGLE EMAIL
# =================================================
elif st.session_state.page == "single":
    st.title("Single Email Classification")

    email_text = st.text_area(
        "Enter Email Content",
        height=250,
        placeholder="System is down, need fix ASAP..."
    )

    if st.button("Classify Email"):
        if not email_text.strip():
            st.warning("Please enter email content")
        else:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"email": email_text}
            )
            if res.status_code == 200:
                out = res.json()
                st.success("Prediction Result")
                st.metric("Category", out["category"])
                st.metric("Urgency", out["urgency"])
                st.metric("Confidence", out["confidence"])

# =================================================
# 📂 BULK / GMAIL
# =================================================
elif st.session_state.page == "bulk":
    st.title("Bulk Email Classification")

    # ---------- CSV UPLOAD ----------
    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("CSV must contain column: text", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            st.session_state.classified_emails = []

            for text in df["text"].dropna():
                res = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"email": text}
                )
                if res.status_code == 200:
                    out = res.json()
                    st.session_state.classified_emails.append({
                        "email": text[:80],
                        "category": out["category"],
                        "urgency": out["urgency"],
                        "confidence": out["confidence"],
                        "date": datetime.date.today().isoformat()
                    })

            st.session_state.bulk_done = True
            st.success("CSV classified successfully")

    st.markdown("---")

    # ---------- GMAIL ----------
    st.subheader("📬 Gmail Synchronization")

    if st.button("Sync ALL Gmail Emails"):
        from src.gmail.fetch_emails import fetch_emails
        from src.inference.predict import predict_email

        raw_emails = fetch_emails(max_emails=1000)
        st.session_state.classified_emails = []

        progress = st.progress(0)
        total = len(raw_emails)

        for i, mail in enumerate(raw_emails):
            text = f"{mail['subject']} {mail['body']}".strip()
            if not text:
                continue

            pred = predict_email(text)

            st.session_state.classified_emails.append({
                "email": mail["subject"],
                "category": pred["category"],
                "urgency": pred["urgency"],
                "confidence": pred["confidence"],
                "date": datetime.date.today().isoformat()
            })

            progress.progress((i + 1) / total)

        st.session_state.bulk_done = True
        st.success(f"Synced & classified {len(st.session_state.classified_emails)} emails")

    if st.session_state.bulk_done:
        if st.button("➡️ Go to Dashboard"):
            st.session_state.page = "dashboard"

# =================================================
# 📊 DASHBOARD
# =================================================
elif st.session_state.page == "dashboard":
    import numpy as np
    import seaborn as sns

    st.title("📊 Intelligent Email Analytics Dashboard")

    df = pd.DataFrame(st.session_state.classified_emails)

    if df.empty:
        st.warning("No data available")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])

    # ================= FILTERS =================
    st.markdown("### ⏳ Date Filter")
    col1, col2 = st.columns(2)
    start = col1.date_input("From", df["date"].min())
    end = col2.date_input("To", df["date"].max())

    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]

    # ================= KPIs =================
    st.markdown("### ⚡ Key Metrics")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("📨 Total Emails", len(df))
    k2.metric("🚨 High Priority", (df["urgency"] == "high").sum())
    k3.metric("⚠️ Medium Priority", (df["urgency"] == "medium").sum())
    k4.metric("✅ Low Priority", (df["urgency"] == "low").sum())

    st.divider()

    # ================= PIE / DONUT =================
    st.markdown("### 🧠 Category Distribution")

    fig1, ax1 = plt.subplots(figsize=(5,5))
    colors = ["#ff595e", "#1982c4", "#6a4c93", "#8ac926"]

    ax1.pie(
        df["category"].value_counts(),
        labels=df["category"].value_counts().index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45)
    )
    ax1.set_title("Email Categories (Donut View)")
    st.pyplot(fig1)

    # ================= BAR CHART =================
    st.markdown("### 🚦 Urgency Breakdown")

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(
        x=df["urgency"].value_counts().index,
        y=df["urgency"].value_counts().values,
        palette=["#ff006e", "#ffbe0b", "#3a86ff"],
        ax=ax2
    )
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Urgency Level")
    st.pyplot(fig2)

    # ================= LINE CHART =================
    st.markdown("### 📈 Email Volume Over Time")

    timeline = df.groupby(df["date"].dt.date).size()

    fig3, ax3 = plt.subplots(figsize=(7,4))
    ax3.plot(
        timeline.index,
        timeline.values,
        marker="o",
        color="#8338ec",
        linewidth=3
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Emails")
    ax3.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig3)

    # ================= STACKED BAR =================
    st.markdown("### 🧩 Category vs Urgency (Stacked View)")

    pivot = pd.pivot_table(
        df,
        index="category",
        columns="urgency",
        aggfunc="size",
        fill_value=0
    )

    fig4, ax4 = plt.subplots(figsize=(7,4))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax4,
        colormap="tab20c"
    )
    ax4.set_ylabel("Count")
    ax4.legend(title="Urgency")
    st.pyplot(fig4)

    # ================= HEATMAP =================
    st.markdown("### 🔥 Density Heatmap")

    heat = pd.crosstab(df["category"], df["urgency"])

    fig5, ax5 = plt.subplots(figsize=(6,4))
    sns.heatmap(
        heat,
        annot=True,
        cmap="rocket",
        fmt="d",
        linewidths=0.5,
        ax=ax5
    )
    st.pyplot(fig5)

    # ================= TABLE =================
    st.markdown("### 📨 Recent Classified Emails")
    st.dataframe(df.tail(20), use_container_width=True)

    # ================= DOWNLOAD =================
    st.download_button(
        "⬇ Download Dashboard Data (CSV)",
        data=df.to_csv(index=False),
        file_name="dashboard_emails.csv"
    )
