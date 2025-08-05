import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

st.set_page_config(page_title="📊 Real-Time Productivity Dashboard", layout="wide")

# Set refresh intervals (10 min & 16 min = 600s & 960s)
REFRESH_INTERVAL = 600  # 10 minutes

# --- Load Data ---
@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    df = pd.read_csv(r"C:\Users\bhumi\OneDrive\Desktop\professional\internship\mini\data\final_fatigue_dataset.csv")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])
    return df

df = load_data()

# --- UI Title ---
st.title("🧠 Real-Time Productivity & Efficiency Dashboard")
st.markdown("Track webcam-based real-time metrics for productivity monitoring. *Auto-refreshes every 10 minutes.*")

# --- Latest Metrics ---
latest = df.sort_values('end_datetime', ascending=True).iloc[-1]
st.subheader("🧾 Latest Session Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Status", latest['status'])
    st.metric("Blinks", latest['blink_count'])
with col2:
    st.metric("Efficiency (%)", latest['efficiency'])
    st.metric("Yawn Count", latest['yawn_count'])
with col3:
    st.metric("Model Prediction", latest['model_prediction'])
    st.metric("Confidence", f"{latest['confidence']:.2f}")

# --- Raw Data Table ---
st.subheader("📄 Raw Dataset – All Sessions")
st.dataframe(df.sort_values('end_datetime', ascending=False), use_container_width=True)

# --- Session Timeline ---
st.subheader("📅 Session Timeline")
timeline = px.timeline(df, x_start='start_datetime', x_end='end_datetime', y='session_id',
                       color='status')
timeline.update_yaxes(autorange="reversed")
st.plotly_chart(timeline, use_container_width=True)

# --- Efficiency Trend ---
st.subheader("📈 Efficiency Over Time")
efficiency_line = px.line(df, x='end_datetime', y='efficiency', color='status', markers=True)
st.plotly_chart(efficiency_line, use_container_width=True)

# --- Active vs Inactive Time ---
st.subheader("🕒 Active vs Inactive Time")
melted_time = df[['session_id', 'active_time_min', 'inactive_time_min']].melt(
    id_vars='session_id', var_name='Type', value_name='Minutes')
time_fig = px.bar(melted_time, x='session_id', y='Minutes', color='Type', barmode='group')
st.plotly_chart(time_fig, use_container_width=True)

# --- Blink and Yawn Analysis ---
st.subheader("👁️ Blink & Yawn Analysis")
events_df = df[['session_id', 'blink_count', 'yawn_count']].melt(id_vars='session_id', 
            var_name='Event', value_name='Count')
blink_yawn_fig = px.bar(events_df, x='session_id', y='Count', color='Event', barmode='group')
st.plotly_chart(blink_yawn_fig, use_container_width=True)

# --- Model Confidence ---
st.subheader("🔍 Prediction Confidence")
confidence_fig = px.scatter(df, x='end_datetime', y='confidence', color='model_prediction',
                            size='efficiency', hover_data=['status'])
st.plotly_chart(confidence_fig, use_container_width=True)

# --- Session Efficiency vs Prediction ---
st.subheader("📊 Efficiency vs Model Prediction")
eff_pred_fig = px.scatter(df, x='efficiency', y='status', color='model_prediction',
                          hover_data=['session_id', 'blink_count', 'yawn_count'])
st.plotly_chart(eff_pred_fig, use_container_width=True)

# --- Status Distribution ---
st.subheader("📊 Status Distribution")
status_counts = df['status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']
status_fig = px.bar(status_counts, x='Status', y='Count', color='Status', text='Count')
status_fig.update_traces(textposition='outside')
st.plotly_chart(status_fig, use_container_width=True)

# --- Auto Refresh Timer ---
st.markdown("⏱️ Auto-refresh in **10 minutes** or **16 minutes**, whichever comes first.")
st.stop()  # Ends here but refreshes automatically every 10 minutes
