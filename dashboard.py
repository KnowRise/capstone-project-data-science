from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'output'


@st.cache_data
def load_data():
	job_skill_freq = pd.read_csv(OUTPUT_DIR / 'it_job_skill_freq_technical.csv')
	resume_skill_freq = pd.read_csv(OUTPUT_DIR / 'it_resume_skill_freq_technical.csv')
	role_supply_demand = pd.read_csv(OUTPUT_DIR / 'it_role_supply_demand.csv')
	skill_gap = pd.read_csv(OUTPUT_DIR / 'it_skill_gap_technical.csv')
	job_features = pd.read_csv(OUTPUT_DIR / 'it_job_features_model_ready.csv')
	resume_features = pd.read_csv(OUTPUT_DIR / 'it_resume_features_model_ready.csv')
	data_dictionary = pd.read_csv(OUTPUT_DIR / 'it_data_dictionary.csv')
	return job_skill_freq, resume_skill_freq, role_supply_demand, skill_gap, job_features, resume_features, data_dictionary


st.set_page_config(page_title='KaburAjaDulu AI Dashboard', page_icon='📊', layout='wide')

st.title('KaburAjaDulu AI - IT Career Readiness Dashboard')
st.caption('Dashboard ini menampilkan technical skill demand lowongan IT, technical skill pada resume, dan gap supply-demand per role.')

job_skill_freq, resume_skill_freq, role_supply_demand, skill_gap, job_features, resume_features, data_dictionary = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Lowongan IT Terfilter', f"{len(job_features):,}")
col2.metric('Resume IT Terfilter', f"{len(resume_features):,}")
col3.metric('Role IT Terdeteksi', f"{role_supply_demand['role_bucket'].nunique()}")
col4.metric('Skill Katalog', f"{skill_gap['skill'].nunique()}")

st.divider()

left, right = st.columns([1.1, 0.9])

with left:
	st.subheader('Top Technical Skill pada Lowongan IT')
	top_job_skill = job_skill_freq.sort_values('job_mentions', ascending=False).head(10)
	st.bar_chart(top_job_skill.set_index('skill')['job_mentions'])

with right:
	st.subheader('Top Technical Skill pada Resume IT')
	top_resume_skill = resume_skill_freq.sort_values('resume_mentions', ascending=False).head(10)
	st.bar_chart(top_resume_skill.set_index('skill')['resume_mentions'])

st.divider()

role_options = role_supply_demand.sort_values('job_postings', ascending=False)['role_bucket'].tolist()
selected_role = st.selectbox('Pilih role IT untuk melihat gap skill detail', role_options)

selected_role_supply = role_supply_demand.loc[role_supply_demand['role_bucket'] == selected_role].copy()
selected_role_gap = skill_gap.loc[skill_gap['role_bucket'] == selected_role].sort_values('gap_pct', ascending=False).head(10)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric('Lowongan', f"{int(selected_role_supply['job_postings'].iloc[0]):,}")
metric_col2.metric('Resume', f"{int(selected_role_supply['resume_count'].iloc[0]):,}")
metric_col3.metric('Rasio Resume/Lowongan', f"{selected_role_supply['supply_demand_ratio'].iloc[0]:.3f}")

gap_col1, gap_col2 = st.columns([1, 1])

with gap_col1:
	st.subheader(f'Top Skill Gap untuk {selected_role}')
	st.bar_chart(selected_role_gap.set_index('skill')['gap_pct'])

with gap_col2:
	st.subheader('Tabel Gap Skill Detail')
	st.dataframe(
		selected_role_gap[['skill', 'job_mentions', 'resume_mentions', 'gap_mentions', 'gap_pct']],
		use_container_width=True,
		hide_index=True,
	)

st.divider()

st.subheader('Role dengan Gap Demand vs Supply Tertinggi')
st.dataframe(
	role_supply_demand.sort_values('demand_supply_gap', ascending=False)[
		['role_bucket', 'job_postings', 'resume_count', 'demand_supply_gap', 'supply_demand_ratio']
	],
	use_container_width=True,
	hide_index=True,
)

st.subheader('Data Dictionary Output')
st.dataframe(data_dictionary, use_container_width=True, hide_index=True)

st.divider()
st.caption('Output CSV dibaca dari folder output/ yang dihasilkan oleh main.ipynb. Dashboard ini tidak memakai kolom target sebagai fitur training.')
