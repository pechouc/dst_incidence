
from datetime import date

import os

from dst_incidence.utils import draw_random_sku_country

# Streamlit import
import streamlit as st

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


path_to_dir = os.path.dirname(os.path.abspath(__file__))
path_to_price_data = os.path.join(path_to_dir, 'data', 'prices')

path_to_data_raw = "/Users/Paul-Emmanuel/Desktop/PhD/3_DST_incidence/data_raw"

st.set_page_config(layout="wide")

st.title('Random Checks')

(
    random_sku,
    random_country,
    new_trusted_dfs,
    new_not_trusted_dfs,
    amazon_trusted_dfs,
    amazon_not_trusted_dfs,
    max_prices,
    min_prices
) = draw_random_sku_country(path_to_price_data=path_to_price_data)

paths_to_charts = {}

for seller in ['new', 'amazon']:

    paths_to_charts[seller] = os.path.join(path_to_data_raw, f"charts_{seller}_{random_country}", f"{random_sku}.png")


st.markdown("# Useful information")

st.markdown(random_sku)
st.markdown(random_country)

st.markdown("---")

st.markdown("# Choice of seller")

seller = st.selectbox(label="Type of seller", options=['new', 'amazon'])

st.markdown("---")

st.markdown("# Graphs")

col1, col2 = st.columns(spec=2)

with col1:
    st.image(paths_to_charts[seller])

# --- Graphs based on our data

if seller == "amazon":
    trusted_dfs = amazon_trusted_dfs.copy()
    not_trusted_dfs = amazon_not_trusted_dfs.copy()
else:
    trusted_dfs = new_trusted_dfs.copy()
    not_trusted_dfs = new_not_trusted_dfs.copy()

fig, ax = plt.subplots(figsize=(15, 11))

min_date_trusted = date.fromisoformat('2026-12-04')
max_date_trusted = date.fromisoformat('2001-12-04')
min_date_not_trusted = date.fromisoformat('2026-12-04')
max_date_not_trusted = date.fromisoformat('2001-12-04')

for i, trusted_df in enumerate(trusted_dfs):
    if i == 0:
        min_date_trusted = trusted_df['date'].min()
    if i == len(trusted_dfs) - 1:
        max_date_trusted = trusted_df['date'].max()

    ax.plot(
        trusted_df['date'],
        trusted_df['price'],
        color='darkblue' if seller == "new" else "green"
    )

for i, not_trusted_df in enumerate(not_trusted_dfs):
    if i == 0:
        min_date_not_trusted = not_trusted_df['date'].min()
    if i == len(not_trusted_dfs) - 1:
        max_date_not_trusted = not_trusted_df['date'].max()

    ax.scatter(
        not_trusted_df['date'],
        not_trusted_df['price'],
        marker='s',
        s=1,
        color='darkred'
    )

ax.hlines(
    (max_prices[seller], min_prices[seller]),
    xmin=min(min_date_trusted, min_date_not_trusted),
    xmax=max(max_date_trusted, max_date_not_trusted),
    colors=('red', 'green'),
    linestyles='dashed',
)

# fig.savefig("/Users/Desktop/temp.png")

with col2:
    st.pyplot(fig, use_container_width=True)
