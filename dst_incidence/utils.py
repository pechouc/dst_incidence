import os

import time

import re

import numpy as np
import pandas as pd


def clean_date_string(string):

    string = string.split('(')[1].strip().strip(')').replace('.', '')

    day, fr_month, year = string.split(' ')

    month = {
        'jan': 'jan',
        'fév': 'feb',
        'mar': 'mar',
        'avr': 'apr',
        'mai': 'may',
        'juin': 'jun',
        'juil': 'jul',
        'août': 'aug',
        'sept': 'sep',
        'oct': 'oct',
        'nov': 'nov',
        'déc': 'dec',
    }.get(fr_month, fr_month)

    string = ' '.join([day, month, year])

    return string


def draw_random_sku_country(path_to_price_data):

    path_to_price_history = os.path.join(path_to_price_data, 'price_history.csv')
    path_to_price_history_summary = os.path.join(path_to_price_data, 'price_history_summary.csv')

    sku_country = pd.read_csv(path_to_price_history, usecols=["sku", "country"])

    random_draw = sku_country.sample(1, random_state=int(time.time())).reset_index(drop=True)
    random_sku, random_country = random_draw.loc[0, 'sku'], random_draw.loc[0, 'country']

    price_history = pd.read_csv(path_to_price_history)
    price_history = price_history[
        np.logical_and(
            price_history['sku'] == random_sku,
            price_history['country'] == random_country
        )
    ].copy()

    price_history['date'] = pd.to_datetime(price_history['date'])

    new_price_history = price_history[price_history['seller'] == 'new'].copy()
    amazon_price_history = price_history[price_history['seller'] == 'amazon'].copy()

    del price_history

    new_price_history['transition'] = np.logical_and(
        new_price_history['trusted'] != new_price_history['trusted'].shift(1),
        ~new_price_history['trusted'].shift(1).isnull()
    )

    new_price_history['transition#'] = new_price_history['transition'].cumsum()

    new_trusted_dfs = []
    new_not_trusted_dfs = []

    for i in range(new_price_history['transition#'].max() + 1):
        extract = new_price_history[new_price_history['transition#'] == i].copy()

        if extract['trusted'].unique()[0]:
            new_trusted_dfs.append(extract)

        else:
            new_not_trusted_dfs.append(extract)

    amazon_price_history['transition'] = np.logical_and(
        amazon_price_history['trusted'] != amazon_price_history['trusted'].shift(1),
        ~amazon_price_history['trusted'].shift(1).isnull()
    )

    amazon_price_history['transition#'] = amazon_price_history['transition'].cumsum()

    amazon_trusted_dfs = []
    amazon_not_trusted_dfs = []

    price_history_summary = pd.read_csv(path_to_price_history_summary)
    price_history_summary = price_history_summary[
        np.logical_and(
            price_history_summary['sku'] == random_sku,
            price_history_summary['country'] == random_country
        )
    ].copy()

    amazon_price_history_summary = price_history_summary[
        price_history_summary["seller"] == "amazon"
    ].reset_index(drop=True)
    new_price_history_summary = price_history_summary[
        price_history_summary["seller"] == "new"
    ].reset_index(drop=True)

    max_prices = {
        'amazon': amazon_price_history_summary.loc[0, 'max_price'],
        'new': new_price_history_summary.loc[0, 'max_price']
    }
    min_prices = {
        'amazon': amazon_price_history_summary.loc[0, 'min_price'],
        'new': new_price_history_summary.loc[0, 'min_price']
    }

    for i in range(amazon_price_history['transition#'].max() + 1):
        extract = amazon_price_history[amazon_price_history['transition#'] == i].copy()

        if extract['trusted'].unique()[0]:
            amazon_trusted_dfs.append(extract)

        else:
            amazon_not_trusted_dfs.append(extract)

    return (
        random_sku,
        random_country,
        new_trusted_dfs.copy(),
        new_not_trusted_dfs.copy(),
        amazon_trusted_dfs.copy(),
        amazon_not_trusted_dfs.copy(),
        max_prices.copy(),
        min_prices.copy()
    )


def get_dates_from_text_extract(text_extract):

    # Getting the first month
    months = np.array(
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )

    months_idx = []

    for month in months:
        months_idx.append(
            text_extract.find(month) if text_extract.find(month) > -1 else np.nan
        )

    months_idx = np.array(months_idx)

    first_month = months[np.where(months_idx == np.nanmin(months_idx))][0]

    # Getting the last month
    months_idx_right = []

    for month in months:
        months_idx_right.append(
            text_extract.rfind(month) if text_extract.rfind(month) > -1 else np.nan
        )

    months_idx_right = np.array(months_idx_right)

    last_month = months[np.where(months_idx_right == np.nanmax(months_idx_right))][0]

    # Getting first and last years
    first_year = 2000 + int(re.findall(r'[^€\d+,](\d\d)', text_extract)[0])
    last_year = 2000 + int(re.findall(r'[^€\d+,](\d\d)', text_extract)[-1])

    # Correcting first and last months if needed
    if text_extract.find(str(first_year - 2000)) < text_extract.find(first_month):
        first_month = 'Jan'
    else:
        first_year = first_year - 1
    if text_extract.rfind(str(last_year - 2000)) > text_extract.rfind(last_month):
        last_month = 'Jan'

    return first_month, first_year, last_month, last_year
