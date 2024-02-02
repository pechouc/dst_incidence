# ======================================================================================================================
# --- Housekeeping -----------------------------------------------------------------------------------------------------
# ======================================================================================================================

import os
path_to_dir = os.path.dirname(os.path.abspath(__file__))

import time
import copy

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract/5.2.0/bin/tesseract"

import re

from datetime import datetime, date

from dst_incidence.utils import clean_date_string, get_dates_from_text_extract


# ======================================================================================================================
# --- Python class to collect price histories --------------------------------------------------------------------------
# ======================================================================================================================

class PriceHistoryCollector():

    def __init__(
        self,
        data_raw_dir,
        sku_reference,
        country
    ):

        # Paul-Emmanuel - 22/01/2023
        # Relevant "data_raw_dir" on my laptop: /Users/Paul-Emmanuel/Desktop/PhD/3_DST_incidence/data_raw

        self.data_raw_dir = data_raw_dir

        self.sku_reference = sku_reference
        self.country = country

        self.sellers = {'amazon': {}, 'new': {}}

        self.seller_colors = {
            'amazon': [99, 168, 94, 255],
            'new': [0, 51, 204, 255]
        }

        self.seller_correspondence = {
            'amazon': 'Amazon',
            'new': 'Nouveauté 3ème partie',
            'used': '3eme occasion',
        }

        self.path_to_price_data = os.path.join(path_to_dir, "data", "prices")

        self.unit_correspondence = {'France': '€', 'Germany': '€'}

    def collect_and_clean_data(self):

        self.fetch_product_information()

        self.build_price_history_from_charts()
        self.get_start_end_dates_from_charts()
        self.convert_coordinates_to_timestamps()

        self.format_price_histories()
        self.format_price_history_summary()
        self.format_product_data()
        self.build_collection_metadata()

        self.save_results()

    def fetch_product_information(self):

        product_country_dir = f"products_{self.country}"
        product_country_dir = os.path.join(self.data_raw_dir, product_country_dir)

        relevant_files = [
            file for file in os.listdir(product_country_dir) if file.endswith('html')
        ]

        for html_file in relevant_files:

            path_to_html_file = os.path.join(product_country_dir, html_file)

            with open(path_to_html_file, 'r') as html_file_opened:
                html_content = html_file_opened.read()

                if html_content.count(self.sku_reference) > 2:
                    self.product_page_source = html_content
                    self.full_product_name = html_file

                else:
                    continue

        soup = BeautifulSoup(self.product_page_source, "html.parser")

        price_history_summary = pd.read_html(
            str(soup.find_all(class_='table-scroll camelegend')[0].find('table'))
        )[0]
        self.price_history_summary = price_history_summary.copy()

        min_prices = copy.deepcopy(self.sellers)
        max_prices = copy.deepcopy(self.sellers)

        for seller, _ in self.sellers.items():
            if seller == 'amazon':
                row_seller = 'Amazon'

            elif seller == 'new':
                row_seller = 'Nouveauté 3ème partie'

            else:
                row_seller = '3eme occasion'

            min_price = price_history_summary.set_index(
                'Type de prix'
            ).loc[
                row_seller, 'Le plus bas jamais atteint *'
            ]
            if '€' in min_price:
                min_price = float(min_price[:min_price.find('€')].replace(',', '.'))
            else:
                if min_price.strip() == '-':
                    min_price = np.nan
                else:
                    raise Exception("Special case in the price history summary table.")
            min_prices[seller] = min_price

            max_price = price_history_summary.set_index(
                'Type de prix'
            ).loc[
                row_seller, 'Le plus élevé de tous les temps *'
            ]
            if '€' in max_price:
                max_price = float(max_price[:max_price.find('€')].replace(',', '.'))
            else:
                if max_price.strip() == '-':
                    max_price = np.nan
                else:
                    raise Exception("Special case in the price history summary table.")
            max_prices[seller] = max_price

        self.min_prices = copy.deepcopy(min_prices)
        self.max_prices = copy.deepcopy(max_prices)

        self.product_data = pd.read_html(str(soup.find_all(class_='product_fields')[0]))[0]

    def build_price_history_from_charts(self):

        self.paths_to_charts = copy.deepcopy(self.sellers)

        for seller in self.paths_to_charts.keys():

            seller_country_dir = os.path.join(self.data_raw_dir, f'charts_{seller}_{self.country}')

            self.paths_to_charts[seller] = os.path.join(seller_country_dir, f'{self.sku_reference}.png')

        self.price_histories = copy.deepcopy(self.sellers)

        for seller, file_name in self.paths_to_charts.items():

            img = Image.open(file_name)
            img_array = np.array(img)

            # Pixels that have the RGB code for the maximum price
            y, x = np.where(np.all(img_array == [194, 68, 68, 255], axis=2))

            if len(x) == 0:
                print("No maximum price for the chart with seller ==", seller)
                print("--- Skipping the extraction of the corresponding price history.")

                empty_df = {'x_axis_coord': [], 'trusted': [], 'price': []}
                empty_df = pd.DataFrame(empty_df)

                self.price_histories[seller] = empty_df.copy()

                continue

            # Average y-coordinate of the left-most pixels with this RGB code
            avg_y_coord_max = img_array.shape[0] - y[np.where(x == min(x))].mean()

            # Minimum x-coordinate of the pixels with this RGB code outside of the horizontal dashed line
            y_coords_max_price_dashed_line = list(y[np.where(x == min(x))])
            y_coords_max_price_dashed_line_tolerance = y_coords_max_price_dashed_line.copy()
            y_coords_max_price_dashed_line_tolerance.append(max(y_coords_max_price_dashed_line) + 1)
            y_coords_max_price_dashed_line_tolerance.append(min(y_coords_max_price_dashed_line) - 1)
            outside_chart_x_coord = min(
                x[
                    np.where(
                        ~np.isin(y, y_coords_max_price_dashed_line_tolerance)
                    )
                ]
            )

            # Pixels that have the RGB code for the minimum price
            y, x = np.where(np.all(img_array == [119, 195, 107, 255], axis=2))

            if len(x) == 0:
                print("No minimum price for the chart with seller ==", seller)
                print("--- Skipping the extraction of the corresponding price history.")

                empty_df = {'x_axis_coord': [], 'trusted': [], 'price': []}
                empty_df = pd.DataFrame(empty_df)

                self.price_histories[seller] = empty_df.copy()

                continue

            # Average y-coordinate of the left-most pixels with this RGB code
            avg_y_coord_min = img_array.shape[0] - y[np.where(x == min(x))].mean()

            # Minimum x-coordinate of the pixels with this RGB code outside of the horizontal dashed line
            y_coords_min_price_dashed_line = list(y[np.where(x == min(x))])
            y_coords_min_price_dashed_line_tolerance = y_coords_min_price_dashed_line.copy()
            y_coords_min_price_dashed_line_tolerance.append(max(y_coords_min_price_dashed_line) + 1)
            y_coords_min_price_dashed_line_tolerance.append(min(y_coords_min_price_dashed_line) - 1)
            outside_chart_x_coord_bis = min(
                x[
                    np.where(
                        ~np.isin(y, y_coords_min_price_dashed_line_tolerance)
                    )
                ]
            )

            if outside_chart_x_coord != outside_chart_x_coord_bis:
                raise Exception('Writings on RHS of the graph seem misaligned.')

            y, x = np.where(np.all(img_array == self.seller_colors[seller], axis=2))

            if len(x) == 0:
                print("No pixel with the relevant RGB code for the chart with seller ==", seller)
                print("--- Skipping the extraction of the corresponding price history.")

                empty_df = {'x_axis_coord': [], 'trusted': [], 'price': []}
                empty_df = pd.DataFrame(empty_df)

                self.price_histories[seller] = empty_df.copy()

                continue

            x_axis_coords = []
            avg_y_coords = []

            max_y_axis_coords = []
            min_y_axis_coords = []

            for x_axis_coord in np.unique(x):
                if x_axis_coord < outside_chart_x_coord_bis:
                    y_coords = y[np.where(x == x_axis_coord)]
                    y_coords = y_coords[y_coords < 2400 - avg_y_coord_min].copy()

                    if len(y_coords) == 0:
                        continue

                    avg_y_coord = img_array.shape[0] - y_coords.mean()

                    max_y_axis_coord = img_array.shape[0] - y_coords.min()
                    min_y_axis_coord = img_array.shape[0] - y_coords.max()

                    max_y_axis_coords.append(max_y_axis_coord)
                    min_y_axis_coords.append(min_y_axis_coord)
                    avg_y_coords.append(avg_y_coord)
                    x_axis_coords.append(x_axis_coord)

                else:
                    continue

            max_y_axis_coords = np.array(max_y_axis_coords)
            min_y_axis_coords = np.array(min_y_axis_coords)
            x_axis_coords = np.array(x_axis_coords)
            avg_y_coords = np.array(avg_y_coords)

            price_history = pd.DataFrame(
                [x_axis_coords, avg_y_coords, max_y_axis_coords, min_y_axis_coords],
                index=['x_axis_coord', 'avg_y_coords', 'max_y_axis_coord', 'min_y_axis_coord']
            ).T

            price_history['trusted'] = True

            for i in range(1, 5):
                price_history[f'lagged{i}_x_axis_coord'] = price_history['x_axis_coord'].shift(i)

                price_history[f'lagged{i}_x_axis_coord_match'] = (
                    price_history[f'lagged{i}_x_axis_coord'] == price_history['x_axis_coord'] - i
                )

                price_history['trusted'] = np.logical_and(
                    price_history['trusted'],
                    price_history[f'lagged{i}_x_axis_coord_match']
                )

            price_history['trusted'] = np.logical_or(
                price_history['trusted'],
                price_history['lagged4_x_axis_coord'].isnull()
            )

            price_history['trusted'] = np.logical_or(
                price_history['trusted'],
                price_history['max_y_axis_coord'] - price_history['min_y_axis_coord'] > 3
            )

            price_history = price_history.drop(
                columns=price_history.columns[
                    price_history.columns.map(lambda col: col.startswith('lagged'))
                ]
            )

            price_history = price_history.drop(columns=['max_y_axis_coord', 'min_y_axis_coord'])

            price_history['price'] = (
                self.min_prices[seller]
                + (price_history['avg_y_coords'] - avg_y_coord_min)
                / (avg_y_coord_max - avg_y_coord_min)
                * (self.max_prices[seller] - self.min_prices[seller])
            )

            price_history = price_history.drop(columns=['avg_y_coords'])

            self.price_histories[seller] = price_history.copy()

    def get_start_end_dates_from_charts(self):

        self.start_end_dates = copy.deepcopy(self.sellers)

        for seller, file_name in self.paths_to_charts.items():

            img = Image.open(file_name)

            img_array = np.array(img)

            text_extracts = []

            for k, multiplier in enumerate(
                [
                    1 / 2,   # Bottom half of the image
                    1 / 3,   # Bottom two thirds of the image
                    2 / 3,   # Bottom third of the image
                    3 / 4    # Bottom quarter of the image
                ]
            ):
                img_array_tmp = img_array[int(img_array.shape[0] * multiplier):, :, :].copy()

                img = Image.fromarray(img_array_tmp)

                text = pytesseract.image_to_string(img, lang='eng')

                # print(text)

                text_tmp = text[text.find('€'):text.rfind('Price type')]

                if len(text_tmp) == 0 and k == 2:
                    if len(text_extracts[0]) == 0 and len(text_extracts[1]) == 0:
                        text_tmp = text[:text.rfind('Price type')]

                text_extracts.append(text_tmp)

            first_months = []
            first_years = []
            last_months = []
            last_years = []

            for i, text_extract in enumerate(text_extracts):

                if i < len(text_extracts) - 1:

                    if len(text_extract) > 0:

                        try:

                            first_month, first_year, last_month, last_year = get_dates_from_text_extract(text_extract)

                            first_months.append(first_month)
                            first_years.append(first_year)
                            last_months.append(last_month)
                            last_years.append(last_year)

                        except IndexError:

                            continue

                    else:

                        continue

                else:

                    # Once we have analysed three text extracts, if at least one was successful, we break the iteration
                    if (
                        len(first_months) > 0
                        and len(first_years) > 0
                        and len(last_months) > 0
                        and len(last_years) > 0
                    ):

                        break

                    # If none was successful, we have to make another try
                    else:

                        first_month, first_year, last_month, last_year = get_dates_from_text_extract(text_extract)

                        first_months.append(first_month)
                        first_years.append(first_year)
                        last_months.append(last_month)
                        last_years.append(last_year)

            # For each component of the dates we are after, we take the value that comes out most often
            first_month = max(first_months, key=first_months.count)
            first_year = max(first_years, key=first_years.count)
            last_month = max(last_months, key=last_months.count)
            last_year = max(last_years, key=last_years.count)

            # Converting to dates
            first_date = datetime.strptime(' '.join([first_month, str(first_year)]), "%b %Y")
            last_date = datetime.strptime(' '.join([last_month, str(last_year)]), "%b %Y")

            # Storing results
            self.start_end_dates[seller]['first_date'] = first_date
            self.start_end_dates[seller]['last_date'] = last_date

    def convert_coordinates_to_timestamps(self):

        self.first_date_x_axis = copy.deepcopy(self.sellers)
        self.last_date_x_axis = copy.deepcopy(self.sellers)
        self.hour_increment = copy.deepcopy(self.sellers)

        for seller in self.price_histories.keys():

            first_date = self.start_end_dates[seller]['first_date']
            last_date = self.start_end_dates[seller]['last_date']

            self.first_date_x_axis[seller] = first_date
            self.last_date_x_axis[seller] = last_date

            final_last_date = last_date

            if final_last_date < pd.to_datetime("2024-01-01"):
                print("'Manually' correcting the final date for seller ==", seller)
                final_last_date = pd.to_datetime("2024-01-19" if seller == "amazon" else "2024-01-22")

            time_delta = final_last_date - first_date

            hour_increment = time_delta.days * 24 / (
                self.price_histories[seller]['x_axis_coord'].max()
                - self.price_histories[seller]['x_axis_coord'].min()
            )

            self.hour_increment[seller] = hour_increment

            all_time_deltas = pd.to_timedelta(
                (
                    self.price_histories[seller]['x_axis_coord'] - self.price_histories[seller]['x_axis_coord'].min()
                ) * hour_increment,
                unit='hours'
            )

            self.price_histories[seller]['date'] = all_time_deltas + first_date

            self.price_histories[seller] = self.price_histories[seller].drop(columns=['x_axis_coord'])

    def format_price_histories(self):

        for seller in self.price_histories.keys():

            self.price_histories[seller]['sku'] = self.sku_reference

            self.price_histories[seller]['country'] = self.country
            self.price_histories[seller]['unit'] = self.unit_correspondence[self.country]

            self.price_histories[seller]['seller'] = seller

    def format_price_history_summary(self):

        self.price_history_summary['sku'] = self.sku_reference

        seller_correspondence_inv = {v: k for k, v in self.seller_correspondence.items()}
        self.price_history_summary['seller'] = self.price_history_summary['Type de prix'].map(seller_correspondence_inv)
        self.price_history_summary = self.price_history_summary.drop(columns=['Type de prix'])

        self.price_history_summary['country'] = self.country
        self.price_history_summary['unit'] = self.unit_correspondence[self.country]

        self.price_history_summary['min_price'] = self.price_history_summary['Le plus bas jamais atteint *'].map(
            lambda string: float(
                re.findall(r'\d+,\d+', string.replace('.', ''))[0].replace(',', '.')
            ) if string != '-' else np.nan
        )
        self.price_history_summary['min_price_date'] = self.price_history_summary['Le plus bas jamais atteint *'].map(
            lambda string: datetime.strptime(clean_date_string(string), "%d %b %Y") if string != '-' else np.nan
        )

        self.price_history_summary['max_price'] = self.price_history_summary['Le plus élevé de tous les temps *'].map(
            lambda string: float(
                re.findall(r'\d+,\d+', string.replace('.', ''))[0].replace(',', '.')
            ) if string != '-' else np.nan
        )
        self.price_history_summary['max_price_date'] = self.price_history_summary[
            'Le plus élevé de tous les temps *'
        ].map(
            lambda string: datetime.strptime(clean_date_string(string), "%d %b %Y") if string != '-' else np.nan
        )

        self.price_history_summary['current_price'] = self.price_history_summary['Actuel +'].map(
            lambda string: float(
                re.findall(r'\d+,\d+', string.replace('.', ''))[0].replace(',', '.')
            ) if string != '-' else np.nan
        )
        self.price_history_summary['current_price_date'] = self.price_history_summary['Actuel +'].map(
            lambda string: datetime.strptime(clean_date_string(string), "%d %b %Y") if string != '-' else np.nan
        )

        self.price_history_summary['average_price'] = self.price_history_summary['Moyenne *'].map(
            lambda string: float(
                re.findall(r'\d+,\d+', string.replace('.', ''))[0].replace(',', '.')
            ) if string != '-' else np.nan
        )

        self.price_history_summary = self.price_history_summary[
            [
                'sku', 'country', 'unit', 'seller',
                'min_price', 'min_price_date', 'max_price', 'max_price_date', 'current_price', 'current_price_date',
                'average_price',
            ]
        ].copy()

    def format_product_data(self):

        product_data = self.product_data.set_index(0).T.copy()

        product_data['sku'] = self.sku_reference
        product_data['country'] = self.country

        product_data = product_data.rename(
            columns={
                'Groupe de produits': 'product_group',
                'Catégorie': 'category',
                'Fabricant': 'manufacturer',
                'Modèle': 'model',
                'Locale': 'local',
                'Prix de vente': 'selling_price',
                'EAN': 'EAN',
                'UPC': 'UPC',
                'Dernier scan de mise à jour': 'last_scan',
                'Dernier suivi': 'last_followed',
            }
        )

        product_data['selling_price'] = product_data['selling_price'].map(
            lambda string: float(
                re.findall(r'\d+,\d+', string.replace('.', ''))[0].replace(',', '.')
            ) if string != '-' else np.nan
        )

        self.product_data = product_data.copy()

    def build_collection_metadata(self):

        sellers = list(self.sellers.keys())

        collection_metadata = {
            'sku': [self.sku_reference] * len(sellers),
            'country': [self.country] * len(sellers),
            'seller': sellers,
            'collection_date': [date.today()] * len(sellers),
            'first_date_x_axis': [self.first_date_x_axis[seller] for seller in sellers],
            'last_date_x_axis': [self.last_date_x_axis[seller] for seller in sellers],
            'nb_x_axis_coords': [self.price_histories[seller].shape[0] for seller in sellers],
            'hour_increment': [self.hour_increment[seller] for seller in sellers],
        }

        self.collection_metadata = pd.DataFrame.from_dict(collection_metadata)

    def save_results(self):

        # Saving the price history
        price_history = pd.concat([v for _, v in self.price_histories.items()])
        price_history['trusted'] = price_history['trusted'].astype(bool)

        try:

            price_history_old = pd.read_csv(
                os.path.join(self.path_to_price_data, 'price_history.csv'),
                dtype={"trusted": bool, "price": float, "date": str, "sku": str, "country": str, "seller": str},
                parse_dates=['date']
            )

            price_history_old = price_history_old[
                np.logical_or(
                    price_history_old['sku'] != self.sku_reference,
                    price_history_old['country'] != self.country
                )
            ].copy()

            price_history = pd.concat([price_history_old, price_history])

            price_history.to_csv(os.path.join(self.path_to_price_data, 'price_history.csv'), index=False)

        except FileNotFoundError:

            price_history.to_csv(os.path.join(self.path_to_price_data, 'price_history.csv'), index=False)

        # Saving the price history summary
        try:

            price_history_summary_old = pd.read_csv(os.path.join(self.path_to_price_data, 'price_history_summary.csv'))

            price_history_summary_old = price_history_summary_old[
                np.logical_or(
                    price_history_summary_old['sku'] != self.sku_reference,
                    price_history_summary_old['country'] != self.country
                )
            ].copy()

            price_history_summary = pd.concat([price_history_summary_old, self.price_history_summary])

            price_history_summary.to_csv(
                os.path.join(self.path_to_price_data, 'price_history_summary.csv'),
                index=False
            )

        except FileNotFoundError:

            self.price_history_summary.to_csv(
                os.path.join(self.path_to_price_data, 'price_history_summary.csv'),
                index=False
            )

        # Saving product data
        try:

            product_data_old = pd.read_csv(os.path.join(self.path_to_price_data, 'product_data.csv'))

            product_data_old = product_data_old[
                np.logical_or(
                    product_data_old['sku'] != self.sku_reference,
                    product_data_old['country'] != self.country
                )
            ].copy()

            product_data = pd.concat([product_data_old, self.product_data])

            product_data.to_csv(
                os.path.join(self.path_to_price_data, 'product_data.csv'),
                index=False
            )

        except FileNotFoundError:

            self.product_data.to_csv(
                os.path.join(self.path_to_price_data, 'product_data.csv'),
                index=False
            )

        # Saving collection metadata
        try:

            collection_metadata_old = pd.read_csv(os.path.join(self.path_to_price_data, 'collection_metadata.csv'))

            collection_metadata_old = collection_metadata_old[
                np.logical_or(
                    collection_metadata_old['sku'] != self.sku_reference,
                    collection_metadata_old['country'] != self.country
                )
            ].copy()

            collection_metadata = pd.concat([collection_metadata_old, self.collection_metadata])

            collection_metadata.to_csv(
                os.path.join(self.path_to_price_data, 'collection_metadata.csv'),
                index=False
            )

        except FileNotFoundError:

            self.collection_metadata.to_csv(
                os.path.join(self.path_to_price_data, 'collection_metadata.csv'),
                index=False
            )


if __name__ == "__main__":

    data_raw_dir = "/Users/Paul-Emmanuel/Desktop/PhD/3_DST_incidence/data_raw"

    relevant_combinations = []

    for folder in os.listdir(data_raw_dir):

        folder = folder.strip("/")

        if folder.startswith("charts_amazon"):

            combination = {}

            folder_decomposed = folder.split("_")
            combination["country"] = folder_decomposed[2]

            charts = os.path.join(data_raw_dir, folder)
            combination["skus"] = [sku.strip(".png") for sku in os.listdir(charts) if sku.startswith("B")]

            relevant_combinations.append(combination)

        else:
            continue

    for combination in relevant_combinations:

        print("Moving to country:", combination["country"])
        print("-----------------------------------")

        nb_skus = len(np.unique(combination["skus"]))

        for i, sku in enumerate(np.unique(combination["skus"])):

            print(f'SKU: {sku} ({i + 1} / {nb_skus})')

            collector = PriceHistoryCollector(
                data_raw_dir=data_raw_dir,
                sku_reference=sku,
                country=combination["country"]
            )

            try:

                collector.collect_and_clean_data()

            except IndexError:

                print("--- Moving forward but error here!")

        print("===================================")
