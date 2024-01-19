# ======================================================================================================================
# --- Housekeeping -----------------------------------------------------------------------------------------------------
# ======================================================================================================================

import os
path_to_dir = os.path.dirname(os.path.abspath(__file__))

import time

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from mimetypes import guess_extension
from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

from PIL import Image

import matplotlib.pyplot as plt

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract/5.2.0/bin/tesseract"

import re

from datetime import datetime, date

from utils import clean_date_string

# ======================================================================================================================
# --- Core Python class ------------------------------------------------------------------------------------------------
# ======================================================================================================================

class AmazonPriceCollector():

    def __init__(self, sku_reference, country, temp_folder):

        np.random.seed(int(time.time()))

        self.sku_reference = sku_reference
        self.country = country
        self.temp_folder = temp_folder

        self.height = 6200
        self.width = 12000
        self.base = "https://charts.camelcamelcamel.com/"

        ua = UserAgent()
        user_agent = ua.random

        options = Options()
        options.add_argument(f"window-size={np.random.randint(100, 750)},{np.random.randint(100, 750)}")
        options.add_argument("--auto-open-devtools-for-tabs")
        # options.add_argument(f'--user-agent={user_agent}')

        self.driver = webdriver.Chrome(chrome_options=options)

        self.sellers = {'amazon': {}}

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

        self.unit_correspondence = {'fr': '€'}

    def collect_and_clean_data(self):

        self.fetch_additional_information()
        self.fetch_price_history_charts()

        self.build_price_history_from_charts()
        self.get_start_end_dates_from_charts()
        self.get_end_date_from_price_history_summary()
        self.convert_coordinates_to_timestamps()

        self.format_price_histories()
        self.format_price_history_summary()
        self.format_product_data()
        self.build_collection_metadata()

        self.save_results()

        self.clean_everything()

    def fetch_additional_information(self):

        url_to_info = f"https://{self.country}.camelcamelcamel.com/product/{self.sku_reference}?context=search"

        self.driver.get(url_to_info)

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        if len(soup.find_all(class_="core-msg spacer")) > 0 and soup.find_all(class_="core-msg spacer")[0].text:
            time.sleep(15 * 60)

            soup = BeautifulSoup(self.driver.page_source, "html.parser")

        price_history_summary = pd.read_html(
            str(soup.find_all(class_='table-scroll camelegend')[0].find('table'))
        )[0]
        self.price_history_summary = price_history_summary.copy()

        min_price = price_history_summary.set_index(
            'Type de prix'
        ).loc[
            'Amazon', 'Le plus bas jamais atteint *'
        ]
        self.min_price = float(min_price[:min_price.find('€')].replace(',', '.'))

        max_price = price_history_summary.set_index(
            'Type de prix'
        ).loc[
            'Amazon', 'Le plus élevé de tous les temps *'
        ]
        self.max_price = float(max_price[:max_price.find('€')].replace(',', '.'))

        self.product_data = pd.read_html(str(soup.find_all(class_='product_fields')[0]))[0]

    def fetch_price_history_charts(self):

        del self.driver.requests

        time.sleep(np.random.randint(20, 30))

        self.paths_to_charts = self.sellers.copy()

        for seller in self.paths_to_charts.keys():

            url_to_img = self.base + self.country + "/" + self.sku_reference + "/" + f'{seller}.png'
            url_to_img += f"?force=1&zero=0&w={self.width}&h={self.height}&desired=false&legend=1&ilt=1&tp=all&fo=0&lang=fr_FR"

            self.driver.get(url_to_img)

            time.sleep(np.random.randint(20, 30))

            requests = pd.Series(self.driver.requests)

            request_headers = requests.map(
                lambda request: (
                    request.response.headers['Content-Type']
                    if request.response is not None
                    and isinstance(request.response.headers['Content-Type'], str)
                    else ''
                )
            )

            img_requests = requests[
                request_headers.map(
                    lambda request_header: guess_extension(
                        request_header.split(';')[0].strip()
                    ) == ".png"
                )
            ].copy()

            request = img_requests.iloc[len(img_requests) - 1]

            file_name = os.path.join(self.temp_folder, self.sku_reference + "_" + seller) + '.png'

            with open(file_name, 'wb') as file:
                file.write(request.response.body)

            self.paths_to_charts[seller] = file_name

    def build_price_history_from_charts(self):

        self.price_histories = self.sellers.copy()

        for seller, file_name in self.paths_to_charts.items():

            img = Image.open(file_name)
            img_array = np.array(img)

            # Pixels that have the RGB code for the maximum price
            y, x = np.where(np.all(img_array == [194, 68, 68, 255], axis=2))

            # Average y-coordinate of the left-most pixels with this RGB code
            avg_y_coord_max = img_array.shape[0] - y[np.where(x == min(x))].mean()

            # Minimum x-coordinate of the pixels with this RGB code outside of the horizontal dashed line
            outside_chart_x_coord = min(
                x[
                    np.where(
                        ~np.isin(y, y[np.where(x == min(x))])
                    )
                ]
            )

            # Pixels that have the RGB code for the minimum price
            y, x = np.where(np.all(img_array == [119, 195, 107, 255], axis=2))

            # Average y-coordinate of the left-most pixels with this RGB code
            avg_y_coord_min = img_array.shape[0] - y[np.where(x == min(x))].mean()

            # Minimum x-coordinate of the pixels with this RGB code outside of the horizontal dashed line
            outside_chart_x_coord_bis = min(
                x[
                    np.where(
                        ~np.isin(y, y[np.where(x == min(x))])
                    )
                ]
            )

            if outside_chart_x_coord != outside_chart_x_coord_bis:
                raise Exception('Writings on RHS of the graph seem misaligned.')

            y, x = np.where(np.all(img_array == self.seller_colors[seller], axis=2))

            x_axis_coords = []
            avg_y_coords = []

            for x_axis_coord in np.unique(x):
                if x_axis_coord < outside_chart_x_coord_bis:
                    y_coords = y[np.where(x == x_axis_coord)]
                    y_coords = y_coords[y_coords < 2400 - avg_y_coord_min].copy()

                    avg_y_coord = img_array.shape[0] - y_coords.mean()

                    avg_y_coords.append(avg_y_coord)
                    x_axis_coords.append(x_axis_coord)

                else:
                    continue

            x_axis_coords = np.array(x_axis_coords)
            avg_y_coords = np.array(avg_y_coords)

            prices = (
                self.min_price
                + (avg_y_coords - avg_y_coord_min)
                / (avg_y_coord_max - avg_y_coord_min)
                * (self.max_price - self.min_price)
            )

            price_history = pd.DataFrame([x_axis_coords, prices], index=['x_axis_coord', 'price']).T

            self.price_histories[seller] = price_history.copy()

    def get_start_end_dates_from_charts(self):

        self.start_end_dates = self.sellers.copy()

        for seller, file_name in self.paths_to_charts.items():

            img = Image.open(file_name)

            text = pytesseract.image_to_string(img, lang='eng')

            # print(text)

            text_extract = text[text.find('€'):text.rfind('Price type')]

            # Getting the first month
            months = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

            months_idx = []

            for month in months:
                months_idx.append(
                    text_extract.find(month) if text_extract.find(month) > 0 else np.nan
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
            first_year = 2000 + int(re.findall(r'[^€,](\d\d)', text_extract)[0])
            last_year = 2000 + int(re.findall(r'[^€,](\d\d)', text_extract)[-1])

            # Correcting first and last months if needed
            if text_extract.find(str(first_year - 2000)) < text_extract.find(first_month):
                first_month = 'Jan'
            if text_extract.rfind(str(last_year - 2000)) > text_extract.rfind(last_month):
                last_month = 'Jan'

            # Converting to dates
            first_date = datetime.strptime(' '.join([first_month, str(first_year)]), "%b %Y")
            last_date = datetime.strptime(' '.join([last_month, str(last_year)]), "%b %Y")

            # Storing results
            self.start_end_dates[seller]['first_date'] = first_date
            self.start_end_dates[seller]['last_date'] = last_date

    def get_end_date_from_price_history_summary(self):

        self.clean_end_dates = self.sellers.copy()

        for seller in self.clean_end_dates.keys():

            str_value = self.price_history_summary.set_index(
                'Type de prix'
            ).loc[
                self.seller_correspondence[seller], 'Actuel +'
            ]

            if str_value != '-':

                last_date_clean = datetime.strptime(
                    clean_date_string(str_value),
                    "%d %b %Y"
                )

                self.clean_end_dates[seller] = last_date_clean

            else:

                self.clean_end_dates[seller] = np.nan

    def convert_coordinates_to_timestamps(self):

        self.first_date_x_axis = self.sellers
        self.last_date_x_axis = self.sellers
        self.last_date_clean = self.sellers
        self.hour_increment = self.sellers

        for seller in self.price_histories.keys():

            first_date = self.start_end_dates[seller]['first_date']
            last_date = self.start_end_dates[seller]['last_date']

            last_date_clean = self.clean_end_dates[seller]

            self.first_date_x_axis[seller] = first_date
            self.last_date_x_axis[seller] = last_date
            self.last_date_clean[seller] = last_date_clean

            if isinstance(last_date_clean, float) and np.isnan(last_date_clean):
                final_last_date = last_date

            else:
                final_last_date = last_date_clean

            time_delta = final_last_date - first_date

            hour_increment = time_delta.days * 24 / len(self.price_histories[seller])

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
                'average_price'
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
            'last_date_clean': [self.last_date_clean[seller] for seller in sellers],
            'nb_x_axis_coords': [self.price_histories[seller].shape[0] for seller in sellers],
            'hour_increment': [self.hour_increment[seller] for seller in sellers],
        }

        self.collection_metadata = pd.DataFrame.from_dict(collection_metadata)

    def save_results(self):

        # Saving the price history
        price_history = pd.concat([v for _, v in self.price_histories.items()])

        try:

            price_history_old = pd.read_csv(os.path.join(self.path_to_price_data, 'price_history.csv'))

            price_history_old = price_history_old[
                np.logical_or(
                    price_history_old['sku'] != self.sku_reference,
                    price_history_old['country'] != self.country
                )
            ].copy()

            price_history = pd.concat([price_history_old, price_history])

            price_history.to_csv(os.path.join(self.path_to_price_data, 'price_history.csv'), index=False)

        except:

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

        except:

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

        except:

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

        except:

            self.collection_metadata.to_csv(
                os.path.join(self.path_to_price_data, 'collection_metadata.csv'),
                index=False
            )

    def clean_everything(self):

        self.driver.quit()

        # for _, file_name in self.paths_to_charts.items():

        #     os.system(f"rm {file_name}")


if __name__ == "__main__":

    collector = AmazonPriceCollector(
        sku_reference="B000CRBEJ2",
        country="fr",
        temp_folder="/Users/Paul-Emmanuel/Desktop/PhD/3_DST_incidence/dst_incidence/notebooks/temp"
    )

    collector.collect_and_clean_data()

    print(collector.price_histories['amazon'].head())
