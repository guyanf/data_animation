# _*_ coding: utf-8 _*_
# /usr/bin/env python

"""
Author: Thomas Chen
Email: guyanf@gmail.com
Company: Thomas

date: 2025/1/18 22:36
desc:
"""

import time
import requests
import logging
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
# from pytz import country_names

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='down.log',
    filemode='w'
)


def download_data():
    # for year in range(1960, 2024):
    for year in range(1960, 2024):
        time.sleep(3)
        with open(f"./download/{year}.txt", "w") as w:

            url = f"https://www.kylc.com/stats/global/yearly/g_gdp/{year}.html"
            response = requests.get(url)
            if response.status_code == 200:
                # print(response)
                soup = BeautifulSoup(response.content, "html.parser")
                for tr in soup.find_all("tr"):
                    tds = tr.find_all("td")
                    lst_td = [td.text for td in tds]
                    if len(lst_td) and lst_td[0].isnumeric():
                        lst_td.insert(0, str(year))
                        logging.info(f"{lst_td}")
                        w.write(f"{'---'.join(lst_td)}\n")
            else:
                print(f"Failed to fetch data: {response.status_code}")


def prepare_data():
    out_file = "./data/gdp.csv"
    country_names = "/python-tools/ai_study/study-chart/utils/cn_name.txt"
    df_countrys = pd.read_csv(country_names, delimiter="\t", engine="python")
    # logging.info(df_countrys)
    f_path = "./download"
    in_path = Path(f_path)
    # with open(out_file, "a") as w:
    lst_column = ["year", "seq", "country", "continent", "gdp_anno", "raise"]
    lst_top_name = []
    df_countrys["cnname"] = df_countrys["cnname"].str.strip(" ")
    logging.info(df_countrys)

    out_lst_column = ["year", "seq", "country", "gdp_anno"]
    out_df = pd.DataFrame(columns=out_lst_column)
    logging.info(out_df)

    for file in in_path.glob("*.txt"):
        df = pd.read_csv(file, delimiter="---", engine="python", header=None)
        df.columns = lst_column

        df['gdp_anno'] = df['gdp_anno'].str.extract(r'\((.*?)\)', expand=False)
        df = df.drop(["continent", "raise"], axis = 1)
        # logging.info(df)

        out_df = pd.concat([out_df, df], ignore_index=True)
        for name in df.loc[df.query('seq <= 15').index , 'country']:
            if name not in lst_top_name:
                lst_top_name.append(name)
                # logging.info(f"{name}")
                # if name not in lst_countrys:
                #     print(name)


    logging.info(",".join(lst_top_name))
    logging.info(out_df)
    out_df = out_df.loc[out_df['country'].isin(lst_top_name)]
    # df['A'].isin(my_list)
    out_df = out_df.sort_values(by=['year', "seq"], ascending=[True, True])

    out_df["year"] = out_df["year"].astype(str) +"-01-01"
    logging.info(out_df)

    df_country_code = df_countrys[["cnname", "iso3"]]
    df_country_code = df_country_code.rename(columns={"cnname": "country", "iso3": "code"})
    logging.info(df_country_code)

    out_df = pd.merge(out_df, df_country_code, on='country', how='inner')

    out_df = out_df.rename(columns={
        "country": "name",
        "year": "datestamp",
        "gdp_anno": "date_value"
    })

    out_df = out_df.drop(["seq"], axis=1)

    out_df['datestamp'] = pd.to_datetime(out_df['datestamp'])
    out_df['date_value'] = out_df['date_value'].str.replace(',', '').astype(int)
    out_df['date_value'] = out_df['date_value'] / 1000000000

    out_df = out_df.round({'date_value': 2})
    # out_df['date_value'] = pd.to_numeric(out_df['date_value'])

    logging.info(out_df)
    out_df.to_csv(out_file, index=False)


def main():
    download_data()
    prepare_data()


if __name__ == '__main__':
    main()
