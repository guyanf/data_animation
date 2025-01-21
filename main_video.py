# _*_ coding: utf-8 _*_
# /usr/bin/env python

"""
Author: Thomas Chen
Email: guyanf@gmail.com
Company: Thomas

date: 2025/1/6 13:12
desc:
"""

import logging
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime
from PIL import Image
from matplotlib.transforms import Affine2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Arc, Circle
from matplotlib.patheffects import withStroke

from bar_chart_race import  prepare_long_data
from bar_chart_race._colormaps import colormaps

plt.rcParams['font.family'] = 'Microsoft YaHei'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='video.log',
    filemode='w'
)

def top_n_boolean_list(data, n=10):
    if len(data) <= n:
        return [True] * len(data)  # If the list has fewer than n elements, all are True
    # Get the top n unique values
    top_values = sorted(data, reverse=True)[:n]
    # Mark values in the original list
    return [val in top_values for val in data]

class MapAnimation:

    def __init__(self, f_config):
        dct_config = json.load(open(f_config, 'r'))
        print(dct_config.keys())
        file_path = dct_config['file_path']
        self.country_path = Path(file_path["country_code"])
        self.csv_path = Path(file_path["csv_path"])
        self.flag_path = Path(file_path["flag_path"])
        self.out_ani_path = Path(file_path["out_ani_path"])
        self.out_gif_path = Path(file_path["out_gif_path"])

        self.check_path()

        basemap = dct_config["basemap"]
        self.figSize = tuple(basemap['figsize'])
        self.backGroundImg = basemap['background']
        self.backGroundAlpha = basemap['backgroundalpha']
        self.titleImg = basemap['titleimage']
        self.titleCoord = basemap['titlecoord']
        self.titleSize = basemap['titlesize']
        self.titleAlpha = basemap['titlealpha']
        self.axisColor = basemap['axiscolor']
        self.xAxisLabel = basemap['xaxislabel']
        self.dataSource = basemap['datasource']
        self.dataSourceColor = basemap['datasourcecolor']
        self.dataSourceSize = basemap['datasourcesize']

        yearinfo = dct_config['yearinfo']
        self.yearLocation = yearinfo['location']
        self.yearFontColor = yearinfo['fontcolor']
        self.yearCircleColor = yearinfo['circlecolor']

        bar = dct_config['bar']
        self.barCmap = bar["cmap"]
        self.barTops = bar["tops_bar"]
        self.barHeight = bar["height"]
        self.barAlpha = bar["alpha"]
        self.barFlagSize = bar["flagsize"]

        video = dct_config['video']
        self.fps = video['fps']
        self.period = int(video['period'])
        self.minDate = pd.to_datetime(video['mindate'], format='%Y-%m-%d')
        self.maxDate = pd.to_datetime(video['maxdate'], format='%Y-%m-%d')
        self.isOutVideo = video['out_video']
        self.isOutGif = video['out_gif']

        self.language = dct_config["language"]

    def check_path(self):
        if not self.csv_path.exists():
            print("File not found")
            exit()

        if not self.flag_path.exists():
            print("Flags not found")
            exit()

        if not self.out_ani_path.parent.exists():
            self.out_ani_path.parent.mkdir(parents=True)

    def get_col_filt(self, df_values, df_ranks):
        col_filt = pd.Series([True] * df_values.shape[1])
        # print(col_filt)
        if self.barTops < df_ranks.shape[1]:
            if True:
                col_filt = (df_ranks > 0).any()
            # else:
                # col_filt = (df_ranks < self.barTops + .99).any()

            # if True and not col_filt.all():
            #     df_values = df_values.loc[:, col_filt]
            #     df_ranks = df_ranks.loc[:, col_filt]

        return col_filt

    def load_flag_images(self, df):
        """Load and resize all flag images once at startup"""
        flag_images = {}
        for iso_code in df['code'].unique():
            try:
                img = Image.open(self.flag_path / f"{iso_code}.png")
                img.thumbnail(self.barFlagSize, Image.Resampling.LANCZOS)
                flag_images[iso_code] = img
            except Exception as e:
                logging.error(f"Could not load flag for {iso_code}, {e}")
        return flag_images

    @staticmethod
    def get_bar_colors(cmap, colums):
        if cmap is None:
            cmap = 'dark12'
            if colums > 12:
                cmap = 'dark24'
        # print(cmap)
        if isinstance(cmap, str):
            try:
                bar_colors = colormaps[cmap.lower()]
            except KeyError:
                raise KeyError(f'Colormap {cmap} does not exist. Here are the '
                               f'possible colormaps: {colormaps.keys()}')
        # elif isinstance(cmap, colors.Colormap):
        #     bar_colors = cmap(range(cmap.N)).tolist()
        elif isinstance(cmap, list):
            bar_colors = cmap
        elif isinstance(cmap, tuple):
            bar_colors = list(cmap)
        elif hasattr(cmap, 'tolist'):
            bar_colors = cmap.tolist()
        else:
            raise TypeError('`cmap` must be a string name of a colormap, a matplotlib colormap '
                            'instance, list, or tuple of colors')

        # bar_colors is now a list
        n = len(bar_colors)
        # orig_bar_colors = bar_colors
        if colums > n:
            bar_colors = bar_colors * (colums // n + 1)
        bar_colors = np.array(bar_colors[:colums])

        return bar_colors

    def setup_plot_style(self, ax):
        """Apply consistent plot styling"""
        ax.set_yticks([])
        ax.set_xlabel(f'{self.xAxisLabel}\n{self.dataSource}', color=self.dataSourceColor, fontsize=self.dataSourceSize)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.set_title(f'Top {self.barTops} Populations', pad=20, fontsize=14, fontweight='bold')

    def add_timestamp_text(self, ax, circle_out, circle_in, current_time):
        ax.add_patch(circle_out)
        ax.add_patch(circle_in)

        """Add year and month text to plot"""
        plt.text(self.yearLocation[0], self.yearLocation[1], f'{current_time.year}',
                 fontsize=12, 
                 fontweight='bold', 
                 color=self.yearFontColor,
                 horizontalalignment='center',  
                 verticalalignment='center',
                 transform=ax.transAxes
                 )

    def year_figure(self, ax):
        circle_out = Circle(self.yearLocation, 0.08,
                            edgecolor=self.yearCircleColor,
                            facecolor='None',
                            linewidth=1,
                            transform=ax.transAxes
                            )

        circle_in = Circle(self.yearLocation, 0.07,
                           edgecolor=self.yearCircleColor,
                           facecolor='None',
                           linewidth=1,
                           transform=ax.transAxes
                           )

        arc = Arc(self.yearLocation, 0.147, 0.147,
                  theta1=0, theta2=45,
                  lw=4.5,
                  color=self.yearCircleColor,
                  transform=ax.transAxes
                  )
        return circle_out, circle_in, arc

    def create_animation(self, df, dict_country_code):

        df_values, df_ranks = prepare_long_data(df,
                                                index='datestamp',
                                                columns='code',
                                                values='date_value',
                                                steps_per_period=self.period
                                                )

        df_values = df_values.fillna(0)
        df_ranks = df_ranks.fillna(0)
        logging.info(df_values)
        logging.info(df_ranks)

        col_filt = self.get_col_filt(df_values, df_ranks)
        # print(col_filt)

        len_rows = len(df_values.index)

        lst_out_idx = []
        for _ in range(int(len_rows/self.period) + 1):
            lst_out_idx.extend([x for x in range(self.period)])

        df_values["temp_day"] = lst_out_idx[:len_rows]
        logging.info(df_values)

        df_values.index = df_values.index + pd.to_timedelta(df_values["temp_day"], unit='D')
        df_values.drop('temp_day', axis=1, inplace=True)

        df_ranks["temp_day"] = lst_out_idx[:len_rows]
        df_ranks.index = df_ranks.index + pd.to_timedelta(df_ranks["temp_day"], unit='D')
        df_ranks.drop('temp_day', axis=1, inplace=True)
        bar_colors = self.get_bar_colors(self.barCmap, colums=len(df_values.columns))
        # print(bar_colors)

        flag_images = self.load_flag_images(df)

        # Initial setup
        fig, ax = plt.subplots(figsize=self.figSize)
        ax.set_facecolor('None')

        # insert background
        if Path(self.backGroundImg).is_file():
            image_basemap = Image.open(self.backGroundImg)
            fig_width, fig_height = fig.get_size_inches() * fig.dpi
            image_basemap.thumbnail((fig_width, fig_height), Image.Resampling.LANCZOS)
            fig.figimage(image_basemap, xo=0, yo=0, alpha=self.backGroundAlpha, zorder=-1)
        #
        if Path(self.titleImg).is_file():
            image_title = Image.open(self.titleImg)
            image_title.thumbnail(self.titleSize, Image.Resampling.LANCZOS)
            fig.figimage(image_title, xo=self.titleCoord[0], yo=self.titleCoord[1], alpha=self.titleAlpha, zorder=0)

        # box bound, show left and bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.axisColor)
        ax.spines['bottom'].set_color(self.axisColor)

        # ax.tick_params(axis='x', labelcolor='gray')
        # ax.tick_params(axis='y', labelcolor='gray')
        ax.tick_params(colors=self.axisColor)
        # x y axis label show
        ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)

        circle_out, circle_in, arc = self.year_figure(ax)

        def animate(frame):
            ax.clear()
            # Create base bars
            bar_location = df_ranks.iloc[frame].values
            top_filt = top_n_boolean_list(bar_location, n=self.barTops)
            bar_location = bar_location[top_filt]
            bar_length = df_values.iloc[frame].values[top_filt]
            bar_length = [x for _, x in sorted(zip(bar_location, bar_length))]

            # top_filt = (bar_location > 0) & (bar_location < self.n_bars + 1)
            # colors = self.bar_colors[top_filt]

            cols = df_values.columns[top_filt]
            # logging.error(f"frame: {frame} cols: {list(cols)}")

            cols = [x for _, x in sorted(zip(bar_location, cols))]

            cur_colors = bar_colors[top_filt]
            cur_colors = [x for _, x in sorted(zip(bar_location, cur_colors))]

            # logging.info(f"frame: {frame} top_filt:{top_filt} cols:{cols}, cur_colors:{cur_colors}")

            y_min = min(bar_location)
            bar_location = sorted(bar_location)

            ax.barh(bar_location, bar_length, height=self.barHeight, tick_label=list(cols),
                    color=cur_colors, alpha=self.barAlpha)
            # Add visual elements for each country
            for i, row in enumerate(cols):
                # Add flag
                if row in flag_images:
                    img_box = OffsetImage(flag_images[row], zoom=0.7)
                    # ab = AnnotationBbox(img_box, (0, y_min + i),
                    ab = AnnotationBbox(img_box, (0, bar_location[i]),
                                        frameon=False,
                                        box_alignment=(0, 0.5),
                                        xybox=(-35, 0),
                                        xycoords="data",  # ('data', 'data'),
                                        boxcoords="offset points"
                                        )
                    ax.add_artist(ab)

                ax.text(bar_length[i], bar_location[i],
                        f' {bar_length[i]:,.2f}',
                        va='center', ha='left')

                cur_text = ax.text(bar_length[i] - 1, bar_location[i],
                        f' {dict_country_code[row]} ',
                        fontsize=8,
                        va='center', ha='right', fontweight='bold')

                cur_text.set_path_effects([
                    withStroke(linewidth=1, foreground='white')  # Black outline
                ])

                rotation_transform = Affine2D().rotate_deg_around(self.yearLocation[0], self.yearLocation[1], -frame*3)
                # Combine with ax.transAxes to maintain proportionate coordinates
                arc.set_transform(rotation_transform + ax.transAxes)

            ax.add_patch(arc)

            # Style the plot
            self.setup_plot_style(ax)
            self.add_timestamp_text(ax, circle_out, circle_in, df_values.index[frame])

            if frame == len_rows - 1:
                # Pause for 2 seconds at the last frame
                time.sleep(2)

        # Create and return animation
        return animation.FuncAnimation(
            fig, func=animate, frames=range(len_rows),
            interval=250, repeat=False
        )

    def get_country_info(self):
        df_country_code = pd.read_csv(self.country_path, sep='\t')
        if self.language == 'zh':
            dct_country_code = pd.Series(df_country_code.cnname.values, index=df_country_code.iso3).to_dict()
        elif self.language == 'en':
            dct_country_code = pd.Series(df_country_code.enname.values, index=df_country_code.iso3).to_dict()
        else:
            dct_country_code = pd.Series(df_country_code.iso3.values, index=df_country_code.iso3).to_dict()

        return dct_country_code


    def run(self):
        dct_country_code = self.get_country_info()
        # df = self.prepare_df()
        df = pd.read_csv(self.csv_path)
        df['datestamp'] = pd.to_datetime(df['datestamp'])

        logging.info(df.dtypes)

        # df['datestamp'] =  pd.to_datetime(df['datestamp'])
        # df['date_value'] =  pd.to_numeric(df['date_value'])


        # filter year
        df = df.query("datestamp < @self.maxDate and datestamp >= @self.minDate")
        logging.info(df)
        anim = self.create_animation(df, dct_country_code)
        if self.isOutVideo:
            anim.save(self.out_ani_path, writer='ffmpeg', fps=self.fps)
        if self.isOutGif:
            anim.save(self.out_gif_path, writer=animation.PillowWriter())


def main():
    print("go", datetime.now())
    f_config = "./config_video.json"
    obj = MapAnimation(f_config)
    obj.run()
    print("over", datetime.now())


if __name__ == "__main__":
    main()
