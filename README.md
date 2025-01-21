
# Data_Animation

## Introduction
Realize dynamic display of historical data, form initialization templates, and quickly build animations of various data ranking changes.

## Function

- Use requests module for download data and initialization data; 
- Based on the downloaded data of various types, the bar_chart_race library is used to expand and process the downloaded and sorted data, and the animation of GDP ranking changes is drawn using matplotlib's animation;
- Use the Flask framework to build a web server to display and download animation data.

```commandline
python3 download_gdp.py
python3 main_video.py
python3 app.py
```

![image](./static/images/gdp.gif)

## Require:

### For Download Data

requests  
bs4  
numpy  
pandas

### For Create Animation

matplotlib  
PIL  
bar_chart_race

### For Show Video

Flask  
