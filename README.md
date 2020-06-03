# Project-4-Capstone



## Installations

This project was done in Python 3.7.5 on a linux machine. here are the libraries necessary for the project:

* NumPy (>= 1.17.4)
* Pandas (>= 0.25.3)
* Matplotlib (>= 3.1.2)
* Pycountry-convert (>=0.7.2)

for the web app, in addition to the above:

* Flask (>=1.1.1)
* Plotly (>=4.4.1)

    
### User installations

pip or conda could be used to install the libraries needed to run this project. As an example, the commands are run as follows to install latest version of Pandas:

```
pip install -U pandas
```
Or

```
conda install -U pandas
```
The commands can be used to install the remaining of the libraries



## Project Motivation

The pandemic has impacted many aspects of normal day-to-day life globally. Among these aspects is people's mobility. The goal of this project is to identify how far did mobility drop for each country and at which period. In addition, averages are taken for each continent to extract meaningful patterns, and correlation of mobility with the number cases is examined 


## File Descriptions

* Project_4_Mobility.ipynb: notebook with all steps from problem statement to conclusion

* Global_Mobility_Report.csv: Google maps aggregated data on mobility by country

* time_series_covid19_confirmed_global.csv: data on coronavirus cases for each country
* run.py: script to run web app
* methods.py: pieces of code from notebook necessary to visualize data in web app
* templates:
  * master_1.html: template for the web app

* README.md: this file

## Summary of Results

Islands with tourism as a main source of income witnessed the greatest drop in mobility. In addition, countries in Europe and South America come next in terms of drop in mobility. in terms of which country reached a min moblity first, Mongolia comes in first for all location types except for parks (Denmark is 1st). However, this is explained by the fact that the baseline is obtained from January, which is too cold for people in Scandinavian countries to go to the park. Hence, the baseline is not very meaningful here because more people will go to the park in the months after January.

As for contients, South America scores highest average drop in mobility for all location types, followed by North America. Africa scores lowest mobility drop for all location types except parks, grocery and pharmacy, and residential.

using spearman correlation, retail, recreation, and transit station have the greatest negative correlation with coronavirus cases. Parks has the least correlation.

## The Web App

After cloning the project, to run the web app cd to 'Project-4-Capstone' and run command:

```
python run.py
```

or
```
python3 run.py
```

After that open the web browser and go to http://0.0.0.0:3001 and the web app page should be loaded

This error might occur: ModuleNotFoundError: No module named 'numpy.testing.nosetester'.

if it does make sure all libraries mentioned above are installed to latest version. if the error is still present, run:

```
pip install -U scipy
```

## Acknowledgements

The mobility dataset was obtained from Google. (https://www.google.com/covid19/mobility/)
The coronavirus cases was obtained from HDX (https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases)
The web app code and template is a modified version of the code provided by Udacity (project 2)

