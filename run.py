import json
import plotly
import pandas as pd

import methods as m

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Figure, Scatter, Table
import plotly.express as px



app = Flask(__name__)


# assign dataset to 'df' (for simplification)
df = m.mobil_df_new



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    '''# extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # find the 10 most common categories
    #get df_cat with category cols only
    df_cat= df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # Sum all cats, sort and get largest 10 vals
    ranks=df_cat.sum().sort_values(ascending= False)[:10]
    rank_vals= ranks.values
    col_names= list(ranks.index)'''
    
    # list of columns of interest in dataset
    cols= list(m.mobil_df_new.columns[4:-1])
    
    #graphs to graph
    graphs=[]
    
    # TODO: show period at which countries reached a min mobility
    
    # get timeline values for countries (using min week method)
    X,Y= m.show_timeline(cols=cols, dates= m.dates_df_week.dropna())
    
    # prepare figure for a scatter plot
    fig= Figure()
    
    # to remove long form of column
    clean_str= lambda s: s.replace('_percent_change_from_baseline', '')
    
    # iterate through everly location type
    for i, col in enumerate(cols):
        
        # add location type to figure
        fig.add_trace(Scatter(x=X[i], y=Y[i], mode='markers', name=clean_str(col)))
        
    
    # add title and labels
    fig.update_layout(
        title="Dates at which countries reached slowest mobility",
        xaxis_title="dates",
        yaxis_title="col seperator (this is a 1-d graph)")
    
    # add fig to graphs
    graphs.append(fig)
    
    '''Now to show dates of min for continents (average)'''
    
    # TODO: show period at which continents reached a min mobility
    
    continent_dates= m.get_continent_mean_date(df=m.dates_df_week, continent_col=m.min_week_df.continent)
    
    # get timeline values for continents 
    X,Y= m.show_timeline(cols=cols, dates= continent_dates)
    
    # prepare figure for a scatter plot
    fig= Figure()
    
    # iterate through everly location type
    for i, col in enumerate(cols):
        
        # add location type to figure
        fig.add_trace(Scatter(x=X[i], y=Y[i], mode='markers', name=clean_str(col)))
        
    
    # add title and labels
    fig.update_layout(
        title="Mean Dates at which continents reached slowest mobility",
        xaxis_title="dates",
        yaxis_title="col seperator (this is a 1-d graph)")
    
    # add fig to graphs
    graphs.append(fig)
    
    
    # TODO: graph mean of slowest mobility drop for each continent
    
    # get the means for each col
    continent_mean_df=m.get_continent_mean(df=m.min_week_df)
    
    # prepare a new fig
    fig= Figure()   
    
    # y axis is constant
    y=list(continent_mean_df.index)
    
    
    for col in continent_mean_df:
        
        # get col vals
        x= continent_mean_df[col].values
    
        fig.add_trace(Bar(
            x=x,
            y=y,
            orientation='h',
            name=clean_str(col)))
        
    # add title and labels
    fig.update_layout(
        title="Mean of min mobility for each continent",
        xaxis_title="percent change from baseline",
        yaxis_title="continent")
    
    # add to graph
    graphs.append(fig)
    
    
    # TODO: show countries with earliest and latest min mobility
    
    # get dates col with columns in short form
    dates_df_week_clean=m.clean_col_name(m.dates_df_week)
    
    # countries with earliest mobility slow-down (starts with earliest country)
    ranks_earliest=dates_df_week_clean.apply(m.earliest_countries, axis=0).transpose()
    
    # create table
    fig = Figure(data=[Table(header=dict(values=list(ranks_earliest.index)),
                 cells=dict(values=ranks_earliest.values))
                     ])
    
    # add title
    fig.update_layout(
    title="Countries with earliest min mobility")
        
    # add table to graphs
    graphs.append(fig)
    
    
    # countries with latest mobility slow-down (starts with earliest country)
    ranks_latest=dates_df_week_clean.apply(m.latest_countries, axis=0).transpose()
    
    #create table
    fig = Figure(data=[Table(header=dict(values=list(ranks_latest.index)),
                 cells=dict(values=ranks_latest.values))
                     ])
    # add title
    fig.update_layout(
    title="Countries with latest min mobility")
    
    # add table to graphs
    graphs.append(fig)
    
    
    # TODO: get a table of countries with most and least mobility drop
    # modify min_week_df to have only baseline cols, and to have residential be consistent with other lambda functions
    min_week_df_mod= m.min_week_df.copy()

    # flip sign of values
    min_week_df_mod.residential_percent_change_from_baseline=-m.min_week_df.residential_percent_change_from_baseline

    # select columns of interest
    min_week_df_mod= min_week_df_mod[min_week_df_mod.columns[3:]]

    # 10 countries with greatest mobility drop
    rank_highest= m.clean_col_name(min_week_df_mod).apply(m.get_first_n).transpose()
        
    #create table
    fig = Figure(data=[Table(header=dict(values=list(rank_highest.index)),
                 cells=dict(values=rank_highest.values))
                     ])
    # add title
    fig.update_layout(
    title="Countries that experienced greatest mobility drop")
    
    # add table to graphs
    graphs.append(fig)
    
    
    # 10 countries with least mobility drop
    rank_least= m.clean_col_name(min_week_df_mod).apply(m.get_last_n).transpose()
    
        #create table
    fig = Figure(data=[Table(header=dict(values=list(rank_least.index)),
                 cells=dict(values=rank_least.values))
                     ])
    # add title
    fig.update_layout(
    title="Countries that experienced least mobility drop")
    
    # add table to graphs
    graphs.append(fig)
    
    
    
    
    # TODO: graph countries with most and least mobility drop
    
    # iterate through every location type col
    cols= list(m.mobil_df_new.columns[4:-1])
    
    for col in cols:
        
        # get a sorted Series largest countries
        sort=m.sort(col, df=m.min_week_df.dropna())
        
        # get x values
        x_1=sort.head(10).values
        x_2= sort.tail(10).values
        title_1='{}: countries with greatest drop in mobility'.format(clean_str(col))
        
        # get y values
        y_1=sort.head(10).index
        y_2= sort.tail(10).index
        title_2='{}: countries with least drop in mobility'.format(clean_str(col))
        
        if col!='residential_percent_change_from_baseline':
            
            graph_1= {
                        'data': [
                            Bar(
                                x=x_1,
                                y=y_1,
                                orientation='h'
                            )
                        ],

                        'layout': {
                            'title': title_1,
                            'yaxis': {
                                'title': "Country"
                            },
                            'xaxis': {
                                'title': "Percent change from baseline"
                            }
                        }
                    }


            graph_2= {
                        'data': [
                            Bar(
                                x=x_2,
                                y=y_2,
                                orientation='h'
                            )
                        ],

                        'layout': {
                            'title': title_2,
                            'yaxis': {
                                'title': "Country"
                            },
                            'xaxis': {
                                'title': "Percent change from baseline"
                            }
                        }
                    }

        
        else:    
        # flip title for 'residential'
            graph_1= {
                        'data': [
                            Bar(
                                x=x_1,
                                y=y_1,
                                orientation='h'
                            )
                        ],

                        'layout': {
                            'title': title_2,
                            'yaxis': {
                                'title': "Country"
                            },
                            'xaxis': {
                                'title': "Percent change from baseline"
                            }
                        }
                    }


            graph_2= {
                        'data': [
                            Bar(
                                x=x_2,
                                y=y_2,
                                orientation='h'
                            )
                        ],

                        'layout': {
                            'title': title_1,
                            'yaxis': {
                                'title': "Country"
                            },
                            'xaxis': {
                                'title': "Percent_change_from baseline"
                            }
                        }
                    }
    
        
        
        
        
        # add graph to graphs
        graphs.append(graph_1)
        graphs.append(graph_2)
        

    # spearman correlation
    corr=m.mobil_df_new.corrwith(m.mobil_df_new.corona_cases, method='spearman')
    
    cols= list(map(clean_str, corr.index))
    # get a table of correlations with coronavirus cases
    fig = Figure(data=[Table(header=dict(values=cols),
                 cells=dict(values=corr.values))
                     ])
    
     # add title
    fig.update_layout(
    title="correlations with coronavirus cases")
    
    # add to graphs
    graphs.append(fig)
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master_1.html', ids=ids, graphJSON=graphJSON)

'''
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )'''


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()