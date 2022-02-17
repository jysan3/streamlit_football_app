''' Streamlit football application
Author: Julien Lefaure
Version: v1.0
'''

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import folium
#from streamlit_folium import folium_static

#pip freeze > requirements.txt

# Set page config
st.set_page_config(page_title='International football results from 1872 to 2021',
                   page_icon=':soccer:',
                   initial_sidebar_state='expanded')


@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/martj42/international_results/master/results.csv")
    return df


df = load_data()


@st.cache
def load_data():
    cities_df = pd.read_csv("worldcities.csv")
    return cities_df


cities_df = load_data()


def main():
    st.title("International football results from 1872 to 2021")
    st.markdown("""
      This app performs simple data visualisations of international football results from 1872 to 2021.
      * **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn, folium
      * **Data source:** [kaggle.com](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017).
      """)

    # Data pre-processing

    # check the null values
    df.drop(df[df.isna().any(axis=1)].index, inplace=True)

    # create new columns
    df['outcome'] = df.apply(lambda x: 'H' if x['home_score'] > x['away_score']
    else ('A' if x['home_score'] < x['away_score'] else 'D'),
                             axis=1)
    df['winning_team'] = df.apply(lambda x: x['home_team'] if x['home_score'] > x['away_score']
    else (x['away_team'] if x['home_score'] < x['away_score'] else np.nan),
                                  axis=1)
    df['losing_team'] = df.apply(lambda x: x['away_team'] if x['home_score'] > x['away_score']
    else (x['home_team'] if x['home_score'] < x['away_score'] else np.nan),
                                 axis=1)
    df['total_goals'] = df['home_score'] + df['away_score']
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['decade'] = df['year'] - df['year'] % 10
    df['month'] = pd.DatetimeIndex(df['date']).month_name()

    # load list of teams belonging to each FIFA confederation

    afc = [team.strip() for team in open('conf/AFC', 'r').read().split('\n')]
    caf = [team.strip() for team in open('conf/CAF', 'r').read().strip().split('\n')]
    concacaf = [team.strip() for team in open('conf/Concacaf', 'r').read().strip().split('\n')]
    conmebol = [team.strip() for team in open('conf/Conmebol', 'r').read().strip().split('\n')]
    ofc = [team.strip() for team in open('conf/OFC', 'r').read().strip().split('\n')]
    uefa = [team.strip() for team in open('conf/UEFA', 'r').read().strip().split('\n')]

    # pre-processing on cities.csv

    cities_df['number_of_games'] = df.groupby(['city'])['city'].transform('count')
    cities_df['number_of_games'] = cities_df['number_of_games'].fillna(value=0)
    cities_df['number_of_games'] = cities_df['number_of_games'].astype(int)

    # new dataframe of confederations

    conf_df = pd.DataFrame({
        'affiliation': ['AFC'] * len(afc) + ['CAF'] * len(caf) + ['CONCACAF'] * len(concacaf) + \
                       ['CONMEBOL'] * len(conmebol) + ['OFC'] * len(ofc) + ['UEFA'] * len(uefa),
        'region': ['Asia'] * len(afc) + ['Africa'] * len(caf) + ['North and Central America'] * len(concacaf) + \
                  ['South America'] * len(conmebol) + ['Oceania'] * len(ofc) + ['Europe'] * len(uefa),
        'team': afc + caf + concacaf + conmebol + ofc + uefa
    })

    teams = list(df['home_team'].unique())
    teams.extend(list(df['away_team'].unique()))
    teams_df = pd.DataFrame({'team': list(set(teams))})
    teams_df['first_game_year'] = teams_df['team'].apply(
        lambda x: df[(df['home_team'] == x) | (df['away_team'] == x)].head(1)['year'].values[0])
    teams_df = teams_df.merge(conf_df, how='left', on=['team'])
    teams_df.sort_values(by='first_game_year', inplace=True)

    # Add subtitle to the main interface

    st.video('video.mp4', format="video/mp4", start_time=0)
    st.markdown(
        "Dataset summary : over 30000 results from international men's football matches from 1972 to 2021. It "
        "includes competitions and friendly matches.")
    st.markdown("""**Features selection**
                """)
    st.markdown(""" 
                |     **date**     |  **home_team**  |  **away_team**  |   **home_score**   |    **away_score**   | **tournament** |
|:----------------:|:---------------:|:---------------:|:------------------:|:-------------------:|:--------------:|
| **winning_team** | **losing_team** | **total_goals** |      **year**      |      **decade**     |    **month**   |
|     **city**     |   **country**   |   **neutral**   | **confederations** | **number_of_games** |   **outcome**  |
                 """)
    st.markdown('##')

    # Sidebar menu

    st.sidebar.title('Steps and data visualization selection')

    menu1 = st.sidebar.radio('Here you can select dataframes / data visualizations you want :',
                             ['Dataframes presentation', 'Number of matches per year', 'Cities host',
                              'Most games per tournament', 'Goals scored per match'])

    if menu1 == 'Dataframes presentation':
        menu2 = st.sidebar.radio('Dataframes selection',
                                 ['Overall dataframe', 'Teams dataframe', 'Confederations dataframe'])
        if menu2 == 'Overall dataframe':
            city = df['city'].unique()
            st.selectbox('Choose a city', city)
            st.write('Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
            st.dataframe(df)
        elif menu2 == 'Teams dataframe':
            team = teams_df['team'].unique()
            st.selectbox('Choose a team', team)
            st.write('Data Dimension: ' + str(teams_df.shape[0]) + ' rows and ' + str(teams_df.shape[1]) + ' columns.')
            st.dataframe(teams_df)
        elif menu2 == 'Confederations dataframe':
            region = conf_df['region'].unique()
            st.selectbox('Choose a region', region)
            st.write('Data Dimension: ' + str(conf_df.shape[0]) + ' rows and ' + str(conf_df.shape[1]) + ' columns.')
            st.dataframe(conf_df)
        else:
            st.error('Error: selection invalid')
    elif menu1 == "Number of matches per year":

        first_viz = st.container()

        with first_viz:

            # first visualization

            st.title('How many matches are there per year ?')

            # data prep
            games_per_year = df['year'].value_counts().sort_index()
            # Get list of decades to define the labels of the x-axis
            decades = df['decade'].unique()
            # Get World Cup years to point out the number of matches in those years
            world_cup_years = df[df['tournament'] == 'FIFA World Cup']['year'].unique()
            world_cup_years_games = games_per_year[games_per_year.index.isin(world_cup_years)]

        def lineplot():

            fig = plt.figure(figsize=(20, 10))
            # Mark the time intervals of the World Wars on the chart
            plt.axvline(x=1939, color='r', linestyle='dashed', linewidth=1, ymax=0.2)
            plt.axvline(x=1945, color='r', linestyle='dashed', linewidth=1, ymax=0.2)
            plt.text(1938, 200, 'World War II', color='red')
            # Highlight years of the World Cup
            plt.plot(world_cup_years_games, 'go', label='World Cup')
            # Annotate emergence of COVID-19
            plt.text(2017, 250, 'COVID-19', color='red');
            ln = sns.lineplot(data=games_per_year, x=games_per_year.index, y=games_per_year.values)
            ln.set_title('How many matches per year?', size=20)
            ln.set_xlabel(xlabel='Year', size=12)
            ln.set_ylabel(ylabel='Matches', size=12)
            ln.set(xticks=decades);
            st.pyplot(fig)

        lineplot()

    elif menu1 == "Cities host":

        second_viz = st.container()

        with second_viz:
            # second visualization

            st.subheader('Which cities hosted most games ?')

            with st.echo():
                m = folium.Map(location=[20, 0], tiles='OpenStreetMap', zoom_start=2)

                for index, value in cities_df.iterrows():
                    if not pd.isnull(value['lat']) and not pd.isnull(value['lng']):
                        folium.Circle(
                            location=[value['lat'], value['lng']],
                            popup=f"{value['city_ascii']}: {value['number_of_games']}",
                            radius=value['number_of_games'] * 1000,
                            color='crimson',
                            fill=True,
                            fill_color='crimson'
                        ).add_to(m)

                # After several attempts, I was unable to plot the map, the line below is the line that plot the map...
                # folium_static(m)

    elif menu1 == "Most games per tournament":

        third_viz = st.container()

        with third_viz:

            # third visualization

            # most games per tournament
            st.subheader('Which tournaments had the most games?')

            tournament_counts = pd.DataFrame(df['tournament'].value_counts().reset_index()).rename(
                columns={'index': 'tournament', 'tournament': 'Count'})
            tournament_counts = tournament_counts[(tournament_counts['tournament'] != 'Friendly') &
                                                  (~ tournament_counts['tournament'].str.endswith('qualification')) &
                                                  (tournament_counts['Count'] >= 100)]
            tournament_counts.reset_index(inplace=True)

            def lineplot():
                fig = plt.figure(figsize=(20, 10))
                ax = sns.barplot(y=tournament_counts['tournament'], x=tournament_counts['Count'], orient='h')
                ax.set_title('Which tournaments had the most games?', size=20)
                ax.set_xlabel(xlabel='Matches', size=12)
                ax.set_ylabel(ylabel='Tournament', size=12)
                st.pyplot(fig)

                # Annotate value labels to each team
                for index, value in tournament_counts.iterrows():
                    plt.annotate(value['Count'], xy=(value['Count'] - 20, index + 0.2), color='white')

            lineplot()

    elif menu1 == 'Goals scored per match':

        fourth_viz = st.container()

        with fourth_viz:
            # fourth visualization

            # goals per match
            st.subheader('How many goals are scored per match?')

            goals_per_year = pd.DataFrame(df.groupby(['year'])['total_goals'].sum().reset_index())
            games_per_year = pd.DataFrame(df['year'].value_counts().sort_index().reset_index())
            games_per_year.rename(columns={'index': 'year', 'year': 'matches'}, inplace=True)

            goals_per_year = goals_per_year.merge(games_per_year, how='inner', on=['year'])
            goals_per_year['goals_per_game'] = goals_per_year['total_goals'] / goals_per_year['matches']

            def lineplot():
                fig = plt.figure(figsize=(20, 10))
                ax = sns.lineplot(data=goals_per_year, x='year', y='goals_per_game')
                ax.set_title('How many goals are scored per match?', size=20)
                ax.set_xlabel(xlabel='Year', size=12)
                ax.set_ylabel(ylabel='Goals Per Game', size=12)
                st.pyplot(fig)

            lineplot()
    else:
        st.error('Error: selection invalid')

    st.sidebar.radio('Are the graphs interesting ?', ['Yes', 'No'])

    footer = st.container()

    with footer:

        st.markdown('##')
        st.slider('What rating would you give to my app ?', 0, 10)
        st.text_input('Do you have any feedback ?')
        st.subheader('About')
        st.markdown("""
        >**Data source:** [kaggle.com](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017)
        >
        >**Source code on GitHub:** https://github.com/AndriiGoz/football_game_stats
        >
        >**Author:** Julien Lefaure
        """)
        st.code('Made with <3')


if __name__ == '__main__':
    main()
