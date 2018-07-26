# download  flu prediction data sources
# https://github.com/cmu-delphi/delphi-epidata
# https://www.hhs.gov/about/agencies/regional-offices/index.html
# https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf

import os
import subprocess
import tarfile
import numpy as np
import pandas as pd
import feather
from pandas.io.json import json_normalize
from utils.delphi_epidata import Epidata

# national and hhs_regions https://github.com/hrbrmstr/cdcfluview
regions = ['nat', 'hhs1', 'hhs2', 'hhs3', 'hhs4', 'hhs5', 'hhs6', 'hhs7', 'hhs8', 'hhs9', 'hhs10']


# 51 states + Puerto rico and virgin islands, 53 total states
states = ['AK', 'AL',' AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
          'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
          'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
          'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI',
          'VT', 'WA', 'WI', 'WV', 'WY']


articles = ['amantadine', 'antiviral_drugs', 'avian_influenza', 'canine_influenza', 'cat_flu',
            'chills', 'common_cold', 'cough', 'equine_influenza', 'fatigue_(medical)', 'fever',
            'flu_season', 'gastroenteritis', 'headache', 'hemagglutinin_(influenza)', 'human_flu',
            'influenza', 'influenzalike_illness', 'influenzavirus_a', 'influenzavirus_c',
            'influenza_a_virus', 'influenza_a_virus_subtype_h10n7', 'influenza_a_virus_subtype_h1n1',
            'influenza_a_virus_subtype_h1n2', 'influenza_a_virus_subtype_h2n2', 'influenza_a_virus_subtype_h3n2',
            'influenza_a_virus_subtype_h3n8', 'influenza_a_virus_subtype_h5n1', 'influenza_a_virus_subtype_h7n2',
            'influenza_a_virus_subtype_h7n3', 'influenza_a_virus_subtype_h7n7', 'influenza_a_virus_subtype_h7n9',
            'influenza_a_virus_subtype_h9n2', 'influenza_b_virus', 'influenza_pandemic', 'influenza_prevention',
            'influenza_vaccine', 'malaise', 'myalgia', 'nasal_congestion', 'nausea', 'neuraminidase_inhibitor',
            'orthomyxoviridae', 'oseltamivir', 'paracetamol', 'rhinorrhea', 'rimantadine', 'shivering',
            'sore_throat', 'swine_influenza', 'viral_neuraminidase', 'viral_pneumonia', 'vomiting', 'zanamivir']

def download_epidata(source, locations, epiweek_bounds, filename):
    """
    source: fluview, nowcast, gft, nowcast, wiki
    locations: list of states, regions or articles
    epiweek_bounds e.g tuple of start and end of epiweek e.g (199740, 201810)
    filename: full path to save dataframe
    """

    df = []
    for location in locations:
        try:
            res = eval('Epidata.{}({}, epiweeks=[Epidata.range({}, {})])'.format(
                source, [str(location)], epiweek_bounds[0], epiweek_bounds[1]))
            df.append(json_normalize(res['epidata']))
        except Exception as e: print(e)

    df = pd.concat(df)
    df.to_csv(filename, index=False)

    return df


def download_epidata_all(epiweek_start, epiweek_end, data_path):


    # FluView: Influenza-like illness (ILI) from U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet).
    # Teemporal Resolution: Weekly* from 1997w40
    print('processing fluview ...')
    filename = data_path + 'fluview_regions.csv'
    if not os.path.exists(filename):
        fluview_regions = download_epidata('fluview', regions, (epiweek_start, epiweek_end), filename)

    filename = data_path + 'fluview_states.csv'
    if not os.path.exists(filename):
        fluview_states = download_epidata('fluview', states, (epiweek_start, epiweek_end), filename)


    # ILI-Nearby: A nowcast of U.S. national, regional, and state-level (weighted) %ILI, available seven days
    # (regionally) or five days (state-level) before the first ILINet report for the corresponding week.
    # Temporal Resolution: Weekly, from 2010w30*
    print('processing nowcast ...')
    filename = data_path + 'nowcast_regions.csv'
    if not os.path.exists(filename):
        nowcast_regions = download_epidata('nowcast', regions, (epiweek_start, epiweek_end), filename)

    filename = data_path + 'nowcast_states.csv'
    if not os.path.exists(filename):
        nowcast_states = download_epidata('nowcast', states, (epiweek_start, epiweek_end), filename)

    """
    # Google Flu Trends: Estimate of influenza activity based on volume of certain search queries.
    # Google has discontinued Flu Trends, and this is now a static data source.
    # Temporal Resolution: Weekly from 2003w40 until 2015w32
    print('processing gft ...')
    filename = data_path + 'gft_regions.csv'
    if not os.path.exists(filename):
        gft_regions = download_epidata('gft', regions, (epiweek_start, epiweek_end), filename)

    filename = data_path + 'gft_states.csv'
    if not os.path.exists(filename):
        gft_states = download_epidata('gft', states, (epiweek_start, epiweek_end), filename)
    """

    # Wikipedia Access Logs
    # Temporal Resolution: Hourly, daily, and weekly from 2007-12-09 (2007w50)
    print('processing wiki ...')
    filename = data_path + 'wiki_articles.csv'
    if not os.path.exists(filename):
        wiki_articles = download_epidata('wiki', articles, (epiweek_start, epiweek_end), filename)


def download_weather_data(epiweek_start=197740, epiweek_end=201810):
    # Average Global Historical Climatology Network (GHCN) of US station for
    # each epiweek since 199740
    print('processing ghcn ...')
    ghcnd_path = data_path + 'ghcnd'
    if not os.path.exists(ghcnd_path):
        os.makedirs(ghcnd_path)
        files = ['readme.txt', 'ghcnd-stations.txt', 'ghcnd-states.txt', 'ghcnd-countries',
                 'ghcnd-inventory.txt', 'ghcnd_all.tar.gz']
        for f in files:
            cmd = 'wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/{} -P {}'.format(f, ghcnd_path)
            subprocess.call(cmd, shell=True)

    if not os.path.exists('{}/{}'.format(ghcnd_path, 'ghcnd_all')):
        tar = tarfile.open(ghcnd_path+'/'+'ghcnd_all.tar.gz')
        tar.extractall(path=ghcnd_path)
        tar.close()

    filename = 'ghcn_epiweek.csv'

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    """ This is a function for joining tables on specific fields. By default, we'll be doing a
    left outer join of right on the left argument using the given fields for each table.
    """
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on,
                      suffixes=("", suffix))


def proc_data(epiweek_start, epiweek_end, data_path, with_forecast=True):
        # create path for saving data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print('########## Downloading Epidata #############')
    download_epidata_all(epiweek_start, epiweek_end, data_path)

    print('########## Loading Data Into Pandas Dataframes  #############')
    table_names = ['fluview_regions', 'fluview_states', 'nowcast_regions',
                   'nowcast_states', 'wiki_articles']

    tables = [pd.read_csv(data_path+tname+'.csv', low_memory=False)
              for tname in table_names]
    fluview_regions, fluview_states, nowcast_regions, nowcast_states, wiki_articles = tables


    print('########## Adding Forecast Columns to Fluview Data #############')
    fluview = pd.concat([fluview_regions, fluview_states], ignore_index=True)

    # split epiweek number and year
    fluview['epiweeknum'] = fluview['epiweek'].apply(lambda x: int(str(x)[4:]))
    fluview['epiyear'] = fluview['epiweek'].apply(lambda x: int(str(x)[:4]))

    # add column of bin_start_incl values of wili percentages
    bin_start_incl = np.round(np.r_[np.arange(0, 13.1, .1), 100], decimals=1)
    inds = pd.np.digitize(fluview['wili'], bin_start_incl)
    fluview['wili_bin'] = fluview.index.map(lambda x: bin_start_incl[inds[x]-1])

    if with_forecast:
        # add weekly forcast columns and prior weeks columns
        nweeks = 4
        fluview_grp = fluview.groupby(['region'])
        fluview = []

        for name, grp in fluview_grp:
            grp = grp.sort_values('epiweek', ascending=True)
            for nw in range(1, nweeks+1):
                grp['week_ahead_wili_{}'.format(nw)] = grp['wili_bin'].shift(-nw)
                fluview.append(grp)
        fluview = pd.concat(fluview, ignore_index=True).drop_duplicates()

        # add season forecast column
        fluview_grp = fluview.groupby(['region', 'epiyear'])
        fluview = []
        for name, grp in fluview_grp:
            grp = grp.sort_values('wili', ascending=False).reset_index(drop=True)
            grp['peak_week'] = str(grp['epiweek'].iloc[0])[4:]
            grp['peak_intensity'] = grp['wili_bin'].iloc[0]
            fluview.append(grp)
        fluview = pd.concat(fluview).drop_duplicates().reset_index(drop=True)


    print('########## Combining Data #############')
    # join now cast data
    nowcast = pd.concat([nowcast_regions, nowcast_states], ignore_index=True)
    nowcast.rename(columns={'location':'region'}, inplace=True)
    nowcast['region'] = nowcast['region'].str.lower()
    nowcast.columns = [str(col) + '_nc' for col in nowcast.columns]
    fluview_nowcast = join_df(fluview, nowcast, ['epiweek', 'region'], ['epiweek_nc', 'region_nc'])

    # remove analysis irrelevant columns
    fluview_nowcast = fluview_nowcast.drop(['epiweek_nc', 'region_nc', 'issue', 'release_date'], 1)

    # join wiki artilcles
    wiki_articles_grp = wiki_articles.groupby(['article'])
    wiki_articles_ = []
    for name, grp in wiki_articles_grp:
        grp = grp[['epiweek', 'count', 'total', 'value']]
        grp.columns = ['epiweek', name + '_count_wiki',  name + '_total_wiki',  name + '_value_wiki']
        wiki_articles_.append(grp)
    wiki_articles = wiki_articles_[0]
    for wi in wiki_articles_[1:]:
        wiki_articles = join_df(wiki_articles, wi, 'epiweek')

    joined_df = join_df(fluview_nowcast, wiki_articles, 'epiweek')

    # add region type to each row [state=0, hhs=1, national=2]
    label_region = lambda x: 2 if x == 'nat' else (1 if x[0:3] == 'hhs' else 0)
    joined_df['region_type'] = joined_df['region'].apply(label_region)


    print('########## Filling Missing Data and Adding Feature Engineering Columns #############')
    # fill missing continous value columns with median value from region and epiweeknum groups
    num_ages_cols = ['num_age_0', 'num_age_1', 'num_age_2', 'num_age_3', 'num_age_4', 'num_age_5']
    wiki_cols = [c for c in joined_df.columns.tolist() if 'wiki' in c]
    cont_cols = ['ili', 'lag', 'num_ili', 'num_patients', 'num_providers',
                 'wili', 'std_nc', 'value_nc']
    cont_cols = cont_cols + num_ages_cols + wiki_cols

    for c in cont_cols:
        joined_df[c] = joined_df[c].fillna(
            joined_df.groupby(['region', 'epiweeknum'])[c].transform('median'))

    # std_nc and value_nc still have nans so we fill based on region type and epiweenum
    for c in cont_cols:
        joined_df[c] = joined_df[c].fillna(
            joined_df.groupby(['region_type', 'epiweeknum'])[c].transform('median'))

    # age_num is not available at state level so we set value to 0 for states
    #joined_df.loc[joined_df['region_type'] == 0, num_ages_cols] = 0
    joined_df[num_ages_cols] = joined_df[num_ages_cols].fillna(0)

    # drop prediction rows with null values
    joined_df = joined_df.dropna(how='any')

    # remove epiweek and wili_bin columns as they are no longer needed for computations
    #joined_df = joined_df.drop(['epiweek', 'wili_bin'], 1)


    print('########## Saving Processed Dataframe  #############')
    #joined_df.to_csv(data_path + 'joined_df.csv', index=False)
    feather.write_dataframe(joined_df, data_path + 'joined_df.feather')

    return joined_df


if __name__ == '__main__':

    epiweek_start = 199740
    epiweek_end = 201810
    data_path = os.environ['DATA_DIR'] + 'epidata_flu/'
    with_forecast = True

    joined_df = proc_data(epiweek_start, epiweek_end, data_path, with_forecast)

    """
    # null test and forecast check
    joined_df.isnull().sum()
    year_ = 2015
    region_ =  'oh'
    forecast_check = joined_df[(joined_df['epiyear']==year_) & (joined_df['region']==region_)][[
        'epiweeknum', 'wili', 'week_ahead_wili_1', 'week_ahead_wili_2',
        'week_ahead_wili_3', 'week_ahead_wili_4', 'peak_week', 'peak_intensity']].sort_values('epiweeknum')
    """

    """
    # add feature engineering columns
    """
