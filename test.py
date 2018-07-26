"""

create test function that will

given epiweek, model, mappers, params e.g week prior
1.download data
2 transform data
3.make predictions


for previous past epiweeks predictions can be compared to actual
"""
from prep_data import *
from flu_predict import *
from keras.models import load_model

metrics.top1 = top1

def proc_test(epiweek_start, epiweek_end, weeks_prior):
    # download epiweek data and convert to dataframe
    data_path = 'data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    test_x = proc_data(epiweek_start, epiweek_end, data_path, with_forecast=False)

    # load train data transformations and variables
    ml_data, _ , _ = get_ml_data()
    train_xp = ml_data['train_xp']
    mapper = ml_data['mapper']
    le_wili = ml_data['le_wili']
    le_epi = ml_data['le_epi']
    cat_vars = ml_data['cat_vars']
    contin_vars = ml_data['contin_vars']
    wiki_cols = ml_data['wiki_cols']

    # fill missing feature columns with zeros
    u = set([x for x in train_xp if 'wiki' in x])
    v = set([x for x in test_x if 'wiki' in x])
    uv_diff = u.difference(v)
    for uvd in uv_diff:
        test_x[uvd] = 0
    test_x = test_x.fillna(0)

    # numericalize and standardize data
    test_x = test_x[cat_vars + contin_vars]
    apply_cats(test_x, train_xp)
    region_mapper = dict( enumerate(test_x['region'].cat.categories) )

    for v in contin_vars:
        test_x[v] = test_x[v].astype('float32')

    test_x, _ = proc_df(test_x, mapper=mapper)

    # concat priors for each region
    df_grp = test_x.groupby(['region'])
    test_x = []
    test_regions = []
    import ipdb; ipdb.set_trace()
    for name, grp in df_grp:
        grp = grp.sort_values('epiweeknum', ascending=False).reset_index(drop=True)
        grp = grp.values.flatten()
        test_x.append(grp)
        test_regions.append(region_mapper[name])
    test_x = np.array(test_x)

    # load model and predict forecast values
    model = load_model('model.h5')

    test_y = model.predict(test_x)
    test_y = [np.argmax(ty, axis=1) for ty in test_y]
    for idx, ty in enumerate(test_y[:-1]):
        test_y[idx] = le_wili.inverse_transform(ty)
    test_y[-1] = le_epi.inverse_transform(test_y[-1])
    return test_y, test_regions


if __name__ == '__main__':
    weeks_prior = 4
    epiweek_start = 201806
    epiweek_end = 201810
    test_y , test_regions= proc_test(epiweek_start, epiweek_end, weeks_prior)
    y_wili_1, y_wili_2, y_wili_3, y_wili_4, y_peak_wili, y_peak_week = test_y
