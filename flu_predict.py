# flu prediction
import os
import pandas as pd
import feather
from utils.fastai.structured import *
from utils.fastai.column_data import *
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics

pd.set_option('display.width', 250)


data_path = os.environ['DATA_DIR'] + 'epidata_flu/'

def drop_columns(df, cols):
    """drop columns form dataframe"""
    df = df.drop(cols, axis=1)

    return df


def show_prediction(model, raw_df, epiyear):
    """
    compare prediction from actual values given epiyear
    """



def proc_df(df, max_n_cat=None, mapper=None):

    """ standardizes continuous columns and numericalizes categorical columns
    Parameters:
    -----------
    df: The data frame you wish to process.

    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.

    mapper: calculates the values used for scaling of variables during training time(mean
            and standard deviation).

    Returns:
    --------
    x: x is the transformed version of df. x will not have the response variable
        and is entirely numeric.

    mapper: A DataFrameMapper which stores the mean and standard deviation of the
            corresponding continous variables which is then used for scaling of during test-time.
    """

    df = df.copy()

    mapper = scale_vars(df, mapper)
    for n, c in df.items():
        numericalize(df, c, n, max_n_cat)

    return pd.get_dummies(df, dummy_na=True), mapper


def concat_prior(df, cols, shift_num=4):
    """
    shift dataframe forward to compute prior epiweek features
    returns a dataframe with concatenated prior features
    cols is a list of columns to shift
    shift_num is how many prior weeks to shift
    """

    # add concatenated features

    df_grp = df.groupby(['region'])
    df = []
    for name, grp in df_grp:
        grp = grp.sort_values(['epiyear', 'epiweeknum'], ascending=True).reset_index(drop=True)

        grp_ = [grp]
        for idx in range(1, shift_num+1):
            grp_.append(grp[cols].shift(idx))
            grp_[idx].columns = [c + '_prior_{}'.format(idx) for c, idx  in zip(cols, len(cols)*[idx])]
        grp = pd.concat(grp_, axis=1).loc[shift_num:]

        df.append(grp)

    return  pd.concat(df).fillna(0)


def get_ml_data(recompute=False):
    """
    preprocess data for ml training
    num_shift is number of prior epiweeks to cancatenate for feature columns
    """

    ml_data_path = data_path + 'ml_data/'
    feather_names = ['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y', 'train_xp']
    mapper_names = ['mapper', 'le_wili', 'le_epi']
    var_names = ['cat_vars', 'contin_vars', 'wiki_cols', 'pred_wili', 'pred_epi', 'pred_vars']

    if not os.path.exists(ml_data_path):
        os.makedirs(ml_data_path)


    # read flu dataframe and shuffle
    df = pd.read_feather(data_path + 'joined_df.feather')
    #df = df.reindex(np.random.permutation(df.index))

    # set categorial and continous data (prediction variables are also categorial)
    wiki_cols = [c for c in df.columns.tolist() if 'wiki' in c]
    cat_vars = ['region', 'epiweeknum', 'region_type']


    contin_vars = ['epiyear', 'lag', 'ili', 'num_age_0', 'num_age_1', 'num_age_2', 'num_age_3',
                   'num_age_4', 'num_age_5', 'num_ili', 'num_patients', 'num_providers', 'wili',
                   'std_nc', 'value_nc'] + wiki_cols

    pred_wili = ['week_ahead_wili_1', 'week_ahead_wili_2', 'week_ahead_wili_3',
                 'week_ahead_wili_4', 'peak_intensity']
    pred_epi = ['peak_week']
    pred_vars = pred_wili + pred_epi

    # use data from 1997 to 2016 for train and 2017 for validation
    # use 2018 for weekly forecast testing, we can't tes season onset
    # and peak because data is not yet available for 2018
    val_year = 2017
    test_year = 2018

    #df_ = df.copy()
    #n = 50000 # len(df) # training data sample
    #idxs = get_cv_idxs(n, val_pct=150000/n)
    #df = df.iloc[idxs].reset_index(drop=True)

    train_df = df[(df['epiyear'] != val_year) &
                  (df['epiyear'] != test_year)].reset_index(drop=True)
    val_df = df[df['epiyear'] == val_year].reset_index(drop=True)
    test_df = df[df['epiyear'] == test_year].reset_index(drop=True)

    del df

    if not os.listdir(ml_data_path) or recompute:

        # split data into features and prediction variables
        print ('Splitting data into features and prediction variables ...')
        train_x = train_df[cat_vars + contin_vars].copy()
        train_y = train_df[pred_vars].copy()
        val_x = val_df[cat_vars + contin_vars].copy()
        val_y = val_df[pred_vars].copy()
        test_x = test_df[cat_vars + contin_vars].copy()
        test_y = test_df[pred_vars].copy()


        # numericalize and standardize training features / fit prediction values
        # train_xp contains pre transformation values
        print ('Numericalizing and Standardizing training data ...')
        train_xp = train_x.copy()
        for v in cat_vars:
            train_xp[v] = train_x[v].astype('category').cat.as_ordered()
        for v in contin_vars:
            train_xp[v] = train_xp[v].astype('float32')
            train_x, mapper = proc_df(train_xp)


        print ('Applying label transformation ...')
        bin_start_incl = np.round(np.r_[np.arange(0, 13.1, .1), 100], decimals=1)
        le_wili = preprocessing.LabelEncoder() # wili forecast percentages
        le_wili.fit(bin_start_incl)
        le_epi = preprocessing.LabelEncoder() # wili peak weak
        le_epi.fit(range(1,54))

        for v in pred_wili:
            train_y[v] = train_y[v].transform(le_wili.transform)

        for v in pred_epi:
            train_y[v] = train_y[v].transform(pd.to_numeric)
            train_y[v] = train_y[v].transform(le_epi.transform)


        # apply training transformation to validation and test data
        print ('Numericalizing and Standardizing validatan and test data ...')
        apply_cats(val_x, train_xp)
        for v in contin_vars:
            val_x[v] = val_x[v].astype('float32')
        val_x, _ = proc_df(val_x, mapper=mapper)


        for v in pred_wili:
            val_y[v] = val_y[v].transform(le_wili.transform)
        for v in pred_epi:
            val_y[v] = val_y[v].transform(pd.to_numeric)
            val_y[v] = val_y[v].transform(le_epi.transform)


        if not test_df.empty:
            apply_cats(test_x, train_xp)
            for v in contin_vars:
                test_x[v] = test_x[v].astype('float32')
            test_x, _ = proc_df(test_x, mapper=mapper)

            for v in pred_wili:
                test_y[v] = test_y[v].transform(le_wili.transform)

        # generate dictionary from processed data
        ml_data = {}
        for fname in feather_names + mapper_names:
            ml_data[fname] = eval(fname)

        # save processed data to disk
        for fname in feather_names:
            feather.write_dataframe(eval(fname), ml_data_path + fname + '.feather')
        for fname in mapper_names:
            pd.to_pickle(eval(fname), ml_data_path + fname + '.pkl')
        for fname in var_names:
            pd.to_pickle(eval(fname), ml_data_path + fname + '.pkl')

    else:
        # read data from desk
        ml_data = {}
        for fname in feather_names:
            ml_data[fname] = pd.read_feather(ml_data_path + fname + '.feather')
        for fname in mapper_names:
            ml_data[fname] = pd.read_pickle(ml_data_path + fname + '.pkl')
        for fname in var_names:
            ml_data[fname] = pd.read_pickle(ml_data_path + fname + '.pkl')


    return ml_data, cat_vars + contin_vars, pred_vars


# Embedding learning scripts
def cat_map_info(feat):
    return feat[0], len(feat[1].classes_)

def my_init(scale):
    return lambda shape, name=None: initializations.uniform(shape, scale=scale, name=name)

def emb_init(shape, name=None):
    return initializations.uniform(shape, scale=2/(shape[1]+1), name=name)

def get_emb(feat):
    name, c = cat_map_info(feat)
    #c2 = cat_var_dict[name]
    c2 = (c+1)//2
    if c2>50: c2=50
    inp = Input((1,), dtype='int64', name=name+'_in')
    # , W_regularizer=l2(1e-6)
    u = Flatten(name=name+'_flt')(Embedding0(c, c2, input_length=1, init=emb_init)(inp))
#     u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1)(inp))
    return inp,u

def get_contin(feat):
    name = feat[0][0]
    inp = Input((1,), name=name+'_in')
    return inp, Dense(1, name=name+'_d', init=my_init(1.))(inp)

def top1(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)

def get_model(dim, num_classes_wili, num_classes_epi):

    input_ = Input(shape=(dim, ), dtype='float32')
    x = Dropout(0.02)(input_)
    x = Dense(2000, activation='relu', init='uniform')(x)
    x = Dense(1000, activation='relu', init='uniform')(x)
    x = Dense(500, activation='relu', init='uniform')(x)
    x = Dropout(0.2)(x)

    y_wili_1 = Dense(num_classes_wili, activation='softmax', name='y_wili_1')(x) # 1 week ahead  prediction
    y_wili_2 = Dense(num_classes_wili, activation='softmax', name='y_wili_2')(x) # 2 week ahead  prediction
    y_wili_3 = Dense(num_classes_wili, activation='softmax', name='y_wili_3')(x) # 3 week ahead  prediction
    y_wili_4 = Dense(num_classes_wili, activation='softmax', name='y_wili_4')(x) # 4 week ahead  prediction
    y_peak_wili = Dense(num_classes_wili, activation='softmax', name='y_peak_wili')(x) # peak wili prediction
    y_peak_week = Dense(num_classes_epi, activation='softmax', name='y_peak_week')(x) # peak wili predictio

    model = Model(inputs=[input_], outputs=[y_wili_1, y_wili_2, y_wili_3, y_wili_4, y_peak_wili, y_peak_week])

    return model

if __name__ == '__main__':


    # generate ml data2
    print ("Generating ML Data ...")
    ml_data, x_vars, y_vars = get_ml_data()


    train_xp = ml_data['train_xp']
    mapper = ml_data['mapper']
    le_wili = ml_data['le_wili']
    le_epi = ml_data['le_epi']

    train_x = ml_data['train_x']
    val_x = ml_data['val_x']
    test_x = ml_data['test_x']

    train_y = ml_data['train_y']
    val_y = ml_data['val_y']
    test_y = ml_data['test_y']

    # add time series variables
    prior = 4
    train_x = pd.concat([train_x, train_y], axis=1)
    train_x = concat_prior(train_x, x_vars, shift_num=prior)


    train_x = train_x.sample(frac=1).reset_index(drop=True) # shuffle data
    train_y = train_x[y_vars]
    train_x = train_x.drop(y_vars, axis=1)
    val_x = pd.concat([val_x, val_y], axis=1)
    val_x = concat_prior(val_x, x_vars, shift_num=prior)
    val_y = val_x[y_vars]
    val_x = val_x.drop(y_vars, axis=1)

    #create multilabel classification model for wili predictions
    print ("Generating NN Model ...")
    num_epoch = 50
    dim = len(train_x.columns) # imput feature dimension
    num_classes_wili = len(le_wili.classes_)
    num_classes_epi = 53
    model = get_model(dim, num_classes_wili, num_classes_epi)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=[top1])

    train_labels = [keras.utils.to_categorical(l, num_classes_wili) for l in train_y.values.T[:-1]]
    train_labels.append(keras.utils.to_categorical(train_y.values.T[-1], num_classes_epi))

    val_labels = [keras.utils.to_categorical(l, num_classes_wili) for l in val_y.values.T[:-1]]
    val_labels.append(keras.utils.to_categorical(val_y.values.T[-1], num_classes_epi))

    monitor = 'y_wili_4_loss'
    checkpointer = ModelCheckpoint(filepath='model_weights.hdf5', monitor=monitor, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor=monitor, min_delta=0, patience=0, verbose=0, mode='auto')

    model.fit(train_x.values, train_labels, epochs=num_epoch, validation_data=(val_x.values, val_labels),
              batch_size=32, callbacks=[checkpointer])

    model.save('model.h5')
    score = model.evaluate(val_x.values, val_labels, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    pred_str= 'week_ahead_wili_1'
    val_y_= np.argmax(model.predict(val_x)[1], axis=1)
    print(classification_report(val_y[pred_str], val_y_, le_wili.classes_))
    print(confusion_matrix(val_y[pred_str], val_y_, le_wili.classes_))


    # generate prediction for specific epiweek


    """
    sklearn/random forest ffiting
    print ("Fitting Data ...")
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.metrics import classification_report, confusion_matrix


    import sys
    sys.path.append('/home/bernard/apps/xgboost/python-package')
    from xgboost import XGBClassifier
    params = { "n_estimators": 400, 'tree_method':'gpu_hist', 'predictor':'gpu_predictor' }

    pred_str= 'week_ahead_wili_1'
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    #clf = XGBClassifier(**params)
    clf = clf.fit(train_x, train_y[pred_str])
    print ("Predicting Data ...")
    val_y_= clf.predict(val_x)
    print(classification_report(val_y[pred_str], val_y_, le_wili.classes_))
    print(confusion_matrix(val_y[pred_str], val_y_, le_wili.classes_))
    #np.savetxt('x.tsv', x.astype(int), fmt='%10.0f', delimiter=',')
    """


    """
    import time
    import matplotlib.pyplot as plt

    uu = [x for x in train_x.columns if 'wili_prior' in x]
    uu = ['wili'] + uu
    arr = np.fliplr(train_x[uu].values)

    fig = plt.figure()
    plt.ion()
    plt.show()

    for aa in arr:
        plt.plot(aa)
        plt.ylim((-1, 2.5))
        plt.draw()
        plt.pause(1)
        plt.clf()

    """
