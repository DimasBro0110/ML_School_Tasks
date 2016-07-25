__author__ = 'DmitriyBrosalin'

"""
1) First of all I'll create data set to fit model
    - The data must contain only f - features
    - I have to create vectors with all features for each films(if there is not feature, it should be 0)
    - I have to create dictionary where key is id of each film and value is vektor of film's feature
2) Then I'll make vektor of labels: -1 will be something like dislike, 1 will be like
    - I have to create dictionary where key is id of each film and value is a label
3) After that I will calculate Soft Nearest Classification
"""

import json
import pandas as pd
import numpy as np
from scipy.spatial import distance


with open('items.json') as f:
    items = json.load(f)


def make_prediction(user_id, film_id):
    all_features = [u'f_130892', u'f_44251', u'f_144583', u'f_207186', u'year', u'duration', u'f_74564', u'id',
                    u'f_110704',
                    u'f_110705', u'f_168477', u'f_39933', u'f_153601', u'f_46518', u'f_187511', u'f_119091',
                    u'f_158463',
                    u'f_49968', u'f_110698', u'f_163719', u'f_191091', u'f_205162', u'f_122282', u'genre', u'f_151440',
                    u'f_210900', u'f_148972', u'f_148957', u'f_122494', u'f_30859']

    lst_feat = []

    for row in items:
        keys = row.keys()
        value = row.values()
        temp = []

        for k in all_features:
            if k in keys:
                temp.append(row[k])
            else:
                temp.append(0)

        lst_feat.append(temp)

    """Creation of dataset """

    data = pd.DataFrame(data=lst_feat, columns=all_features)
    data = data.drop_duplicates(subset=['id'], keep=False)

    dat_likes = pd.read_csv('train_likes.csv')
    dat_likes = dat_likes[['item_id', 'channel', 'user_id']]

    dat_likes = dat_likes.loc[dat_likes['user_id'] == user_id]
    lst_of_films = dat_likes['item_id']
    lst_of_films = list(lst_of_films)

    """ Here i get features of films which are liked by user """

    lst = []

    for i in lst_of_films:
        dat = data.loc[data['id'] == i]
        dat = np.array(dat.iloc[:, 0:30].values)
        if len(dat) > 0:
            lst.append(dat.ravel())
        else:
            print("Can't predict, user has not liked any films !!!")
            return 0

    if len(lst) > 0:
        feat = pd.DataFrame(data=lst, columns=all_features)
        feat.drop('id', 1, inplace=True)
        feat.fillna(0, inplace=True)
    else:
        print("Can't predict, user has not liked any films !!!")
        return 0

    """ Here i get features of entered film """

    entered_film = data.loc[data['id'] == film_id]
    entered_film = np.array(entered_film.iloc[:, 0:30].values)
    entered_film = entered_film.ravel()
    if entered_film.shape[0] != 0:
        entered_film = np.delete(entered_film, 7)
    else:
        print("Can't predict, user has not liked any films !!!")
        return 0

    """ Here I have to make Calculations """

    # I wil use Soft Neighbour Classification
    # the equation is : p(i,j) = exp(dist(i,j)) / sum(dist(i,k))


    m_array = feat.values
    euclid_distances_selected_film = []

    for i in range(0, m_array.shape[0]):
        dst = distance.euclidean(m_array[i], entered_film)
        euclid_distances_selected_film.append(dst)

    exp_dist = [np.exp(i) for i in euclid_distances_selected_film]
    summ = 0
    for i in exp_dist:
        summ += i

    probabilities = []
    for i in range(0, m_array.shape[0]):
        probabilities.append(euclid_distances_selected_film[i] / summ)

    answer = np.mean(probabilities)

    return answer

# Example

l = ['4dcc64cc0d1daf524d0eb3e1c1f8a05a',
     '5548ea89b36f00ae3a59b81a3c05db7e',
     '32de887562ea3c90b93de2d9d0b56c43',
     '928f815de0f1b2e9da38e9a1447224d6',
     'e4c0f0d6067602b97c434aee8352c745',
     '7fcbb0c2a1bb45f2400b3247d12dd536',
     '928f815de0f1b2e9da38e9a1447224d6',
     'b257097004e1490476f6eaf181d4389c',
     'a61f5adf20280b8e8590ac442770b9e3',
     '3b091fd54b4caebd395ba064cb0a3d1e',
     'c1e2d06fadd1fddce00ee2f22410b787',
     'd720202a16de6c3bc337095588405d77',
     '16e2c85f96097ee3e622bda8489ca194',
     'a05af990b83efe92a765931005be73c7',
     '80278ad7c0f0dad889589efbe2fe154a']

for i in l:
    ans = make_prediction(i, '7f73d20ea0d110a9155777c1a0cf9bb2')
    print(ans)