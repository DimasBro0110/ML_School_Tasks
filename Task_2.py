__author__ = 'DmitriyBrosalin'

import pandas as pd
import json

with open('items.json') as f:
    items = json.load(f)

dat_2 = pd.read_csv('train_likes.csv')
dat_2 = dat_2[['item_id', 'channel']]

""" Here I will calculate amount of likes for every genres including NaN """

lst_of_id = []
lst_of_duration = []
lst_of_year = []
lst_of_genre = []
lst_of_none_genre = []

for i in items:
    lst_of_id.append(i['id'])
    lst_of_duration.append(i['duration'])
    lst_of_year.append(i['year'])
    lst_of_genre.append(i['genre'])

dat = {
    'item_id': lst_of_id,
    'duration': lst_of_duration,
    'year': lst_of_year,
    'genre': lst_of_genre
}

data_frame = pd.DataFrame(data=dat)
whole_data = pd.merge(data_frame, dat_2, on='item_id', how='outer')
whole_data['genre'].fillna(10, inplace=True)
answer_1 = whole_data['genre'].value_counts().sort_index()

# print(answer_1)

""" Here I will calculate amount of Top-10 liked channels """

dat = dat_2['channel'].value_counts()
dat = dat.head(10)
name_of_top_channels = dat.index.tolist()

# print(name_of_top_channels)

lst_of_genre_like = []
dict_genre_like = {}
dict_genre_like_2 = {}

for channel in name_of_top_channels:
    data = whole_data[whole_data['channel'] == channel]
    answer = data['genre'].value_counts(dropna=False).sort_index()
    indx = answer.index.values
    indexes = xrange(0, 11, 1)
    ans = []
    for i in indexes:
        if i in indx:
            ans.append(answer[i])
        else:
            ans.append(0)

    dict_genre_like_2[channel] = answer
    dict_genre_like[channel] = ans


""" According to the result of consistences of items in the list, some of channels don't have all genres in it """

""" Report """

print("The result of analysis for the first part of Task 2.\nThe 10 label means films where genre is unknown")
print(answer_1)

print("The result of analysis for the second part of Task 2.")
for key, value in dict_genre_like.items():
    cnt = 0
    print("\n\nThe channel is: " + key)
    for i in value:
        print("genre " + str(cnt) + ": " + str(i))
        cnt += 1
