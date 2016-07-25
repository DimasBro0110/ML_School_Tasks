__author__ = 'DmitriyBrosalin'

import pandas as pd

data_train_likes = pd.read_csv('train_likes.csv')
data_train_likes = data_train_likes.dropna()

count_likes = data_train_likes['channel'].value_counts()
mean_likes_channels = count_likes.mean()
answer_1 = mean_likes_channels  # the first answer of the first task

films_likes = data_train_likes['item_id'].value_counts()
lst = [i for i in films_likes if i >= 5]
answer_2 = len(lst)  # the second answer of the first task

print("The mean amount of likes: " + str(answer_1))
print("Amount of films with more than 5 likes: " + str(answer_2))
