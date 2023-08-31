import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji

extract = URLExtract()
def fetch_stats(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]

    # fetch no of msgs
    num_msgs = df.shape[0]

    # no of words
    words = []
    for msg in df["message"]:
        words.extend(msg.split())

    # fetch no. of media using media omitted
    num_media = df[df["message"] == '<Media omitted>\n'].shape[0]

    # fetch no. of links
    links = []
    for msg in df['message']:
        links.extend(extract.find_urls(msg))

    return num_msgs, len(words), len(links), num_media

def most_active_user(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    df = df.rename(columns={'index': 'name', 'user': 'percent'})
    return x, df

def create_wordcloud(user_selected, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if user_selected != 'Overall':
        df = df[df['user'] == user_selected]

    # Remove entries of no significance
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    # Remove stop words according to text file "stop_hinglish.txt"
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    # Dimensions of wordcloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

    # Actual removing
    temp['message'] = temp['message'].apply(remove_stop_words)

    # Word cloud generated
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_used_words(user_selected, df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]
    temp = df[df['user'] != 'group_notification']      # removing these
    temp = temp[temp['message'] != '<Media omitted>\n']       # removing links

    words = []
    # removing stop words
    for msg in temp['message']:
        for word in msg.lower().split():
            if word not in stop_words:
                words.append(word)

    # choosing the most 20 used words
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]

    # Collecting emojis
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

# How many chats per month
def monthly_timeline(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]

    return df['day_name'].value_counts()

def month_activity_map(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]

    return df['month'].value_counts()

def activity_heatmap(user_selected, df):

    if user_selected != "Overall":
        df = df[df['user'] == user_selected]
    user_heatmap = df.pivot_table(index='day_name', columns='period',
                                  aggfunc='count', fill_value=0)['message']
    return user_heatmap


# -1 => Negative
# 0 => Neutral
# 1 => Positive

# Will return count of messages of selected user per day having k(0/1/-1) sentiment
def week_activity_map_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['day_name'].value_counts()


# Will return count of messages of selected user per month having k(0/1/-1) sentiment
def month_activity_map_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['month'].value_counts()


# Will return hear map containing count of messages having k(0/1/-1) sentiment
def activity_heatmap_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]

    # Creating heat map
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


# Will return count of messages of selected user per date having k(0/1/-1) sentiment
def daily_timeline_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    # count of message on a specific date
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


# Will return count of messages of selected user per {year + month number + month} having k(0/1/-1) sentiment
def monthly_timeline_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == -k]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline


# Will return percentage of message contributed having k(0/1/-1) sentiment
def percentage_sentiment(df, k):
    df = round((df['user'][df['value'] == k].value_counts() / df[df['value'] == k].shape[0]) * 100,
               2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df


# Return wordcloud from words in message
def create_wordcloud_sentiment(selected_user, df, k):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove entries of no significance
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    # Remove stop words according to text file "stop_hinglish.txt"
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    # Dimensions of wordcloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

    # Actual removing
    temp['message'] = temp['message'].apply(remove_stop_words)
    temp['message'] = temp['message'][temp['value'] == k]

    # Word cloud generated
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


# Return set of most common words having k(0/1/-1) sentiment
def most_common_words_sentiment(selected_user, df, k):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Creating data frame of most common 20 entries
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df
