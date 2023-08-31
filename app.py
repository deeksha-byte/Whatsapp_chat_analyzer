# Importing modules
import nltk
import streamlit as st
import re
import preprocess,helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.sidebar.title("Whatsapp Chat and Sentiment Analysis")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:

    # Main heading
    st.markdown("<h1 style='text-align: center; color: black;'>Whatsapp Chat Analyzer</h1>",
                unsafe_allow_html=True)

    # file received is a stream of byte data which needs to be converted into string
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Perform preprocessing
    df = preprocess.preprocessor(data)

    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different columns for (Positive/Negative/Neutral)
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]  # Neutral


    # To indentify true sentiment per row in message column
    def sentiment(data):
        if data["po"] >= data["ne"] and data["po"] >= data["nu"]:
            return 1
        if data["ne"] >= data["po"] and data["ne"] >= data["nu"]:
            return -1
        if data["nu"] >= data["po"] and data["nu"] >= data["ne"]:
            return 0

    # Creating new column & Applying function
    df['value'] = df.apply(lambda row: sentiment(row), axis=1)

    # fetch unique users
    user_list = df['user'].unique().tolist()

    # removing group notification from user_list, sort it and add value overall to it
    user_list.sort()
    user_list.insert(0, 'Overall')        # 0 is index of overall -- means group level analysis

    user_selected = st.sidebar.selectbox("Show Analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):        # show analysis button is clicked

        # create 4 columns
        num_msgs, words, num_links, num_media = helper.fetch_stats(user_selected, df)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_msgs)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media)

        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly analysis of chats on a particular year
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(user_selected, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(user_selected, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Most active users based on day and month -- activity map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(user_selected, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(user_selected, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Activity Heatmap
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(user_selected, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(only for grp level)
        if user_selected == 'Overall':
            st.title("Most Active Users")
            x, new_df = helper.most_active_user(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color = 'red')
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # wordCloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(user_selected, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most common meaning full words
        most_common_df = helper.most_common_used_words(user_selected, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common words")
        st.pyplot(fig)

        # Emoji analysis top 10
        emoji_df = helper.emoji_helper(user_selected, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(10), labels=emoji_df[0].head(10), autopct="%0.2f")
            st.pyplot(fig)

        # Main heading
        st.markdown("<h1 style='text-align: center; color: black;'>Whatsapp Sentiment Analyzer</h1>",
                    unsafe_allow_html=True)

        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map_sentiment(user_selected, df, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map_sentiment(user_selected, df, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map_sentiment(user_selected, df, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map_sentiment(user_selected, df, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map_sentiment(user_selected, df, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map_sentiment(user_selected, df, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",
                        unsafe_allow_html=True)

            user_heatmap = helper.activity_heatmap_sentiment(user_selected, df, 1)

            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",
                        unsafe_allow_html=True)

            user_heatmap = helper.activity_heatmap_sentiment(user_selected, df, 0)

            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",
                        unsafe_allow_html=True)

            user_heatmap = helper.activity_heatmap_sentiment(user_selected, df, -1)

            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline_sentiment(user_selected, df, 1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline_sentiment(user_selected, df, 0)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline_sentiment(user_selected, df, -1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline_sentiment(user_selected, df, 1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline_sentiment(user_selected, df, 0)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline_sentiment(user_selected, df, -1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Percentage contributed
        if user_selected == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = helper.percentage_sentiment(df, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = helper.percentage_sentiment(df, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = helper.percentage_sentiment(df, -1)

                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if user_selected == 'Overall':
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # WORDCLOUD......
        col1, col2, col3 = st.columns(3)
        with col1:
            # heading
            st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",
                        unsafe_allow_html=True)

            # Creating wordcloud of positive words
            df_wc = helper.create_wordcloud_sentiment(user_selected, df, 1)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
        with col2:
            # heading
            st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",
                            unsafe_allow_html=True)

            # Creating wordcloud of neutral words
            df_wc = helper.create_wordcloud_sentiment(user_selected, df, 0)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
        with col3:
            # heading
            st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",
                        unsafe_allow_html=True)

            # Creating wordcloud of negative words
            df_wc = helper.create_wordcloud_sentiment(user_selected, df, -1)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            # Data frame of most common positive words.
            most_common_df = helper.most_common_words_sentiment(user_selected, df, 1)

            # heading
            st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            # Data frame of most common neutral words.
            most_common_df = helper.most_common_words_sentiment(user_selected, df, 0)

            # heading
            st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            # Data frame of most common negative words.
            most_common_df = helper.most_common_words_sentiment(user_selected, df, -1)

            # heading
            st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


