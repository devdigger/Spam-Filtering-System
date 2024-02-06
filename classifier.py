import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import os
from datetime import datetime, timedelta
import sys,time

class EmailClassifier:
    def __init__(self):
        self.svm_classifier = None
        self.hashing_vectorizer = HashingVectorizer(stop_words='english')

    def load_model(self, model_filename="svm_classifier.joblib"):
        current_directory = os.getcwd()
        svm_path = os.path.join(current_directory, model_filename)
        hashv_path = os.path.join(current_directory, "hashing_vectorizer.joblib")
        if os.path.isfile(svm_path):
            self.svm_classifier = joblib.load(svm_path)
            
            print(f"Model loaded from {svm_path}")
        else:
            print(f"The model file {svm_path} does not exist. Please train and save the model first.")
            self.svm_classifier = None
        if os.path.isfile(hashv_path):
            self.hashing_vectorizer = joblib.load(hashv_path)
            print(f"Model loaded from {hashv_path}")
        else:
            self.hashing_vectorizer = HashingVectorizer(stop_words='english', lowercase=True)

    def load_dataset(self, csv_filename="spam_ham_dataset.csv"):
        current_directory = os.getcwd()
        dataset_path = os.path.join(current_directory, csv_filename)
        print(dataset_path)
        if os.path.isfile(dataset_path):
            print("Dataset Found in current directory")
            df = pd.read_csv(dataset_path)
            df = df.drop(df.columns[0], axis=1)
            df = df.drop("label", axis=1)
            print(df['text'])
            return df
        else:
            print(f"The dataset file {dataset_path} does not exist. Please make sure it's in the current directory.")
            return None
    
    def visualize_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    
    
    def visualize_distribution(self, df):
        spam_count = df['label_num'].sum()
        non_spam_count = len(df) - spam_count
        labels = ['Spam', 'Non-Spam']
        sizes = [spam_count, non_spam_count]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title('Email Classification Distribution')

        # Display the pie chart using st.pyplot
        return fig
    
    def prepare_features_labels(self, df):
        X = self.hashing_vectorizer.fit_transform(df['text'])
        y = df['label_num']
        return X, y
    
    
    def train_model_batch(self, X_train, y_train,st,batch_size=32):
        # self.svm_classifier = SGDClassifier(loss="hinge", alpha=0.001, max_iter=100, random_state=42)
        # for i in range(X_train.shape[0]):
        #     x_i = X_train[i]
        #     y_i = y_train.iloc[i]
        #     x_i = x_i.reshape(1, -1)
        #     self.svm_classifier.partial_fit(x_i, [y_i], classes=[0, 1])
        #     if (i + 1) % 100 == 0:
        #         sys.stdout.write("\rTraining progress: {}/{} data points processed".format(i + 1, X_train.shape[0]))
        #         sys.stdout.flush()
        # sys.stdout.flush()
        # print("\nModel training complete.")
        # self.svm_classifier = SGDClassifier(loss="hinge", alpha=0.001, max_iter=100, random_state=42)
        # total_data_points = X_train.shape[0]

        # progress_text = "Model Training in progress.... "
       
        # progress_bar = st.progress(0,text=progress_text)

        # for i in range(total_data_points):
        #     x_i = X_train[i]
        #     y_i = y_train.iloc[i]
        #     x_i = x_i.reshape(1, -1)
        #     self.svm_classifier.partial_fit(x_i, [y_i], classes=[0, 1])
        #     # print(i,total_data_points)
        #     # Update the progress bar
        #     progress_percentage = ((i ) / total_data_points)
        #     append = " [ "+ f"{i}/{total_data_points}" + " ] "
        #     progress_bar.progress(progress_percentage,text=progress_text+append)

        # progress_bar.empty()
        # print("\nModel training complete.")
        progress_text = "Model Training in progress.... "
       
        progress_bar = st.progress(0,text=progress_text)
        

        # Initialize the SGDClassifier
        self.svm_classifier = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=100, random_state=42)

        # Iterate over mini-batches
        for i in range(0, X_train.shape[0], batch_size):
            # Get the current mini-batch
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train.iloc[i:i+batch_size]

            # Update the SVM model with the current mini-batch
            self.svm_classifier.partial_fit(X_batch, y_batch, classes=[0, 1])

            # Print training progress
            sys.stdout.write("\rTraining progress: {}/{} mini-batches processed".format(i // batch_size + 1, X_train.shape[0] // batch_size +1))
            sys.stdout.flush()
        
            progress_percentage = ((i // batch_size + 1) / (X_train.shape[0] // batch_size +1))
            append = " [ "+ f"{(i // batch_size + 1)}/{(X_train.shape[0] // batch_size +1)}" + " ] "
            progress_bar.progress(progress_percentage,text=progress_text+append)

        progress_bar.empty()
        print("\nModel training complete.")
    
    
    def train_model(self, X_train, y_train,st):

        self.svm_classifier = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=100, random_state=42)
        with st.spinner("Training Model..."):
            self.svm_classifier.fit(X_train, y_train)
        print("\nModel training complete.")

    
    def evaluate_model(self, X_test, y_test):
        # y_pred = self.svm_classifier.predict(X_test)
        # test_accuracy = accuracy_score(y_test, y_pred)
        # print("Test Accuracy:", test_accuracy)

        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.title('Confusion Matrix')
        # plt.show()

        # class_names = ['spam', 'non-spam']
        # report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        # report_df = pd.DataFrame(report_dict).transpose()
        # styled_report_df = report_df.style \
        #     .format("{:.2f}", subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']]) \
        #     .background_gradient(cmap='coolwarm', subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']]) \
        #     .set_caption('Classification Report') \
        #     .set_table_styles([{'selector': 'caption',
        #                         'props': [('font-size', '16px'), ('font-weight', 'bold')]}])
        # styled_report_df
        y_pred = self.svm_classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy:", test_accuracy)

        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=['spam', 'non-spam'], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        return test_accuracy, cm, report_df

    def test_single_mail(self, single_message):
        single_message_lower = single_message.lower()
        single_message = self.hashing_vectorizer.transform([single_message_lower])

        prediction = self.svm_classifier.predict(single_message)
        if prediction == 1:
            print("The message is predicted as spam.")
        else:
            print("The message is predicted as not spam.")
        # self.svm_classifier.partial_fit(single_message_tfidf, prediction, classes=[0, 1])
        print("returning :",prediction,type(prediction))
        return prediction
    
    def update_model(self,label,single_message):
        # single_message_lower = single_message.lower()
        # single_message_tfidf = self.tfidf_vectorizer.transform([single_message_lower])
        # label_flattened = np.array(label).ravel()
        # self.svm_classifier.partial_fit(single_message_tfidf, label_flattened, classes=[0, 1])
        # self.save_model()
        single_message_features = self.hashing_vectorizer.transform([single_message])

        label_flattened = np.array(label).ravel()
        # Update the SVM model with the new data
        self.svm_classifier.partial_fit(single_message_features, label_flattened, classes=[0, 1])
        self.save_model()

    def save_model(self, model_filename="svm_classifier.joblib"):
        joblib.dump(self.svm_classifier, model_filename)
        joblib.dump(self.hashing_vectorizer, 'hashing_vectorizer.joblib', compress=True)

        current_utc_time = datetime.utcnow()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist_time = current_utc_time + ist_offset
        pretty_datetime_ist = current_ist_time.strftime("%A, %B %d, %Y %I:%M %p IST")

        print(f"Model saved as {model_filename}")
        print("Date and Time:", pretty_datetime_ist)

# Example usage:
# email_classifier = EmailClassifier()
# email_classifier.load_model()
# df = email_classifier.load_dataset()
# email_classifier.visualize_wordcloud(' '.join(df['text']))
# email_classifier.visualize_distribution(df)
# X, y = email_classifier.prepare_features_labels(df)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# email_classifier.train_model(X_train, y_train)
# email_classifier.evaluate_model(X_test, y_test)
# email_classifier.test_single_mail("Subject: Exclusive Summer Sale - Up to 50% Off on Beachwear! Dear Valued Customer, Summer is here, and it's time to soak up the sun in style! We are excited to announce our Exclusive Summer Sale, where you can enjoy massive discounts of up to 50% on our latest collection of beachwear and summer essentials.")
# email_classifier.save_model()
