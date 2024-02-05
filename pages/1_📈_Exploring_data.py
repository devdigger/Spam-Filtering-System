import streamlit as st
from classifier import EmailClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

email_classifier = EmailClassifier()
email_classifier.load_model()

if 'df' not in st.session_state:

    df = email_classifier.load_dataset()
    print(df)
    st.session_state['df'] = df
else:
    df = st.session_state['df']




if df is not None:
    st.title("WordCloud Plot")

    if 'wordcloud_fig' not in st.session_state:
        print("Not cacheed")
        with st.spinner("Generating WordCloud..."):
            wordcloud_fig = email_classifier.visualize_wordcloud(' '.join(df['text']))
        
        st.session_state['wordcloud_fig'] = wordcloud_fig
    else:
        print("WordCloud loaded from cache.")
        wordcloud_fig = st.session_state['wordcloud_fig']
    
    st.pyplot(wordcloud_fig)


    st.title("Email Classification Chart")
    loading_text = st.text("Generating PieChart...")
    if 'pie_chart_fig' not in st.session_state:
        pie_chart_fig = email_classifier.visualize_distribution(df)
        st.session_state['pie_chart_fig'] = pie_chart_fig
    else:
        print("pie_chart_fig loaded from cache.")
        pie_chart_fig = st.session_state['pie_chart_fig']
    loading_text.empty()
    st.pyplot(pie_chart_fig)


    


    st.title("Model Evaluation")

    if 'test_accuracy' not in st.session_state or 'cm' not in st.session_state or 'report_df' not in st.session_state:

        X, y = email_classifier.prepare_features_labels(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        email_classifier.train_model(X_train, y_train,st)
        test_accuracy, cm, report_df = email_classifier.evaluate_model(X_test, y_test)
        st.session_state['test_accuracy'],st.session_state['cm'],st.session_state['report_df'] = test_accuracy, cm, report_df
        st.session_state['model'] = email_classifier
    else:
        print("Loaded test_accuracy,cm,report_df from cache")

        test_accuracy, cm, report_df = st.session_state['test_accuracy'],st.session_state['cm'],st.session_state['report_df']
        email_classifier = st.session_state['model']


    
    cm_fig, cm_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, ax=cm_ax)
    cm_ax.set_xlabel('Predicted Labels')
    cm_ax.set_ylabel('True Labels')
    cm_ax.set_title('Confusion Matrix')

    st.subheader(f"Test Accuracy: {str(test_accuracy)}")

    st.subheader("Confusion Matrix:")
    st.pyplot(cm_fig)

    st.subheader("Classification Report:")
    st.write(report_df)

    st.session_state['model'] = email_classifier

