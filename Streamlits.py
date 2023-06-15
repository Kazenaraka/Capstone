import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import sleep
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import TimeSeriesSplit
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import tensorflow as tf
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import  Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.utils import pad_sequences
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia','bahwa','oleh','tak','jadi','hingga','bakal']
stop_words = stop_factory.get_stop_words()+more_stopword
stopword = stop_factory.create_stop_word_remover()
#Headlines
field = ['title','label', 'label_score']
headlines = pd.read_csv("data/annotated/combined/csv/main.csv", usecols=field)
scrapped = pd.read_csv("scraped_data.csv")


def preprocess(data, stopword):
    tokens = word_tokenize(data)
    tokens = [text.lower() for text in tokens]
    tokens = [text for text in tokens if text.isalpha()]
    tokens = [text for text in tokens if text not in stopword]
    return tokens


def clean_text(data):
    words = data[0]
    for num in range(1, len(data)):
        words += " " + data[num]
    return words

headlines["clean_text"] = [clean_text(preprocess(text, stop_words)) for text in headlines.title]
count_vec = CountVectorizer(ngram_range=(1,2), min_df = 2)
token = count_vec.fit_transform(headlines.clean_text)
with open('token.pkl', 'wb') as file:
    pickle.dump(token, file)

def create_wordcloud(data):
    comment_words = ""
    for val in data:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words, min_font_size=10).generate(
        comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()


def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0, 1], ['Not clickbait', 'Clickbait'])
    plt.yticks([0, 1], ['Not clickbait', 'Clickbait'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')


def main():
    # Load your dataset here
    headlines = pd.read_csv("data/annotated/combined/csv/main.csv", usecols=field)
    scraped = pd.read_csv("scraped_data.csv")
    scraped['Date'] = pd.to_datetime(scraped['Date'])
    scraped['Date'] = scraped['Date'].dt.date
    scraped = scraped.set_index('Date')
    st.title("Mendeteksi Clickbait Menggunakan Machine Learning")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Pendahuluan")
        st.write("Di era teknologi ini menjadi semakin mudah untuk mengakses informasi, Pengguna Handphone dan Personal Computer dapat menggunakan internet untuk mencari informasi di website. Eksploitasi berbasis web sudah mulai diketahui oleh mata publik, sehingga countermeasure akan diimplementasikan untuk menjaga experience user dalam menjelajah Internet" 
                 + "Dikarenakan penggunaan berita online semakin banyak, para penulis berita mulai memakai kata-kata yang cukup menyesatkan, dan juga tidak akurat dengan tujuan untuk user melihat headlines dan mendapatkan revenue berdasarkan click yang dilakukan oleh users. Salah satu metode yang digunakan oleh pembawa berita adalah clickbait." )
    with col2:
        st.subheader("Distribusi Headlines yang dijadikan model training memiliki distribusi normal")

        # Calculate the length of each headline
        headlines["length"] = [len(text.split()) for text in headlines.title]
        fig = go.Figure(data=go.Histogram(x=headlines["length"], nbinsx=50))

        # Update layout
        fig.update_layout(
            xaxis_title="Length",
            yaxis_title="Density"
        )
        fig.update_layout(width=400, height=480)
        # Render the interactive plot using Streamlit
        st.plotly_chart(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Hipotesis")
        st.write("Fenomena Clickbait semakin banyak dikarenakan aksesibilitas ke dalam internet yang sudah menjadi kehidupan sehari-hari, dan juga pengaruh perkembangan trends")
    with col4:
        st.subheader("Metode")
        st.write("Deteksi akan menggunakan algoritma Recurrent Neural Network dan menggunakan dataset CLICK-ID")


    # Create a sidebar with title and options
    st.sidebar.title("Clickbait Detection Dashboard")
    option = st.sidebar.selectbox("Select an option", ("Word Cloud", "Neural Network","Clickbait Check"))
    # Function to train the model
    @st.cache(allow_output_mutation=True)
    def train_model():
        # Split the dataset
        train_X, test_X, train_y, test_y = train_test_split(headlines.title, headlines.label_score, test_size=0.2, random_state=42)
        VOCAB_SIZE = 2000
        MAX_LEN = 50

        # Tokenize and pad the sequences
        tkz = Tokenizer(num_words=VOCAB_SIZE)
        tkz.fit_on_texts(train_X)
        sequences = tkz.texts_to_sequences(train_X)
        sequences = pad_sequences(sequences, maxlen=MAX_LEN)

        np.random.seed(42)
        tf.random.set_seed(42)

        # Build and compile the model
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, 50, input_length=MAX_LEN))
        model.add(LSTM(64))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

        sequences_test = tkz.texts_to_sequences(test_X)
        sequences_test = pad_sequences(sequences_test, maxlen=MAX_LEN)

        # Train the model
        RNN = model.fit(sequences, train_y, batch_size=128, validation_data=(sequences_test, test_y), epochs=15)

        # Evaluate the model
        loss, accuracy = model.evaluate(sequences_test, test_y)

        # Generate predictions
        y_pred_prob = model.predict(sequences_test)
        y_pred = np.round(y_pred_prob).flatten()
        
        # Generate the confusion matrix
        confusion_matrix = tf.math.confusion_matrix(test_y, y_pred)

        # Generate the model report
        report = {
            "loss": loss,
            "accuracy": accuracy,
            "epochs": RNN.epoch[-1] + 1,
            "history": RNN.history,
            "confusion_matrix": confusion_matrix,
            "labels" : sorted(list(set(test_y)))
        }
        
        return {"model": model, "report": report, "tokenizer": tkz}


    if option == "Word Cloud":
        # Generate and display the word cloud
        st.header("Kata-kata yang paling sering digunakan pada saat pengambilan dataset")
        create_wordcloud(headlines.title)

    elif option == "Neural Network":
        st.subheader("Neural Network Training")
    # Check if the model has been trained
        if 'model_data' not in st.session_state:
            st.session_state.model_data = None
            st.session_state.tkz = None
        # Train the model
        if st.button("Train Model") and st.session_state.model_data is None:
            st.write("Training in progress...")
            model_data = train_model()
            st.write("Training completed!")
            st.session_state.model_data = model_data

            # Store the tokenizer in session state
            VOCAB_SIZE = 2000
            tkz = Tokenizer(num_words=VOCAB_SIZE)
            tkz.fit_on_texts(headlines.title)
            st.session_state.tkz = tkz

        # Use the trained model
        if st.session_state.model_data is not None:
            model_data = st.session_state.model_data
            st.write("Model is ready for use!")
            # Interactive headline prediction
            st.subheader("Clickbait Detection Report")
            # Output the model report
            st.write("Model Report:")
            st.write(f"Loss: {model_data['report']['loss']}")
            st.write(f"Accuracy: {model_data['report']['accuracy']}")
            st.write(f"Epochs: {model_data['report']['epochs']}")
            st.subheader("Confusion Matrix")
            fig = go.Figure(data=go.Heatmap(
                z=model_data['report']['confusion_matrix'],
                x=model_data['report']['labels'],
                y=model_data['report']['labels'],
                colorscale='Blues'
            ))
            fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )
            fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=np.arange(len(model_data['report']['labels'])) + 0.5,
                        ticktext=model_data['report']['labels']
                    ),
                    yaxis=dict(
                        tickmode='array',
                        tickvals=np.arange(len(model_data['report']['labels'])) + 0.5,
                        ticktext=model_data['report']['labels']
                    )
                )
            fig.update_layout(
                    annotations=[
                        go.layout.Annotation(
                            x=i,
                            y=j,
                            text=str(model_data['report']['confusion_matrix'][j][i].numpy()),
                            showarrow=False,
                            font=dict(color='white' if i != j else 'black')
                        )
                        for i in range(len(model_data['report']['labels']))
                        for j in range(len(model_data['report']['labels']))
                    ]
                )
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig)
                # Plot accuracy and loss
            st.subheader("Accuracy and Loss")
            epochs = list(range(1, model_data['report']['epochs'] + 1))
            history = model_data['report']['history']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], mode='lines', name='Accuracy'))
            fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], mode='lines', name='Validation Accuracy'))
            fig.add_trace(go.Scatter(x=epochs, y=history['loss'], mode='lines', name='Loss'))
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], mode='lines', name='Validation Loss'))
            fig.update_layout(title="Accuracy and Loss", xaxis_title="Epochs", yaxis_title="Value")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig)
    elif option == "Clickbait Check":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            MAX_LEN = 50
            df = pd.read_csv(uploaded_file)

            # Access the 'Article' and 'Date' columns in the DataFrame for processing and classification
            article_texts = df['Article'].tolist()
            dates = pd.to_datetime(df['Date'])
            # Instantiate the tokenizer and fit on the article texts
            model_data = train_model()
            tkz = model_data["tokenizer"]
            # Retrieve the model from session state
            model = st.session_state.model_data['model']
            # Perform clickbait classification for each article
            encoded_articles = tkz.texts_to_sequences(article_texts)
            encoded_articles = pad_sequences(encoded_articles, maxlen=MAX_LEN)
            labels = model.predict(encoded_articles)
            df = pd.DataFrame({"Article": [article_texts], "Label": [int(labels)]})
            st.write(df)
            # Update the DataFrame with the predicted labels
            df['Label'] = [1 if label > 0.5 else 0 for label in labels]

            # Convert the 'Date' column to datetime for easier manipulation
            df['Date'] = pd.to_datetime(df['Date'])
            count_df = df.groupby('Date')['Label'].value_counts().unstack(fill_value=0).reset_index()

            # Group the data by 'Label' and count the occurrences for the pie chart
            label_counts = df.groupby('Label').size().reset_index(name='Count')
            # Map the label values to corresponding names
            label_counts['Label'] = label_counts['Label'].map({0: 'non-clickbait', 1: 'clickbait'})
            # Create a pie chart to show the distribution of clickbait and non-clickbait articles
            pie_chart = px.pie(label_counts, values='Count', names='Label', 
                            title='Distribution of Clickbait and Non-Clickbait Articles')

            # Display the line chart, bar chart, and pie chart in the dashboard
            if not df.empty:
                st.plotly_chart(pie_chart)
            else:
                st.write("No data to display.")

# Run the main function
if __name__ == '__main__':
    main()
