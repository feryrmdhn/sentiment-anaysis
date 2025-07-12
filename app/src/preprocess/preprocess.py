import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from app.src.preprocess.get_data import load_dataframe
from app.utils.utils import get_s3_client, upload_to_s3
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv('AWS_BUCKET_NAME')
if not bucket_name:
    raise ValueError("Environment variable AWS_BUCKET_NAME harus di set diawal!")

s3 = get_s3_client()

# Folder asset lokal
os.makedirs('asset/train', exist_ok=True)
os.makedirs('asset/test', exist_ok=True)
os.makedirs('asset/artifact', exist_ok=True)

if __name__ == '__main__':

    # Load data
    df = load_dataframe()

    # Rename kolom & drop kolom tidak perlu
    df.rename(columns={'Instagram Comment Text': 'Instagram_Comment_Text'}, inplace=True)
    df.drop(columns=['Id'], inplace=True)

    # Encode label
    encoder = LabelEncoder()
    df['Sentiment_Encoded'] = encoder.fit_transform(df['Sentiment'])

    # Split data stratify by Sentiment
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Sentiment']
    )

    # Preprocessing text
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = stopword.remove(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    train_df['cleaned_comment'] = train_df['Instagram_Comment_Text'].apply(preprocess_text)
    test_df['cleaned_comment']  = test_df['Instagram_Comment_Text'].apply(preprocess_text)

    train_df.drop(columns=['Instagram_Comment_Text'], inplace=True)
    test_df.drop(columns=['Instagram_Comment_Text'], inplace=True)

    train_df.rename(columns={'cleaned_comment': 'Instagram_Comment_Text'}, inplace=True)
    test_df.rename(columns={'cleaned_comment': 'Instagram_Comment_Text'}, inplace=True)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    X_train_tfidf = vectorizer.fit_transform(train_df['Instagram_Comment_Text'])
    X_test_tfidf = vectorizer.transform(test_df['Instagram_Comment_Text'])

    y_train = train_df['Sentiment_Encoded']
    y_test = test_df['Sentiment_Encoded']

    # Gabungkan fitur dan label (label di kolom pertama)
    train_arr = np.hstack([y_train.values.reshape(-1, 1), X_train_tfidf.toarray()])
    test_arr  = np.hstack([y_test.values.reshape(-1, 1), X_test_tfidf.toarray()])

    # Save ke CSV
    train_csv = 'asset/train/train.csv'
    test_csv  = 'asset/test/test.csv'
    np.savetxt(train_csv, train_arr, delimiter=",", fmt="%.6f")
    np.savetxt(test_csv,  test_arr,  delimiter=",", fmt="%.6f")

    # Save vectorizer
    vectorizer_filename = 'asset/artifact/tfidf_vectorizer.joblib'
    joblib.dump(vectorizer, vectorizer_filename)

    prefix = 'linear-learner-asset'

    # Upload ke S3
    upload_to_s3(train_csv, bucket_name, f'{prefix}/train/train.csv')
    upload_to_s3(test_csv,  bucket_name, f'{prefix}/test/test.csv')
    upload_to_s3(vectorizer_filename, bucket_name, f'{prefix}/artifact/tfidf_vectorizer.joblib')

    print('Semua file berhasil di-preprocess dan upload ke S3.')

    # Remove file lokal
    os.remove(train_csv)
    os.remove(test_csv)
    os.remove(vectorizer_filename)
