import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import mlflow

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

mlflow.set_experiment("Course Recommendation")

def load_data(file_path):
    """Memuat dataset dari file CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def remove_duplicates(df):
    """Menghapus data duplikat."""
    try:
        num_duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        mlflow.log_metric("num_duplicates_removed", num_duplicates)
        return df
    except Exception as e:
        print(f"Error removing duplicates: {e}")
        return None

def fill_missing_rating(df):
    """Mengisi nilai NaN pada kolom 'Rating' dengan nilai rata-rata."""
    try:
        mean_rating = df['Rating'].mean()
        df['Rating'].fillna(mean_rating, inplace=True)
        mlflow.log_metric("mean_rating_used", mean_rating)
        return df
    except Exception as e:
        print(f"Error filling missing ratings: {e}")
        return None

def clean_text(text):
    """Membersihkan teks dari karakter non-huruf dan stopwords."""
    try:
        text = re.sub(r'[^a-zA-Z]', ' ', str(text))  # Pastikan input adalah string
        text = text.lower()
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
        return ' '.join(text)
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

def encode_data(df):
    """Melakukan label encoding pada kolom 'Level' dan 'Keyword'."""
    try:
        label_encoder = LabelEncoder()
        df['Level'] = label_encoder.fit_transform(df['Level'])
        df['Keyword'] = label_encoder.fit_transform(df['Keyword'])
        mlflow.log_param("encoded_columns", ["Level", "Keyword"])
        return df
    except Exception as e:
        print(f"Error encoding data: {e}")
        return None

def extract_features(df):
    """Ekstraksi fitur TF-IDF dari kolom 'What you will learn'."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(df['What you will learn'])
        learn_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        mlflow.log_param('Feature_extraction_method', 'TF-IDF')
        mlflow.log_param('Original column', 'What you will learn')
        return learn_df, vectorizer
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None

def preprocess_data(df, file_path):
    """
    Melakukan semua langkah preprocessing secara berurutan.
    Menyimpan dataframe hasil ke CSV dengan nama file `preprocessed_data.csv`.
    """
    try:
        mlflow.log_param("file_path", file_path) 
        df = remove_duplicates(df)
        if df is None: raise Exception("remove_duplicates failed")
        df = fill_missing_rating(df)
        if df is None: raise Exception("fill_missing_rating failed")

        # Hilangkan koma dan kata 'reviews' lalu ubah ke float
        df['Review'] = df['Review'].str.replace(',', '').str.replace(' reviews', '').astype(float)
        df['What you will learn'] = df['What you will learn'].fillna('')
        df['Skill gain'] = df['Skill gain'].fillna('[]')
        df['Level'] = df['Level'].fillna('Unknown')

        df['Course Title'] = df['Course Title'].astype(str).apply(clean_text)
        df['What you will learn'] = df['What you will learn'].astype(str).apply(clean_text)
        df = encode_data(df)
        if df is None: raise Exception("encode_data failed")

        df['Skill gain'] = df['Skill gain'].astype(str)
        df['Offered By'] = df['Offered By'].astype(str)
        df.drop(['Course Url', 'Modules', 'Instructor', 'Schedule', 'Duration'], axis=1, inplace=True)

        learn_df, vectorizer = extract_features(df)
        if learn_df is None: raise Exception("extract_features failed")
        df = pd.concat([df.reset_index(drop=True), learn_df.reset_index(drop=True)], axis=1)

        df.to_csv(file_path, index=False)
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

if __name__ == "__main__":
    with mlflow.start_run(run_name="Preprocessing_Run"):
        file_path = 'CourseraDataset-Unclean.csv'
        data = load_data(file_path)

        if data is not None:
            preprocessed_df = preprocess_data(data, 'preprocessing/preprocessed_data.csv')

            if preprocessed_df is not None:
                 mlflow.log_artifact('preprocessing/preprocessed_data.csv', artifact_path="preprocessed_data")
                 print("Preprocessing selesai! Data tersimpan di 'preprocessing/preprocessed_data.csv'")
            else:
                print("Preprocessing failed. Check the errors above.")
        else:
            print("Failed to load data.")