import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import re
import logging
import gensim.downloader as api
from typing import List, Union, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing class optimized for Quality dataset"""
    def __init__(self, language='english', use_stemming=False):
        try:
            # Download required NLTK data
            for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.warning(f"Error downloading {resource}: {str(e)}")

            self.stop_words = set(stopwords.words(language))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.use_stemming = use_stemming
            logger.info("TextPreprocessor initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing TextPreprocessor: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        try:
            text = str(text)
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            # Remove special characters and digits
            text = re.sub(r'[^\w\s]', '', text)
            # Remove square brackets and their contents
            text = re.sub(r'\[.*?\]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return text

    def preprocess(self, texts: Union[str, List[str]]) -> List[str]:
        try:
            if isinstance(texts, str):
                texts = [texts]

            processed_texts = []
            for text in tqdm(texts, desc="Preprocessing texts"):
                # Clean text
                cleaned = self.clean_text(text)
                # Tokenize
                tokens = word_tokenize(cleaned)
                # Convert to lowercase and remove stopwords
                tokens = [t.lower() for t in tokens if t.lower() not in self.stop_words]
                # Stem or lemmatize
                if self.use_stemming:
                    tokens = [self.stemmer.stem(t) for t in tokens]
                else:
                    tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                processed_texts.append(' '.join(tokens))

            return processed_texts
        except Exception as e:
            logger.error(f"Error in preprocess: {str(e)}")
            raise

class TextEmbeddingExtractor:
    """Text embedding extraction class"""
    def __init__(self,
                 embedding_model: str = 'word2vec-google-news-300',
                 normalize_vectors: bool = True):
        try:
            logger.info(f"Loading {embedding_model} embeddings...")
            self.embedding_model = api.load(embedding_model)
            self.vector_size = self.embedding_model.vector_size
            self.normalize_vectors = normalize_vectors
            self.scaler = StandardScaler() if normalize_vectors else None
            logger.info("TextEmbeddingExtractor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TextEmbeddingExtractor: {str(e)}")
            raise

    def get_document_vector(self, text: str) -> np.ndarray:
        try:
            words = text.split()
            word_vectors = []
            for word in words:
                if word in self.embedding_model:
                    word_vectors.append(self.embedding_model[word])

            if word_vectors:
                return np.mean(word_vectors, axis=0)
            return np.zeros(self.vector_size)
        except Exception as e:
            logger.error(f"Error in get_document_vector: {str(e)}")
            return np.zeros(self.vector_size)

    def extract_features(self, texts: List[str]) -> np.ndarray:
        try:
            feature_vectors = []
            for text in tqdm(texts, desc="Extracting features"):
                vector = self.get_document_vector(text)
                feature_vectors.append(vector)

            feature_vectors = np.array(feature_vectors)

            if self.normalize_vectors:
                feature_vectors = self.scaler.fit_transform(feature_vectors)

            # Additional normalization for better performance
            scaler = MinMaxScaler()
            feature_vectors = scaler.fit_transform(feature_vectors)

            return feature_vectors
        except Exception as e:
            logger.error(f"Error in extract_features: {str(e)}")
            raise

class EnsembleClassifier:
    """Ensemble classifier optimized for Quality dataset"""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

        # Initialize base classifiers
        self.classifiers = {
            'nb': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=random_state),
            'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=None, random_state=random_state),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=random_state)
        }

        self.ensemble = None
        self.model_weights = None

    def fit(self, X: np.ndarray, y: Union[np.ndarray, List[str]]) -> None:
        try:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)

            # Evaluate base models with cross-validation
            weights = []
            for name, model in self.classifiers.items():
                scores = cross_val_score(model, X, y_encoded, cv=5)
                weights.append(scores.mean())
                logger.info(f"{name.upper()} CV Score: {scores.mean():.4f}")

            # Normalize weights
            self.model_weights = np.array(weights) / sum(weights)

            # Create and fit ensemble
            self.ensemble = VotingClassifier(
                estimators=list(self.classifiers.items()),
                voting='soft',
                weights=self.model_weights,
                n_jobs=-1
            )

            self.ensemble.fit(X, y_encoded)
            logger.info("Ensemble model fitted successfully")

        except Exception as e:
            logger.error(f"Error in fit: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            predictions = self.ensemble.predict(X)
            return self.label_encoder.inverse_transform(predictions)
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

    def evaluate(self, X: np.ndarray, y: Union[np.ndarray, List[str]]) -> Dict:
        try:
            y_encoded = self.label_encoder.transform(y)
            y_pred = self.ensemble.predict(X)

            return {
                'accuracy': accuracy_score(y_encoded, y_pred),
                'classification_report': classification_report(y_encoded, y_pred,
                                                           target_names=self.label_encoder.classes_),
                'confusion_matrix': confusion_matrix(y_encoded, y_pred)
            }
        except Exception as e:
            logger.error(f"Error in evaluate: {str(e)}")
            raise

def prepare_quality_data(dataset_split):
    """Prepare Quality dataset for processing"""
    texts = []
    labels = []

    for item in dataset_split:
        # Use the 'text' field for text
        text = item['text']

        # Use the 'label' field for labels
        label = item['label']

        if text and label is not None:  # Only add if both text and label are present
            texts.append(text)
            labels.append(label)

    return texts, labels

def main():
    try:
        # Load dataset
        dataset = load_dataset("quality", split="train")
        logger.info("Dataset loaded successfully")

        # Prepare data
        texts, labels = prepare_quality_data(dataset)

        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        processed_texts = preprocessor.preprocess(texts)

        # Initialize embedding extractor
        embedding_extractor = TextEmbeddingExtractor()
        features = embedding_extractor.extract_features(processed_texts)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Initialize and train classifier
        classifier = EnsembleClassifier()
        classifier.fit(X_train, y_train)

        # Evaluate model
        evaluation_results = classifier.evaluate(X_test, y_test)
        logger.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info("Classification Report:\n" + evaluation_results['classification_report'])
        logger.info("Confusion Matrix:\n" + str(evaluation_results['confusion_matrix']))

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
