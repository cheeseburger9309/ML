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
    """Text preprocessing class optimized for MULD dataset"""
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
            # Remove square brackets and their contents (specific to MULD)
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
    """Ensemble classifier optimized for MULD dataset"""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

        # Initialize base classifiers with optimized parameters
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
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

def prepare_muld_data(dataset_split):
    """Prepare MULD dataset for processing"""
    texts = []
    labels = []


    for item in dataset_split:
        # Use the 'input' field for text
        text = item['input']

        # Use the 'output' field for labels
        # Since 'output' is a list, we'll take the first element
        label = item['output'][0] if item['output'] else None

        if text and label:  # Only add if both text and label are present
            texts.append(text)
            labels.append(label)

    return texts, labels

class MULDClassificationPipeline:
    """Complete classification pipeline for MULD dataset"""
    def __init__(self,
                 embedding_model: str = 'word2vec-google-news-300',
                 use_stemming: bool = False,
                 random_state: int = 42):
        self.preprocessor = TextPreprocessor(use_stemming=use_stemming)
        self.embedding_extractor = TextEmbeddingExtractor(embedding_model=embedding_model)
        self.classifier = EnsembleClassifier(random_state=random_state)
        logger.info("MULD classification pipeline initialized")

    def fit(self, texts: List[str], labels: List[str]) -> None:
        try:
            logger.info("Starting text preprocessing...")
            processed_texts = self.preprocessor.preprocess(texts)

            logger.info("Extracting features...")
            X_features = self.embedding_extractor.extract_features(processed_texts)

            logger.info("Training ensemble classifier...")
            self.classifier.fit(X_features, labels)

            logger.info("Pipeline training completed successfully")

        except Exception as e:
            logger.error(f"Error in pipeline fit: {str(e)}")
            raise

    def predict(self, texts: List[str]) -> np.ndarray:
        try:
            processed_texts = self.preprocessor.preprocess(texts)
            X_features = self.embedding_extractor.extract_features(processed_texts)
            return self.classifier.predict(X_features)
        except Exception as e:
            logger.error(f"Error in pipeline predict: {str(e)}")
            raise

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        try:
            processed_texts = self.preprocessor.preprocess(texts)
            X_features = self.embedding_extractor.extract_features(processed_texts)
            return self.classifier.evaluate(X_features, labels)
        except Exception as e:
            logger.error(f"Error in pipeline evaluate: {str(e)}")
            raise

    def save_pipeline(self, filepath: str) -> None:
        """Save the complete pipeline to disk"""
        joblib.dump(self, filepath)

    @staticmethod
    def load_pipeline(filepath: str) -> 'MULDClassificationPipeline':
        """Load the complete pipeline from disk"""
        return joblib.load(filepath)

def plot_results(evaluation_dict: Dict, pipeline: MULDClassificationPipeline, title_prefix: str = ""):
    """Plot evaluation results"""
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(evaluation_dict['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=pipeline.classifier.label_encoder.classes_,
                yticklabels=pipeline.classifier.label_encoder.classes_)
    plt.title(f'{title_prefix} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot model weights
    plt.figure(figsize=(10, 6))
    model_names = list(pipeline.classifier.classifiers.keys())
    model_weights = pipeline.classifier.model_weights
    plt.bar(model_names, model_weights)
    plt.title(f'{title_prefix} Model Weights in Ensemble')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load MULD dataset
    logger.info("Loading MULD dataset...")
    dataset = load_dataset("ghomasHudson/muld", name="Character Archetype Classification")

   # Print dataset information
    print("\nDataset structure:")
    print(dataset)

    # Prepare data
    train_texts, train_labels = prepare_muld_data(dataset['train'])
    val_texts, val_labels = prepare_muld_data(dataset['validation'])
    test_texts, test_labels = prepare_muld_data(dataset['test'])

    # Print some statistics
    print("\nDataset statistics:")
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print("\nUnique labels:", set(train_labels))

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = MULDClassificationPipeline(
        embedding_model='glove-wiki-gigaword-100',  # Using smaller model for faster processing
        use_stemming=False
    )

    # Train the pipeline
    logger.info("Training pipeline...")
    pipeline.fit(train_texts, train_labels)

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_evaluation = pipeline.evaluate(val_texts, val_labels)
    print("\nValidation Results:")
    print("------------------")
    print("Accuracy:", val_evaluation['accuracy'])
    print("\nClassification Report:")
    print(val_evaluation['classification_report'])

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_evaluation = pipeline.evaluate(test_texts, test_labels)
    print("\nTest Results:")
    print("-------------")
    print("Accuracy:", test_evaluation['accuracy'])
    print("\nClassification Report:")
    print(test_evaluation['classification_report'])

    # Plot results
    plot_results(val_evaluation, pipeline, "Validation")
    plot_results(test_evaluation, pipeline, "Test")

    # Save the pipeline
    pipeline.save_pipeline('muld_classification_pipeline.joblib')
    logger.info("Pipeline saved successfully")
