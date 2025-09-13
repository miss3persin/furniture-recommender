from metaflow import FlowSpec, step
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class FurnitureRecommenderFlow(FlowSpec):

    @step
    def start(self):
        print("Loading dataset...")
        self.df = pd.read_csv("rentals_furniture_100.csv")
        print(f"Loaded {len(self.df)} rows with columns: {list(self.df.columns)}")
        self.next(self.clean_data)

    @step
    def clean_data(self):
        print("Cleaning dataset...")
        required_cols = [
            "rental_id",
            "apartment_type",
            "location",
            "renter_type",
            "budget_range",
            "preferred_style",
            "recommended_furniture",
        ]

        # Drop rows with missing important values
        self.df = self.df.dropna(subset=required_cols)

        # Ensure text columns are strings
        self.df["preferred_style"] = self.df["preferred_style"].astype(str)
        self.df["recommended_furniture"] = self.df["recommended_furniture"].astype(str)

        print(f"After cleaning: {len(self.df)} rows remain.")
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Training recommendation model...")

        # Combine style + furniture into one text field
        self.df["combined_text"] = (
            self.df["preferred_style"] + " " + self.df["recommended_furniture"]
        )

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix
        )

        # Save model artifacts
        with open("model.pkl", "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.tfidf_matrix,
                    "similarity": self.similarity_matrix,
                    "df": self.df,
                },
                f,
            )

        print("✅ Model trained and saved as model.pkl")
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    FurnitureRecommenderFlow()
