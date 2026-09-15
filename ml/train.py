"""
train.py — Dataset generation and ML model training.

Key improvements over v1:
  1. Corpus completely replaces short pangrams with varied English prose —
     pangrams have near-uniform letter distributions (IC ≈ 0.044) that fall
     inside the Vigenere IC range and confuse the model.
  2. All 26 Caesar keys (0–25) are explicitly sampled to ensure every shift
     is represented in training data.
  3. Vigenere training uses a wide range of key lengths (2–12) so the model
     sees the full IC-profile variety for short and long keys.
  4. Text lengths are randomized (80–400 chars) to avoid length-based artifacts.
  5. 800 samples per class (up from 600) for better generalization.
  6. No training/test data leakage — split is done after all samples are built.

Run once to train and save:
    python -m ml.train
"""

import os
import sys
import random
import string
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciphers.caesar import CaesarCipher
from ciphers.vigenere import VigenereCipher
from ciphers.substitution import SubstitutionCipher
from ml.features import extract_features, FEATURE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SAMPLES_PER_CLASS = 800
TEST_SPLIT = 0.20
RANDOM_SEED = 42
LABELS = ["Caesar", "Vigenere", "Substitution"]

# ─────────────────────────────────────────────────────────────────────────────
# Diverse English prose corpus
#
# CRITICAL: No pangrams. Every sentence here uses typical English letter
# frequencies (IC ≈ 0.065–0.068). Pangrams ("THE QUICK BROWN FOX…") use
# all 26 letters nearly equally → IC ≈ 0.038–0.044, which overlaps with
# Vigenere ciphertext. Using pangrams in training makes Caesar look like
# Vigenere to the model.
# ─────────────────────────────────────────────────────────────────────────────

ENGLISH_PROSE = [
    # Literature / general prose
    "it was the best of times it was the worst of times it was the age of wisdom",
    "call me ishmael some years ago never mind how long precisely having little money",
    "in the beginning was the word and the word was with god and the word was god",
    "to be or not to be that is the question whether tis nobler in the mind to suffer",
    "all happy families are alike each unhappy family is unhappy in its own way",
    "it is a truth universally acknowledged that a single man in possession of a good fortune",
    "it was a bright cold day in april and the clocks were striking thirteen",
    "the man in black fled across the desert and the gunslinger followed after him",
    "when he was nearly thirteen my brother jem got his arm badly broken at the elbow",
    "if you really want to hear about it the first thing you probably want to know",
    "many years later as he faced the firing squad colonel aureliano buendia was to remember",
    "the sky above the port was the color of television tuned to a dead channel",
    "whether i shall turn out to be the hero of my own life or whether that station",
    "happy families are all alike every unhappy family is unhappy in its own way indeed",
    "last night i dreamt i went to manderley again it seemed to me i stood by the iron gate",
    "riverrun past eve and adam from swerve of shore to bend of bay brings us by",
    "the man was thin and his shoulders were narrow and his face was long and drawn",
    "it was a dark and stormy night the rain fell in torrents except at occasional intervals",
    "a man walked through the park and sat on a bench near the fountain drinking coffee",
    "she opened the door slowly and looked into the room before stepping inside carefully",
    # Science / education / technology prose
    "cryptography is the practice and study of techniques for securing communications",
    "machine learning is a method of data analysis that automates analytical model building",
    "the frequency analysis of letters in a language provides powerful tools for cryptanalysis",
    "the index of coincidence measures how nonuniform the letter distribution is in a text",
    "classical ciphers operate by substituting or transposing letters in the plaintext",
    "the vigenere cipher uses a repeating keyword to apply multiple caesar shifts in sequence",
    "a substitution cipher replaces each letter with a different letter in a fixed mapping",
    "the caesar cipher shifts each letter a fixed number of positions forward in the alphabet",
    "statistical analysis reveals patterns in encrypted text that help recover the original",
    "natural language has characteristic letter frequencies that differ from random distributions",
    "information theory defines entropy as a measure of the unpredictability of information",
    "random forests are ensemble learning methods that operate by constructing decision trees",
    "feature extraction transforms raw data into numerical vectors suitable for machine learning",
    "the training set is used to fit the model parameters while the test set evaluates performance",
    "cross validation helps detect overfitting by evaluating the model on held out data",
    "precision measures the fraction of relevant instances among the retrieved instances",
    "recall measures the fraction of relevant instances that were actually retrieved correctly",
    "the confusion matrix shows the performance of a classification algorithm in tabular form",
    "neural networks are computing systems vaguely inspired by biological brains and neurons",
    "the internet enables global communication and has transformed how people share information",
    # History / general knowledge
    "ancient civilizations developed writing systems to record information and communicate",
    "the roman empire spanned much of europe africa and parts of asia for many centuries",
    "during the second world war cryptanalysis played a decisive role in military operations",
    "the enigma machine was used by germany to encode military communications during the war",
    "bletchley park was the site where british mathematicians worked to break german codes",
    "alan turing developed theoretical foundations for computation and artificial intelligence",
    "mathematics is the foundation of computer science and all digital communication systems",
    "the renaissance was a period of great intellectual and artistic growth in european history",
    "the printing press revolutionized the spread of knowledge across europe and the world",
    "democracy requires an informed citizenry and freedom of information and expression always",
    # Long prose passages (important for Kasiski-based features to work well)
    "in those days the darkness had not yet come to the land and men were still free to walk",
    "the river wound through the forest casting its reflection on the water below the trees",
    "she had always known that one day she would have to make a decision that would change",
    "the professor stood at the board and wrote a series of equations that none could follow",
    "memories of those early days returned to him as he sat in his study reading old letters",
    "the committee met every thursday morning in the conference room on the third floor level",
    "communication between the two governments had been strained for many years before that",
    "the investigation revealed that several individuals had been involved in the conspiracy",
    "scientists have discovered that many animals communicate using complex systems of signals",
    "the laboratory was equipped with the latest instruments for measuring chemical reactions",
    "historians disagree about the causes of the conflict that began in the summer of that year",
    "the novel explores themes of identity power and the struggle for freedom in modern society",
    "the committee recommended several changes to the policy that would improve efficiency greatly",
    "researchers found that regular practice significantly improved the performance of students",
    "the architecture of the building reflected the aesthetic values of the early twentieth century",
    "observers noted that the situation had changed considerably since the previous assessment",
    "the manuscript contained detailed descriptions of ceremonies and rituals of the ancient people",
    "the expedition traveled through remote territory mapping rivers mountains and settlements",
    "the settlement was established in the early years of the colony and grew rapidly after that",
    "the treaty established terms for trade between the nations and remained in force for decades",
]


def _generate_plaintext(min_words: int = 12, max_words: int = 60) -> str:
    """
    Sample a random plaintext from the prose corpus.
    Word count is randomized to produce varied text lengths (≈80–400 letters).
    Using a real corpus (not random words) ensures English letter frequencies
    are preserved, giving Caesar ciphertext a characteristic IC ≈ 0.065.
    """
    all_words = " ".join(ENGLISH_PROSE).split()
    num_words = random.randint(min_words, max_words)
    return " ".join(random.choices(all_words, k=num_words))


def generate_dataset(samples_per_class: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Generate a labeled dataset of ciphertext samples.

    Caesar:  All 26 keys (0–25) are explicitly sampled round-robin to ensure
             every shift is represented. key=0 is included as it preserves
             English statistics (IC ≈ 0.067) and teaches the model that
             the flat IC profile — not just "looks like English" — is the
             Caesar signature.

    Vigenere: Keys of length 2..12 are sampled so the model sees the full
              range of IC-profile shapes. Short keys (2–3) produce IC profiles
              closer to Caesar; long keys produce very flat, low IC profiles.

    Substitution: Random 26-letter permutations ensure diverse frequency
                  orderings while IC stays near English IC (≈ 0.065).
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    caesar = CaesarCipher()
    vigenere = VigenereCipher()
    subst = SubstitutionCipher()

    records = []

    print(f"Generating {samples_per_class} samples per class...")

    # ── Caesar samples ────────────────────────────────────────────────────────
    # Explicitly cycle through all 26 keys so each key appears ~samples/26 times.
    # This prevents the model from learning key-specific artifacts.
    print("  -> Caesar cipher samples...")
    for i in range(samples_per_class):
        plain = _generate_plaintext()
        key = i % 26          # ensures every key 0–25 is used uniformly
        ciphertext = caesar.encrypt(plain, key)
        feat = extract_features(ciphertext)
        if not np.all(feat == 0):
            records.append(list(feat) + ["Caesar"])

    # ── Vigenere samples ──────────────────────────────────────────────────────
    # Use a wide range of key lengths (2..12) with realistic keywords.
    # Short keys (2–3) are most confusable with Caesar, so we include many.
    print("  -> Vigenere cipher samples...")
    vigenere_keys = [
        # 2-letter keys (most confusable with Caesar — short key → IC closer to Caesar)
        "AB", "XY", "QR", "MN", "ZA",
        # 3-letter keys
        "KEY", "ABC", "CAT", "DOG", "RUN",
        # 4-letter keys
        "CODE", "LOCK", "PASS", "WORD", "SAFE",
        # 5-letter keys
        "ALPHA", "SIGMA", "DELTA", "OMEGA", "THETA",
        # 6-letter keys
        "SECRET", "CIPHER", "PYTHON", "CRYPTO", "HIDDEN",
        # 7-letter keys
        "KEYWORD", "MYSTERY", "PROTECT", "ENCRYPT", "DIGITAL",
        # 8-letter keys
        "SECURITY", "ENCODING", "LANGUAGE", "COMPUTER", "ANALYSIS",
        # 9-letter keys
        "ALGORITHM", "FREQUENCY", "KNOWLEDGE", "CLASSICAL", "EDUCATION",
        # 10+ letter keys (very flat IC, clearly distinguishable)
        "CRYPTOLOGY", "STATISTICS", "INFORMATION", "CRYPTANALYSIS",
    ]
    for i in range(samples_per_class):
        plain = _generate_plaintext()
        key = vigenere_keys[i % len(vigenere_keys)]
        ciphertext = vigenere.encrypt(plain, key)
        feat = extract_features(ciphertext)
        if not np.all(feat == 0):
            records.append(list(feat) + ["Vigenere"])

    # ── Substitution samples ──────────────────────────────────────────────────
    print("  -> Substitution cipher samples...")
    for _ in range(samples_per_class):
        plain = _generate_plaintext()
        key = subst.random_key()
        ciphertext = subst.encrypt(plain, key)
        feat = extract_features(ciphertext)
        if not np.all(feat == 0):
            records.append(list(feat) + ["Substitution"])

    columns = FEATURE_NAMES + ["label"]
    df = pd.DataFrame(records, columns=columns)
    print(f"  Total samples generated: {len(df)}")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest classifier and return measured metrics.

    Model choice: Random Forest
      - Handles non-linear decision boundaries (important: Caesar vs Substitution
        have similar IC but differ in frequency ordering — non-linear boundary)
      - Provides feature importances (educational value for the project)
      - Robust to the scale differences between feature types (no normalization needed)
      - Fast training on datasets of this size

    No metrics are hardcoded. All numbers reported come from the actual
    held-out test set evaluation.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    from sklearn.preprocessing import LabelEncoder

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    le = LabelEncoder()
    le.fit(LABELS)   # fix label order: Caesar=0, Substitution=1, Vigenere=2
    y_enc = le.transform(y)

    # Stratified split — keeps class balance in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )

    print(f"\nTraining Random Forest on {len(X_train)} samples, testing on {len(X_test)}...")

    clf = RandomForestClassifier(
        n_estimators=200,       # more trees → better generalization
        max_depth=None,         # fully grown trees (pruned by min_samples_leaf)
        min_samples_leaf=2,     # avoid overfitting on tiny leaf nodes
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test_labels, y_pred_labels)

    print(f"\n{'='*52}")
    print(f"  ML Model Training Results")
    print(f"{'='*52}")
    print(f"  Test Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix (rows=true, cols=pred):")
    print(f"  Classes: {le.classes_.tolist()}")
    print(cm)
    print(f"{'='*52}\n")

    # Top-10 feature importances
    importances = clf.feature_importances_
    top10 = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 most important features:")
    for name, imp in top10:
        print(f"  {name:<32} {imp:.4f}")

    return {
        "model": clf,
        "label_encoder": le,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist(),
        "class_names": le.classes_.tolist(),
        "feature_importance": {n: float(v) for n, v in zip(FEATURE_NAMES, importances)},
        "report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_model(model_data: dict, model_path: str, dataset: pd.DataFrame, dataset_path: str):
    """Persist trained model, dataset CSV, and metrics JSON to disk."""
    import joblib, json

    joblib.dump(
        {"model": model_data["model"], "label_encoder": model_data["label_encoder"]},
        model_path
    )
    print(f"Model saved    : {model_path}")

    dataset.to_csv(dataset_path, index=False)
    print(f"Dataset saved  : {dataset_path}")

    metrics_path = model_path.replace(".joblib", "_metrics.json")
    metrics = {k: v for k, v in model_data.items() if k not in ("model", "label_encoder")}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved  : {metrics_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    model_path   = os.path.join(data_dir, "cipher_model.joblib")
    dataset_path = os.path.join(data_dir, "sample_dataset.csv")

    print("AI-Assisted Cryptanalysis -- ML Training")
    print("=" * 52)

    df         = generate_dataset(SAMPLES_PER_CLASS)
    model_data = train_model(df)
    save_model(model_data, model_path, df, dataset_path)

    print("\nTraining complete. Run: streamlit run app.py")
