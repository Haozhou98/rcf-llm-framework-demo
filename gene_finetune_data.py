import json
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict


def load_data(file_path):
    """Loads data from a JSONL file and separates by label."""
    label_0_records = []
    label_1_records = []

    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            if "healthy" in record["answer"]:
                label_0_records.append(record)
            elif "faulty" in record["answer"]:
                label_1_records.append(record)

    return label_0_records, label_1_records


def undersample_records(label_0_records, label_1_records, label_0_ratio=1.0):
    """Undersample the 'healthy' records to balance the dataset."""
    if label_0_ratio < 1.0:
        label_0_count = int(len(label_1_records) * label_0_ratio)
    else:
        label_0_count = len(label_1_records)

    label_0_records = random.sample(label_0_records, label_0_count)
    return label_0_records


def create_stratified_folds(label_0_records, label_1_records, n_splits=10, seed=42, train_size=0.8):
    """Create stratified cross-validation splits with a customizable train-test ratio."""
    random.seed(seed)

    # Combine records and create labels for stratification
    all_records = label_0_records + label_1_records
    all_labels = ['healthy'] * len(label_0_records) + ['faulty'] * len(label_1_records)

    # Shuffle the records
    combined = list(zip(all_records, all_labels))
    random.shuffle(combined)
    all_records, all_labels = zip(*combined)

    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = defaultdict(lambda: {'train': [], 'val': [], 'test': []})

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_records, all_labels)):
        # Adjust train-test split according to train_size
        train_val_records = [all_records[i] for i in train_val_idx]
        train_val_labels = [all_labels[i] for i in train_val_idx]

        train_records, test_records = train_test_split(
            train_val_records, test_size=(1 - train_size), stratify=train_val_labels, random_state=seed
        )

        # Further split train into training and validation sets
        train_labels = [record['answer'] for record in train_records]
        train_records_split, val_records_split = train_test_split(
            train_records, test_size=0.2, stratify=train_labels, random_state=seed
        )

        # Store in the fold dictionary
        folds[fold_idx]['train'] = train_records_split
        folds[fold_idx]['val'] = val_records_split
        folds[fold_idx]['test'] = test_records

    return folds


def generate_data_folds(file_path, label_0_ratio=1.0, n_splits=10, train_size=0.8, seed=42):
    """Generate stratified data folds for cross-validation."""
    # Load data
    label_0_records, label_1_records = load_data(file_path)

    # Undersample the "healthy" records
    label_0_records = undersample_records(label_0_records, label_1_records, label_0_ratio=label_0_ratio)

    # Generate ten-fold splits with customizable train-test ratio
    folds = create_stratified_folds(label_0_records, label_1_records, n_splits=n_splits, seed=seed, train_size=train_size)

    return folds


if __name__ == "__main__":
    # Example usage
    file_path = '7_LT_60_SL.jsonl'  # Replace with your actual file path
    train_size = 0.75  # Adjust the train-test split ratio as needed
    folds = generate_data_folds(file_path, label_0_ratio=1.0, n_splits=10, train_size=train_size, seed=33)

    # Display fold statistics for verification
    for fold_idx, split_data in folds.items():
        print(f"Fold {fold_idx + 1}:")
        for split_name, records in split_data.items():
            healthy_count = sum(1 for record in records if record["answer"] == "healthy")
            faulty_count = sum(1 for record in records if record["answer"] == "faulty")
            print(f"  {split_name} - healthy: {healthy_count}, faulty: {faulty_count}")
