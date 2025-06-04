import argparse
import json
from pathlib import Path
import importlib.util


def load_data(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', required=True, help='Model name')
    opts = parser.parse_args(args)

    dataset_dir = Path(opts.dataset)
    train_data = load_data(dataset_dir / 'train_data' / 'train_set.json')
    dev_data = load_data(dataset_dir / 'dev_data' / 'dev_set.json')

    # Simple majority label model
    label_counts = {}
    for inst in train_data:
        label = inst['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    majority_label = max(label_counts, key=label_counts.get)

    # Evaluate on dev set
    spec = importlib.util.spec_from_file_location('score_function', dataset_dir / 'score_function.py')
    score_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_module)
    predictions = [majority_label for _ in dev_data]
    gold = [inst['label'] for inst in dev_data]
    score = score_module.score(predictions, gold)

    results_dir = Path('models') / opts.model / f'results_{dataset_dir.name}'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'model_weights', 'w') as fw:
        fw.write(majority_label)
    with open(results_dir / 'best_parameters', 'w') as fw:
        fw.write('model=majority_label\n')
        fw.write(f'score={score}\n')


if __name__ == '__main__':
    main()
