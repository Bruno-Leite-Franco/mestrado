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
    results_dir = Path('models') / opts.model / f'results_{dataset_dir.name}'
    majority_label = ''
    weights_file = results_dir / 'model_weights'
    if weights_file.exists():
        majority_label = weights_file.read_text().strip()

    spec = importlib.util.spec_from_file_location('score_function', dataset_dir / 'score_function.py')
    score_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_module)

    test_dir = dataset_dir / 'test_data'
    for test_path in sorted(test_dir.glob('*.json')):
        data = load_data(test_path)
        predictions = [majority_label for _ in data]
        gold = [inst['label'] for inst in data]
        instance_scores = [score_module.score(p, g) for p, g in zip(predictions, gold)]
        score = sum(instance_scores) / len(instance_scores) if instance_scores else 0.0
        out_file = results_dir / f'answers_{test_path.stem}.tsv'
        with open(out_file, 'w') as fw:
            fw.write('input\tprediction\tgold\tscore\n')
            for inst, pred, inst_score in zip(data, predictions, instance_scores):
                fw.write(f"{inst['text']}\t{pred}\t{inst['label']}\t{inst_score}\n")
            fw.write(f"# score={score}\n")
        print(f"{test_path.stem} score: {score}")


if __name__ == '__main__':
    main()
