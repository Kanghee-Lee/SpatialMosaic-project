import json
import argparse


def load_results(json_path):
    """Load JSON results and return a dict mapping doc_id -> accuracy"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = {}
    for log in data.get('logs', []):
        doc_id = log.get('doc_id')
        doc = log.get('doc', {})
        accuracy = doc.get('accuracy', 0.0)
        results[doc_id] = {
            'accuracy': accuracy,
            'prediction': doc.get('prediction'),
            'mc_answer': doc.get('mc_answer'),
            'question': doc.get('question'),
            'scene_name': doc.get('scene_name'),
        }
    return results


def compare_results(spatialmosaic_path, vggt_path):
    """Compare SpatialMosaic and VGGT results and categorize by correctness"""
    spatialmosaic_results = load_results(spatialmosaic_path)
    vggt_results = load_results(vggt_path)

    spatialmosaic_correct_vggt_wrong = []
    vggt_correct_spatialmosaic_wrong = []
    both_wrong = []

    for doc_id, spatialmosaic_data in spatialmosaic_results.items():
        if doc_id in vggt_results:
            vggt_data = vggt_results[doc_id]

            entry = {
                'id': doc_id,
                'question': spatialmosaic_data['question'],
                'scene_name': spatialmosaic_data['scene_name'],
                'ground_truth': spatialmosaic_data['mc_answer'],
                'spatialmosaic_pred': spatialmosaic_data['prediction'],
                'vggt_pred': vggt_data['prediction'],
            }

            # spatialmosaic correct (accuracy == 1.0) and vggt wrong (accuracy == 0.0)
            if spatialmosaic_data['accuracy'] == 1.0 and vggt_data['accuracy'] == 0.0:
                spatialmosaic_correct_vggt_wrong.append(entry)

            # vggt correct (accuracy == 1.0) and spatialmosaic wrong (accuracy == 0.0)
            if vggt_data['accuracy'] == 1.0 and spatialmosaic_data['accuracy'] == 0.0:
                vggt_correct_spatialmosaic_wrong.append(entry)

            # both wrong (accuracy == 0.0)
            if spatialmosaic_data['accuracy'] == 0.0 and vggt_data['accuracy'] == 0.0:
                both_wrong.append(entry)

    return spatialmosaic_correct_vggt_wrong, vggt_correct_spatialmosaic_wrong, both_wrong


def print_results(results, title):
    """Print results with title"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Found {len(results)} samples\n")

    ids = [r['id'] for r in results]
    print("IDs:", ids)

    print("\n--- Details ---")
    for r in results[:10]:  # Show first 10
        print(f"ID: {r['id']}, Scene: {r['scene_name']}")
        print(f"  GT: {r['ground_truth']}, SpatialMosaic: {r['spatialmosaic_pred']}, VGGT: {r['vggt_pred']}")

    if len(results) > 10:
        print(f"... and {len(results) - 10} more")

    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare SpatialMosaic and VGGT results')
    parser.add_argument('--spatialmosaic', type=str, required=True, help='Path to SpatialMosaic results JSON')
    parser.add_argument('--vggt', type=str, required=True, help='Path to VGGT results JSON')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path (optional)')
    args = parser.parse_args()

    spatialmosaic_correct_vggt_wrong, vggt_correct_spatialmosaic_wrong, both_wrong = compare_results(args.spatialmosaic, args.vggt)

    # Print SpatialMosaic correct, VGGT wrong
    ids_spatialmosaic_correct = print_results(
        spatialmosaic_correct_vggt_wrong,
        "SpatialMosaic Correct & VGGT Wrong"
    )

    # Print VGGT correct, SpatialMosaic wrong
    ids_vggt_correct = print_results(
        vggt_correct_spatialmosaic_wrong,
        "VGGT Correct & SpatialMosaic Wrong"
    )

    # Print both wrong
    ids_both_wrong = print_results(
        both_wrong,
        "Both Wrong"
    )

    # Save to file if output path provided
    if args.output:
        output_data = {
            'spatialmosaic_correct_vggt_wrong': {
                'ids': ids_spatialmosaic_correct,
                'details': spatialmosaic_correct_vggt_wrong
            },
            'vggt_correct_spatialmosaic_wrong': {
                'ids': ids_vggt_correct,
                'details': vggt_correct_spatialmosaic_wrong
            },
            'both_wrong': {
                'ids': ids_both_wrong,
                'details': both_wrong
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")
