import json
import argparse


def count_correct_by_scene(json_path, scene_name):
    """Count correct answers for a specific scene"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    total = 0
    correct = 0

    for log in data.get('logs', []):
        doc = log.get('doc', {})
        if doc.get('scene_name') == scene_name:
            total += 1
            if doc.get('accuracy', 0.0) == 1.0:
                correct += 1

    return correct, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count correct answers by scene')
    parser.add_argument('--spatialmosaic', type=str, required=True, help='Path to SpatialMosaic results JSON')
    parser.add_argument('--vggt', type=str, required=True, help='Path to VGGT results JSON')
    parser.add_argument('--scene', type=str, required=True, help='Scene name to filter')
    args = parser.parse_args()

    spatialmosaic_correct, spatialmosaic_total = count_correct_by_scene(args.spatialmosaic, args.scene)
    vggt_correct, vggt_total = count_correct_by_scene(args.vggt, args.scene)

    print(f"Scene: {args.scene}")
    print(f"SpatialMosaic: {spatialmosaic_correct}/{spatialmosaic_total} correct")
    print(f"VGGT:  {vggt_correct}/{vggt_total} correct")
