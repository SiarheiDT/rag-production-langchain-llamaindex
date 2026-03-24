"""
Example 6: Show how to switch between local and cloud Deep Lake paths.

This helper script validates and prints the selected dataset path strategy.
It is useful before running the full indexing scripts.

Usage:
    python m_04_006_local_vs_cloud_dataset_path.py --dataset_path ./repository_db
    python m_04_006_local_vs_cloud_dataset_path.py --dataset_path hub://YOUR_ORG/repository_vector_store
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Deep Lake dataset path.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Local path or hub:// path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path

    print(f"dataset_path={dataset_path}")

    if dataset_path.startswith("hub://"):
        print("Mode: cloud / managed Deep Lake")
        print("Requirements: ACTIVELOOP_TOKEN and correct org path")
    else:
        print("Mode: local Deep Lake")
        print("A local directory will be created and used as the vector DB.")


if __name__ == "__main__":
    main()
