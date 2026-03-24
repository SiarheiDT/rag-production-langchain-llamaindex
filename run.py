"""
Unified CLI runner for the RAG course repository.

Purpose:
    - Run any example script by module / part / task number.
    - Forward all script-specific CLI arguments after `--` to the target script.

Examples:
    python run.py --list

    python run.py --module 1 --part 1 --task 1 -- --csv_path ./data/raw/data.csv
    python run.py --module 1 --part 1 --task 2 -- --query "Machine_learning"
    python run.py --module 1 --part 1 --task 3
    python run.py --module 1 --part 1 --task 7 -- --theme "Having a black friday sale with 50% off on everything."

Notes:
    - Arguments before `--` belong to the runner.
    - Arguments after `--` are passed unchanged to the selected script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import os


PROJECT_ROOT = Path(__file__).resolve().parent


def build_registry() -> Dict[Tuple[int, int, int], Path]:
    """Register runnable scripts from the repository.

    Key format:
        (module_number, part_number, task_number)

    Current scope:
        LC_01_RAG_Foundations / Part_1_LangChain_Recap
    """
    registry: Dict[Tuple[int, int, int], Path] = {}

    registry[(1, 1, 1)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_001_csv_loader.py"
    registry[(1, 1, 2)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_002_wikipedia_loader.py"
    registry[(1, 1, 3)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_003_chat_openai_basic.py"
    registry[(1, 1, 4)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_004_openai_embeddings.py"
    registry[(1, 1, 5)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_005_llm_chain_rainbow.py"
    registry[(1, 1, 6)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_006_lcel_rainbow.py"
    registry[(1, 1, 7)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_007_sequential_chain_social_post_review.py"
    registry[(1, 1, 8)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_008_chunking.py"
    registry[(1, 1, 9)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_1" / "m_01_009_embeddings_from_chunks.py"
    # LC_01_RAG_Foundations — Part 2 (LlamaIndex Introduction)
    registry[(1, 2, 1)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_001_wikipedia_reader.py"
    registry[(1, 2, 2)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_002_create_nodes.py"
    registry[(1, 2, 3)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_003_vector_index_query.py"
    registry[(1, 2, 4)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_004_deeplake_vector_store.py"
    registry[(1, 2, 5)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_005_persist_index_local.py"
    registry[(1, 2, 6)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_006_load_or_create_index.py"
    registry[(1, 2, 7)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_007_chunk_size_experiment.py"
    registry[(1, 2, 8)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_2" / "m_02_008_metadata_demo.py"
    # LC_01_RAG_Foundations — Part 4 (Chat with Your Code: GitHub + Deep Lake)
    registry[(1, 4, 1)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_4" / "m_04_001_github_quickstart.py"
    registry[(1, 4, 2)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_4" / "m_04_002_github_index_once.py"
    registry[(1, 4, 3)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_4" / "m_04_003_retriever_topk_demo.py"
    registry[(1, 4, 4)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_4" / "m_04_004_custom_query_engine.py"
    registry[(1, 4, 5)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_4" / "m_04_005_response_modes_demo.py"
    registry[(1, 4, 6)] = PROJECT_ROOT / "LC_01_RAG_Foundations" / "Part_4" / "m_04_006_local_vs_cloud_dataset_path.py"    
    return registry


def parse_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    """Parse runner arguments and preserve script arguments after `--`."""
    parser = argparse.ArgumentParser(
        description="Run course examples by module / part / task."
    )
    parser.add_argument("--list", action="store_true", help="List all registered tasks.")
    parser.add_argument("--module", type=int, help="Module number, e.g. 1")
    parser.add_argument("--part", type=int, help="Part number, e.g. 1")
    parser.add_argument("--task", type=int, help="Task number, e.g. 2")

    if "--" in argv:
        idx = argv.index("--")
        runner_argv = argv[:idx]
        script_argv = argv[idx + 1 :]
    else:
        runner_argv = argv
        script_argv = []

    args = parser.parse_args(runner_argv)
    return args, script_argv


def print_registry(registry: Dict[Tuple[int, int, int], Path]) -> None:
    """Print all registered tasks."""
    print("Available tasks:\n")
    for key in sorted(registry):
        module_num, part_num, task_num = key
        rel_path = registry[key].relative_to(PROJECT_ROOT)
        print(f"{module_num}.{part_num}.{task_num} -> {rel_path}")


def validate_selection(args: argparse.Namespace) -> None:
    """Ensure required selection args are present when not using --list."""
    missing = [
        name for name in ("module", "part", "task")
        if getattr(args, name) is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required arguments: {joined}. "
            f"Use --list or provide --module --part --task."
        )


def run_script(script_path: Path, script_args: List[str]) -> int:
    """Execute the selected script as a subprocess."""
    command = [sys.executable, str(script_path), *script_args]

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)

    print(f"Running: {' '.join(command)}\\n")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def main(argv: List[str] | None = None) -> int:
    """Entry point."""
    argv = argv if argv is not None else sys.argv[1:]
    args, script_args = parse_args(argv)
    registry = build_registry()

    if args.list:
        print_registry(registry)
        return 0

    validate_selection(args)

    key = (args.module, args.part, args.task)
    script_path = registry.get(key)

    if script_path is None:
        print(f"Task {args.module}.{args.part}.{args.task} is not registered.")
        print("Use --list to view available tasks.")
        return 1

    if not script_path.exists():
        print(f"Registered file does not exist: {script_path}")
        return 1

    return run_script(script_path, script_args)


if __name__ == "__main__":
    raise SystemExit(main())
