# run.py usage

List tasks:
    python run.py --list

Run CSV loader:
    python run.py --module 1 --part 1 --task 1 -- --csv_path ./data/raw/data.csv

Run Wikipedia example:
    python run.py --module 1 --part 1 --task 2 -- --query "Machine_learning"

Run ChatOpenAI example:
    python run.py --module 1 --part 1 --task 3

Run embeddings example:
    python run.py --module 1 --part 1 --task 4

Run LLMChain example:
    python run.py --module 1 --part 1 --task 5

Run LCEL example:
    python run.py --module 1 --part 1 --task 6

Run sequential chain example:
    python run.py --module 1 --part 1 --task 7 -- --theme "Having a black friday sale with 50% off on everything."
