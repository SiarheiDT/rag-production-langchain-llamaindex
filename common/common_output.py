import os

def save_result(script_file: str, text: str) -> str:
    os.makedirs("output", exist_ok=True)
    filename = os.path.basename(script_file).replace(".py", ".txt")
    path = os.path.join("output", f"result_{filename}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return path