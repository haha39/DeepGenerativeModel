import subprocess
import time
import os
import re
from datetime import datetime

MODELS = [
    ("llama3.2:1b-instruct", ["q4_K_M", "q8_0", "fp16"]),
    ("llama3.2:3b-instruct", ["q4_K_M", "q8_0", "fp16"]),
]

PROMPT = "What is federated learning?"
DEVICE_OPTIONS = ["gpu", "cpu"]
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SPINNER_CHARS = r"[⠙⠸⠼⠴⠦⠧⠇⠏⠹]+"  # braille spinner pattern


def remove_ansi_escape(s):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    s = ansi_escape.sub('', s)
    s = re.sub(SPINNER_CHARS, '', s)
    return s.strip()


def run_and_measure(model: str, quant: str, device: str):
    full_model_name = f"{model}-{quant}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{OUTPUT_DIR}/{model.replace(':', '_')}_{quant}_{device}_{timestamp}.txt"

    print(f"\nRunning {full_model_name} on {device}...")

    command = ["ollama", "run", full_model_name]
    env = os.environ.copy()
    env["OLLAMA_DEFAULT_DEVICE"] = device  # cpu 或 gpu

    try:
        start_time = time.time()
        result = subprocess.run(command, input=PROMPT.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, env=env)
        end_time = time.time()
    except subprocess.TimeoutExpired:
        print("Timeout!")
        return

    output = result.stdout.decode(errors="ignore") + result.stderr.decode(errors="ignore")
    output = remove_ansi_escape(output)

    duration = end_time - start_time
    token_count = len(output.split())
    tokens_per_second = token_count / duration if duration > 0 else 0

    print(f"Done: {token_count} tokens in {duration:.2f}s ({tokens_per_second:.2f} tokens/s)")

    with open(result_file, "w") as f:
        f.write(f"Model: {full_model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Duration: {duration:.2f} s\n")
        f.write(f"Token count: {token_count}\n")
        f.write(f"Tokens/sec: {tokens_per_second:.2f}\n")
        f.write(f"\n--- Output ---\n{output}\n")

    print(f"📁 Output saved to: {result_file}")



if __name__ == "__main__":
    for model, quants in MODELS:
        for quant in quants:
            for device in DEVICE_OPTIONS:
                run_and_measure(model, quant, device)
