#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

# TODO make this more generic to fit any purpose
def convert_py_to_ipynb(py_file):
    py_path = Path(py_file)
    ipynb_file = py_path.with_suffix('.ipynb')
    
    print(f"🔄 {py_path.name} → {ipynb_file.name}")
    cmd = f"poetry run jupytext --from py:percent --to ipynb {py_path}"
    subprocess.run(cmd, shell=True, check=True)
    print("✅ OK")
    return ipynb_file

def convert_to_html(ipynb_file):
    ipynb_path = Path(ipynb_file)
    html_file = ipynb_path.with_suffix('.html')
    
    print(f"🔄 {ipynb_path.name} → {html_file.name}")
    cmd = f"poetry run jupyter nbconvert --to html --execute --no-input {ipynb_path}"
    subprocess.run(cmd, shell=True, check=True)
    print("✅ OK")

file_paths = [
    "src/value_iteration.py",
    "src/policy_iteration.py",
]

if __name__ == "__main__":
    for py_file in file_paths:
        print(f"\n📁 {py_file}")
        ipynb_file = convert_py_to_ipynb(py_file)
        convert_to_html(ipynb_file)