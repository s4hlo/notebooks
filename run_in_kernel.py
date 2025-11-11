#!/usr/bin/env python
import sys, json, time, os, glob
from jupyter_client import BlockingKernelClient
from jupyter_core.paths import jupyter_runtime_dir

def find_latest_kernel():
    rt = jupyter_runtime_dir()
    kernels = sorted(
        glob.glob(os.path.join(rt, "kernel-*.json")), key=os.path.getmtime, reverse=True
    )
    return kernels[0] if kernels else None

# --- Parse args ---
if len(sys.argv) == 3:
    kernel_file, script_file = sys.argv[1], sys.argv[2]
elif len(sys.argv) == 2:
    kernel_file = find_latest_kernel()
    script_file = sys.argv[1]
    if not kernel_file:
        print("❌ No running Jupyter kernels found.")
        sys.exit(1)
    print(f"🔗 Using latest kernel: {kernel_file}")
else:
    print("Usage:\n  run_in_kernel.py [kernel.json] <script.py>")
    sys.exit(1)
# --- Read files ---
with open(kernel_file) as f:
    conn_info = json.load(f)
with open(script_file) as f:
    code = f.read()

kc = BlockingKernelClient()
kc.load_connection_info(conn_info)
kc.start_channels()

msg_id = kc.execute(code)

while True:
    try:
        msg = kc.get_iopub_msg(timeout=1)
    except Exception:
        break
    msg_type = msg["header"]["msg_type"]
    content = msg["content"]
    if msg_type == "stream":
        print(content.get("text", ""), end="")
    elif msg_type == "execute_result":
        print(content["data"].get("text/plain", ""))
    elif msg_type == "error":
        print("\n".join(content["traceback"]))
    elif msg_type == "status" and content.get("execution_state") == "idle":
        break

kc.stop_channels()
