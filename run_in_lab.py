#!/usr/bin/env python3
import sys, json, time, requests, uuid, datetime
from urllib.parse import urljoin, urlencode
from websocket import create_connection

if len(sys.argv) < 3:
    print("Usage: python run_in_lab.py <SERVER_URL> <TOKEN> <NOTEBOOK_PATH> <SCRIPT>")
    print("Example:")
    print("  python run_in_lab.py http://localhost:8888 abcd1234 my_notebook.ipynb hello.py")
    sys.exit(1)

server, token, notebook_path, script_path = sys.argv[1:5]

# --- find the right kernel via /api/sessions ---
url = urljoin(server, "/api/sessions")
sessions = requests.get(url, params={"token": token}).json()
session = next((s for s in sessions if s["notebook"]["path"] == notebook_path), None)

if not session:
    print(f"Notebook '{notebook_path}' not found among active sessions.")
    print("Active notebook paths:")
    for s in sessions:
        print(" -", s["notebook"]["path"])
    sys.exit(1)

kernel_id = session["kernel"]["id"]
print(f"✅ Found kernel {kernel_id} for notebook '{notebook_path}'")

# --- read code ---
with open(script_path) as f:
    code = f.read()

# --- connect to kernel via websocket ---
ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
ws = create_connection(f"{ws_url}/api/kernels/{kernel_id}/channels?token={token}")

# --- build and send execute_request ---
msg_id = str(uuid.uuid4())
header = {
    "msg_id": msg_id,
    "username": "terminal",
    "session": str(uuid.uuid4()),
    "date": datetime.datetime.utcnow().isoformat() + "Z",
    "msg_type": "execute_request",
    "version": "5.3"
}
content = {
    "code": code,
    "silent": False,
    "store_history": True,
    "user_expressions": {},
    "allow_stdin": False
}
msg = {"header": header, "parent_header": {}, "metadata": {}, "content": content}
ws.send(json.dumps(msg))
print("📤 Sent code to JupyterLab kernel!")

# --- optional: also print output to terminal ---
try:
    while True:
        msg = json.loads(ws.recv())
        msg_type = msg["header"].get("msg_type", "")
        c = msg.get("content", {})
        if msg_type == "stream":
            print(c.get("text", ""), end="")
        elif msg_type == "error":
            print("\n".join(c.get("traceback", [])))
        elif msg_type == "status" and c.get("execution_state") == "idle":
            break
finally:
    ws.close()

print("✅ Done. You should see the output appear in JupyterLab.")
