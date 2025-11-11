#!/usr/bin/env lua
-- Usage:
--   ./jrun.lua /path/to/kernel-123.json hello.py
-- Requires: poetry (so we run "poetry run python"), and the poetry env must have ipykernel / jupyter-client installed.

local kernel = arg[1]
local script = arg[2]

if not kernel or not script then
  io.stderr:write("Usage: jrun.lua /path/to/kernel.json script.py\n")
  os.exit(2)
end

-- basic existence checks
local function exists(path)
  local f = io.open(path, "r")
  if f then f:close(); return true end
  return false
end

if not exists(kernel) then
  io.stderr:write("Kernel file not found: " .. kernel .. "\n")
  os.exit(3)
end
if not exists(script) then
  io.stderr:write("Script file not found: " .. script .. "\n")
  os.exit(4)
end

-- Python helper that will be fed via heredoc. It reads two argv (kernel_file, script_file),
-- connects to the kernel using jupyter_client and executes the code, streaming outputs.
local py = [[
import sys, time
try:
    from jupyter_client import BlockingKernelClient
except Exception as e:
    sys.stderr.write("Missing dependency: jupyter-client (install in your poetry env).\\n" + str(e) + "\\n")
    sys.exit(5)

if len(sys.argv) < 3:
    sys.stderr.write("Expecting: python - kernel.json script.py\\n")
    sys.exit(2)

kernel_file = sys.argv[1]
script_path = sys.argv[2]

with open(script_path, 'r', encoding='utf-8') as f:
    code = f.read()

kc = BlockingKernelClient()
kc.load_connection_file(kernel_file)
kc.start_channels()
try:
    msg_id = kc.execute(code)
    # collect and stream outputs
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=1)
        except Exception:
            break
        mtype = msg.get('msg_type')
        content = msg.get('content', {})
        if mtype == 'stream':
            sys.stdout.write(content.get('text', ''))
            sys.stdout.flush()
        elif mtype in ('execute_result', 'display_data'):
            data = content.get('data', {})
            text = data.get('text/plain')
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
        elif mtype == 'error':
            tb = '\\n'.join(content.get('traceback', []))
            sys.stderr.write(tb + '\\n')
    # wait a short while for final execute_reply on shell channel (best-effort)
    try:
        kc.get_shell_msg(timeout=2)
    except Exception:
        pass
finally:
    try:
        kc.stop_channels()
    except Exception:
        pass
]]

-- build shell command: run the python helper under poetry, pass kernel and script as argv,
-- feed the helper via heredoc, and redirect stderr to stdout so we capture both streams.
local cmd = string.format(
  "poetry run python - %q %q <<'PY'\n%s\nPY 2>&1",
  kernel, script, py
)

-- open a pipe and stream output to this terminal in real time
local p = io.popen(cmd, "r")
if not p then
  io.stderr:write("Failed to run command\n")
  os.exit(10)
end

for line in p:lines() do
  io.write(line .. "\n")
end

local ok, _, code = p:close()
if code and code ~= 0 then
  io.stderr:write(string.format("Helper exited with code %d\n", code))
  os.exit(code)
end
]]

-- Make file executable:
--   chmod +x jrun.lua
-- Example:
--   ./jrun.lua $(poetry run python -c "from jupyter_core.paths import jupyter_runtime_dir; import os; print(os.path.join(os.getcwd(),'doesnotmatter'))") ????
-- Real usage:
--   ./jrun.lua /home/sleig/.local/share/jupyter/runtime/kernel-816202.json hello.py

print("Done.")
