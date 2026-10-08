#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""A minimal QMP client for the migration scripts.

usage: qmp.py SOCKET CMD [CMD ...]

Each CMD is a JSON command, whose reply is printed, or one of:
  wait:STATUS  poll query-migrate until its status is STATUS, failed or cancelled
  sleep:N      sleep N seconds
  hmp:LINE     run LINE with human-monitor-command and print what it printed
  job:ID       poll query-jobs until job ID is concluded and print it
Events that arrive in between are printed with an EVENT prefix.
"""
import json
import socket
import sys
import time


def connect(path):
    for _ in range(200):
        try:
            s = socket.socket(socket.AF_UNIX)
            s.connect(path)
            return s
        except OSError:
            time.sleep(0.05)
    sys.exit("cannot connect to " + path)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    f = connect(sys.argv[1]).makefile("rw")
    json.loads(f.readline())

    def cmd(c):
        f.write(c + "\n")
        f.flush()
        while True:
            r = json.loads(f.readline())
            if "event" in r:
                print("EVENT", json.dumps(r))
                continue
            return r

    cmd('{"execute":"qmp_capabilities"}')
    for c in sys.argv[2:]:
        if c.startswith("wait:"):
            want = c[5:]
            r = None
            for _ in range(6000):
                r = cmd('{"execute":"query-migrate"}')
                if r.get("return", {}).get("status") in (want, "failed", "cancelled"):
                    break
                time.sleep(0.05)
            print(json.dumps(r))
        elif c.startswith("sleep:"):
            time.sleep(float(c[6:]))
        elif c.startswith("hmp:"):
            r = cmd(json.dumps({"execute": "human-monitor-command",
                                "arguments": {"command-line": c[4:]}}))
            print("(hmp) " + c[4:])
            if "return" in r:
                sys.stdout.write(r["return"])
            else:
                print(json.dumps(r))
        elif c.startswith("job:"):
            job = None
            for _ in range(6000):
                jobs = cmd('{"execute":"query-jobs"}').get("return", [])
                job = next((j for j in jobs if j.get("id") == c[4:]), None)
                if job is None or job.get("status") == "concluded":
                    break
                time.sleep(0.05)
            print(json.dumps(job))
        else:
            print(json.dumps(cmd(c)))


main()
