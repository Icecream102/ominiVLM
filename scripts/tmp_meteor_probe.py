"""Probe the raw output format of meteor-1.5.jar under the current JVM."""

import os
import subprocess

jar_dir = "/root/miniconda3/lib/python3.12/site-packages/pycocoevalcap/meteor"
cmd = ["java", "-jar", "-Xmx2G", "meteor-1.5.jar", "-", "-", "-stdio", "-l", "en", "-norm"]
proc = subprocess.Popen(
    cmd, cwd=jar_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)


def send(line):
    proc.stdin.write((line + "\n").encode())
    proc.stdin.flush()
    print(repr(line), "->", flush=True)


send("SCORE ||| a cat sits on the mat ||| a cat on a mat")
send("SCORE ||| a dog runs in the park ||| a dog running in the park")
send("EVAL ||| {} ||| {}".format(
    proc.stdout.readline().decode().strip(),
    proc.stdout.readline().decode().strip(),
))
proc.stdin.flush()

for _ in range(6):
    raw = proc.stdout.readline()
    print("RAW:", repr(raw), flush=True)
    if not raw:
        break

proc.kill()
