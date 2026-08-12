#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
#
# Hardened for modern JVMs and noisy model outputs:
#   - strips newlines / protocol separators from hypotheses and references so
#     the line-oriented jar protocol never misaligns
#   - tolerant numeric parsing on the EVAL response

import os
import subprocess
import threading

METEOR_JAR = 'meteor-1.5.jar'


class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        self.lock = threading.Lock()

    @staticmethod
    def _sanitize(text):
        text = (text or "").replace("\r", " ").replace("\n", " ")
        text = text.replace("|||", " ").replace("  ", " ")
        return text.strip() or "."

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        try:
            for i in imgIds:
                assert(len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
            self.meteor_p.stdin.flush()
            for i in range(0, len(imgIds)):
                raw = self.meteor_p.stdout.readline()
                parts = raw.strip().split()
                # Modern JVMs can occasionally concatenate buffered values;
                # keep reading until we have exactly one numeric token.
                while len(parts) != 1:
                    parts += self.meteor_p.stdout.readline().strip().split()
                scores.append(float(parts[0]))
            raw = self.meteor_p.stdout.readline()
            parts = raw.strip().split()
            while len(parts) != 1:
                parts += self.meteor_p.stdout.readline().strip().split()
            score = float(parts[0])
        finally:
            self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = self._sanitize(hypothesis_str)
        reference_list = [self._sanitize(reference) for reference in reference_list]
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def __del__(self):
        try:
            self.lock.acquire()
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
        except Exception:
            pass
        finally:
            self.lock.release()
