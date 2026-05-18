"""Command line interface for evalsyste."""

from __future__ import annotations

import argparse
import json

from evalsyste.cases import load_builtin_cases
from evalsyste.runner import EvalRunner


