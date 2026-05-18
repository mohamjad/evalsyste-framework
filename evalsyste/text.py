"""Small text utilities used by deterministic eval scorers."""

from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

