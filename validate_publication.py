#!/usr/bin/env python3
import json

from harbin.reporting import validate_publication_files


report = validate_publication_files()
print(json.dumps(report, indent=2))
if report.get("status") == "FAIL":
    raise SystemExit(1)
