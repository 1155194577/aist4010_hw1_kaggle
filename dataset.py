import kagglehub
from pathlib import Path
path = kagglehub.competition_download('aist-4010-fall-2026-a-1')

print("Path to competition files:", path)
