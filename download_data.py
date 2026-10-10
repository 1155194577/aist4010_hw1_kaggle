import kagglehub
from pathlib import Path
from shutil import copy2, copytree


def main():
    source = Path(kagglehub.competition_download("aist-4010-fall-2026-a-1"))
    destination = Path(__file__).resolve().parent / "data"
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Path to competition files: {source}")
    if not source.exists():
        raise FileNotFoundError(f"KaggleHub returned a path that does not exist: {source}")

    if source.is_dir():
        children = list(source.iterdir())
        if len(children) == 1 and children[0].is_dir():
            source = children[0]

    copied_files = 0
    if source.is_file():
        copy2(source, destination / source.name)
        copied_files = 1
    else:
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                copytree(item, target, dirs_exist_ok=True)
                copied_files += sum(1 for path in item.rglob("*") if path.is_file())
            else:
                copy2(item, target)
                copied_files += 1

    if not copied_files:
        raise RuntimeError(f"No files were found in the downloaded path: {source}")
    print(f"Copied {copied_files} file(s) to {destination}")

if __name__ == "__main__":
    main()