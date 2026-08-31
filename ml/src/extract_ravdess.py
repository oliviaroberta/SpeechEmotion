from __future__ import annotations

import argparse
import hashlib
import ntpath
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_MD5 = "bc696df654c87fed845eb13823edef8a"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(name: str) -> PurePosixPath:
    if not name:
        raise ValueError("ZIP entry has an empty name.")
    if name.startswith(("/", "\\")) or ntpath.splitdrive(name)[0]:
        raise ValueError(f"Unsafe absolute or drive-qualified ZIP path: {name!r}")

    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP path traversal: {name!r}")
    return path


def is_symbolic_link(entry: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(entry.external_attr >> 16)


def validate_archive(archive_path: Path) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")

    archive_md5 = calculate_md5(archive_path)
    print(f"Archive MD5: {archive_md5}")
    if archive_md5.lower() != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {EXPECTED_MD5}, got {archive_md5}")

    with zipfile.ZipFile(archive_path) as archive:
        failed_member = archive.testzip()
        if failed_member is not None:
            raise RuntimeError(f"ZIP CRC/integrity validation failed for: {failed_member}")

        members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for entry in archive.infolist():
            if entry.flag_bits & 0x1:
                raise RuntimeError(f"Encrypted ZIP entry is not allowed: {entry.filename}")
            if is_symbolic_link(entry):
                raise RuntimeError(f"Symbolic-link ZIP entry is not allowed: {entry.filename}")
            members.append((entry, safe_member_path(entry.filename)))

    print("ZIP CRC/integrity validation: passed")
    return members


def resolve_destination(root: Path, member_path: PurePosixPath) -> Path:
    destination = root.joinpath(*member_path.parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise RuntimeError(f"ZIP entry resolves outside extraction directory: {member_path}")
    return destination


def expected_files(members: list[tuple[zipfile.ZipInfo, PurePosixPath]]) -> dict[PurePosixPath, int]:
    return {
        member_path: entry.file_size
        for entry, member_path in members
        if not entry.is_dir() and not entry.filename.endswith("/")
    }


def extraction_is_complete(destination: Path, expected: dict[PurePosixPath, int]) -> bool:
    actual_files = {
        path.relative_to(destination).as_posix(): path.stat().st_size
        for path in destination.rglob("*")
        if path.is_file()
    }
    expected_by_name = {path.as_posix(): size for path, size in expected.items()}
    return actual_files == expected_by_name


def extract_archive(archive_path: Path, destination: Path) -> None:
    members = validate_archive(archive_path)
    expected = expected_files(members)
    temporary_destination = destination.with_name(f"{destination.name}.tmp")

    if temporary_destination.exists():
        raise RuntimeError(
            f"Temporary extraction directory already exists; inspect it manually: {temporary_destination}"
        )
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError(f"Extraction destination exists but is not a directory: {destination}")
        if extraction_is_complete(destination, expected):
            print(f"Existing extraction is complete; no extraction performed: {destination}")
            return
        raise RuntimeError(
            f"Extraction destination exists but is incomplete or invalid; inspect it manually: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination.mkdir()
    temporary_root = temporary_destination.resolve()
    print(f"Extracting {len(expected)} files into temporary directory: {temporary_destination}")

    written_files = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for entry, member_path in members:
                member_destination = resolve_destination(temporary_root, member_path)
                if entry.is_dir() or entry.filename.endswith("/"):
                    member_destination.mkdir(parents=True, exist_ok=True)
                    continue

                member_destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry, "r") as source, member_destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if member_destination.stat().st_size != entry.file_size:
                    raise RuntimeError(f"Extracted size mismatch for: {member_path}")
                written_files += 1
                if written_files % 240 == 0 or written_files == len(expected):
                    print(f"Extracted files: {written_files}/{len(expected)}")

        if written_files != len(expected):
            raise RuntimeError(f"Expected {len(expected)} files, wrote {written_files}")
        if not extraction_is_complete(temporary_destination, expected):
            raise RuntimeError("Temporary extraction does not match the ZIP member list.")
        temporary_destination.rename(destination)
    except Exception:
        print(f"Extraction did not complete; temporary files remain at: {temporary_destination}", file=sys.stderr)
        raise

    print(f"Safe extraction completed successfully: {destination}")


def parse_arguments() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Safely extract the verified RAVDESS speech archive.")
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=root / "ml" / "data" / "raw" / "ravdess" / "archive" / "Audio_Speech_Actors_01-24.zip",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=root / "ml" / "data" / "raw" / "ravdess" / "extracted",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        extract_archive(arguments.archive, arguments.destination)
    except Exception as error:
        print(f"RAVDESS extraction failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
