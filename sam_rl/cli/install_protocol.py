"""Safely install the separately distributed, hash-pinned protocol metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = REPO_ROOT / "artifacts" / "protocol"
MANIFEST_NAME = "SHA256SUMS.json"
INSTALL_MARKER = ".protocol-install.json"
ALLOWED_PREFIXES = ("fss_manifests", "kvasir_manifests", "splits")
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024
MAX_MEMBERS = 10_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProtocolBundleError(ValueError):
    """Raised when a protocol archive is incomplete or unsafe to install."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolBundleError(f"Duplicate key in {MANIFEST_NAME}: {key!r}.")
        result[key] = value
    return result


def _safe_member_name(name: str, *, directory: bool = False) -> str:
    if "\\" in name or "\x00" in name:
        raise ProtocolBundleError(f"Unsafe ZIP path: {name!r}.")
    normalized = name[:-1] if directory and name.endswith("/") else name
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or (len(path.parts) == 1 and path.parts[0] == MANIFEST_NAME and not directory)
    ):
        if normalized == MANIFEST_NAME and not directory:
            return normalized
        raise ProtocolBundleError(f"Unsafe ZIP path: {name!r}.")
    if path.parts[0] not in ALLOWED_PREFIXES:
        raise ProtocolBundleError(f"Unexpected file in protocol bundle: {name!r}.")
    if len(path.parts) < 2 and not directory:
        raise ProtocolBundleError(
            f"Protocol bundle entries must be inside a manifest directory: {name!r}."
        )
    if not directory and path.suffix.lower() != ".json":
        raise ProtocolBundleError(f"Only JSON protocol metadata is permitted: {name!r}.")
    return normalized


def _read_manifest(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> dict[str, str]:
    matches = [info for info in infos if info.filename == MANIFEST_NAME]
    if len(matches) != 1:
        raise ProtocolBundleError(f"Protocol bundle must contain exactly one root {MANIFEST_NAME}.")
    info = matches[0]
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ProtocolBundleError("Protocol checksum manifest exceeds the allowed size.")
    try:
        data = json.loads(
            archive.read(info).decode("utf-8"), object_pairs_hook=_json_without_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ProtocolBundleError(f"Cannot read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ProtocolBundleError(f"{MANIFEST_NAME} must be a non-empty JSON object.")
    result: dict[str, str] = {}
    for name, digest in data.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ProtocolBundleError(
                f"Each {MANIFEST_NAME} entry must map a path to a SHA-256 string."
            )
        normalized = _safe_member_name(name)
        if normalized != name or not SHA256_RE.fullmatch(digest):
            raise ProtocolBundleError(f"Invalid checksum entry for {name!r}.")
        result[name] = digest
    return result


def _checked_members(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    if len(infos) > MAX_MEMBERS:
        raise ProtocolBundleError(f"Protocol bundle contains more than {MAX_MEMBERS} ZIP entries.")
    files: dict[str, zipfile.ZipInfo] = {}
    directories: set[str] = set()
    total_size = 0
    for info in infos:
        is_directory = info.is_dir()
        name = _safe_member_name(info.filename, directory=is_directory)
        if name in files or name in directories:
            raise ProtocolBundleError(f"Duplicate ZIP path: {info.filename!r}.")
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if stat.S_ISLNK(unix_mode):
            raise ProtocolBundleError(f"Symbolic links are not permitted: {info.filename!r}.")
        if file_type not in (0, stat.S_IFDIR if is_directory else stat.S_IFREG):
            raise ProtocolBundleError(f"Unsupported ZIP entry type: {info.filename!r}.")
        if is_directory:
            directories.add(name)
            continue
        if name != MANIFEST_NAME and name not in (MANIFEST_NAME,):
            if info.file_size > MAX_MEMBER_BYTES:
                raise ProtocolBundleError(
                    f"Protocol file exceeds the allowed size: {info.filename!r}."
                )
            total_size += info.file_size
            if total_size > MAX_ARCHIVE_BYTES:
                raise ProtocolBundleError(
                    "Expanded protocol bundle exceeds the allowed total size."
                )
        files[name] = info
    if MANIFEST_NAME not in files:
        raise ProtocolBundleError(f"Protocol bundle is missing root {MANIFEST_NAME}.")
    return files


def _verify_existing(destination: Path, archive_sha256: str, checksums: dict[str, str]) -> None:
    marker = destination / INSTALL_MARKER
    if not destination.is_dir() or not marker.is_file():
        raise ProtocolBundleError(
            f"Destination already exists without a matching installation record: {destination}. "
            "Choose a new destination or remove it explicitly."
        )
    try:
        installed = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolBundleError(f"Existing installation record is unreadable: {marker}.") from exc
    if installed.get("archive_sha256") != archive_sha256 or installed.get("files") != checksums:
        raise ProtocolBundleError(
            f"Destination contains a different protocol bundle: {destination}. "
            "Choose a new destination or remove it explicitly."
        )
    expected = set(checksums) | {INSTALL_MARKER}
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ProtocolBundleError(
            f"Existing installation has missing or unexpected files: {destination}."
        )
    for name, expected_sha in checksums.items():
        if _sha256_file(destination / name) != expected_sha:
            raise ProtocolBundleError(f"Installed protocol file has changed: {destination / name}.")


def install_protocol_bundle(
    archive_path: str | Path,
    expected_archive_sha256: str,
    destination: str | Path = DEFAULT_DESTINATION,
) -> Path:
    """Install a verified JSON-only protocol ZIP without overwriting existing files."""
    archive_path = Path(archive_path).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if not SHA256_RE.fullmatch(expected_archive_sha256):
        raise ProtocolBundleError("--sha256 must be 64 lowercase hexadecimal characters.")
    if not archive_path.is_file():
        raise ProtocolBundleError(
            f"Protocol archive not found: {archive_path}. Download the protocol bundle from the results deposit."
        )
    actual_archive_sha256 = _sha256_file(archive_path)
    if actual_archive_sha256 != expected_archive_sha256:
        raise ProtocolBundleError(
            "Protocol archive SHA-256 mismatch: "
            f"expected {expected_archive_sha256}, found {actual_archive_sha256}."
        )
    if archive_path == destination or destination in archive_path.parents:
        raise ProtocolBundleError(
            "The protocol destination cannot contain or replace the source archive."
        )
    if destination.exists():
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                files = _checked_members(infos)
                checksums = _read_manifest(archive, infos)
                if set(checksums) != set(files) - {MANIFEST_NAME}:
                    raise ProtocolBundleError(
                        f"{MANIFEST_NAME} paths do not exactly match the archive contents."
                    )
            _verify_existing(destination, actual_archive_sha256, checksums)
        except zipfile.BadZipFile as exc:
            raise ProtocolBundleError(f"Not a valid ZIP protocol bundle: {archive_path}.") from exc
        return destination

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / f".{destination.name}.protocol-install.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolBundleError(
            f"Another protocol installation may be running for {destination}; lock exists at {lock}."
        ) from exc

    staging: Path | None = None
    try:
        os.close(lock_fd)
        if destination.exists():
            raise ProtocolBundleError(f"Destination appeared during installation: {destination}.")
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            files = _checked_members(infos)
            checksums = _read_manifest(archive, infos)
            if set(checksums) != set(files) - {MANIFEST_NAME}:
                raise ProtocolBundleError(
                    f"{MANIFEST_NAME} paths do not exactly match the archive contents."
                )
            if not any(
                name.startswith(prefix + "/") for name in checksums for prefix in ALLOWED_PREFIXES
            ):
                raise ProtocolBundleError(
                    "Protocol bundle contains no FSS, Kvasir, or split metadata."
                )

            staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
            for name, expected_sha in checksums.items():
                target = staging.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(files[name], "r") as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                if digest.hexdigest() != expected_sha:
                    raise ProtocolBundleError(f"Checksum mismatch in ZIP member {name!r}.")
            marker_payload = {
                "archive_sha256": actual_archive_sha256,
                "checksum_manifest": MANIFEST_NAME,
                "files": checksums,
                "schema_version": 1,
            }
            (staging / INSTALL_MARKER).write_text(
                json.dumps(marker_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if destination.exists():
                raise ProtocolBundleError(
                    f"Destination appeared during installation: {destination}."
                )
            os.rename(staging, destination)
            staging = None
    except zipfile.BadZipFile as exc:
        raise ProtocolBundleError(f"Not a valid ZIP protocol bundle: {archive_path}.") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        try:
            os.unlink(lock)
        except FileNotFoundError:
            pass
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install verified split and episode manifests from a separate protocol ZIP."
    )
    parser.add_argument("--archive", required=True, help="Path to the downloaded protocol ZIP.")
    parser.add_argument(
        "--sha256", required=True, help="Expected 64-character SHA-256 of the complete ZIP."
    )
    parser.add_argument(
        "--destination",
        default=str(DEFAULT_DESTINATION),
        help=f"Install directory (default: {DEFAULT_DESTINATION}).",
    )
    args = parser.parse_args(argv)
    try:
        target = install_protocol_bundle(args.archive, args.sha256, args.destination)
    except ProtocolBundleError as exc:
        print(f"Protocol install failed: {exc}", file=sys.stderr)
        return 2
    print(f"Verified protocol metadata installed at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
