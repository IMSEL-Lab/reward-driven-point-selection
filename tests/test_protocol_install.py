from __future__ import annotations

import hashlib
import json
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from sam_rl.cli.install_protocol import (
    INSTALL_MARKER,
    ProtocolBundleError,
    install_protocol_bundle,
)


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_bundle(
    path: Path,
    entries: dict[str, bytes] | None = None,
    *,
    manifest_override: bytes | None = None,
    symlink: str | None = None,
    duplicate: str | None = None,
) -> str:
    payload = entries or {"fss_manifests/test.json": b'{"split":"test"}\n'}
    checksums = {name: checksum(value) for name, value in payload.items()}
    manifest = manifest_override or (json.dumps(checksums, sort_keys=True) + "\n").encode()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in payload.items():
                archive.writestr(name, value)
                if name == duplicate:
                    archive.writestr(name, value)
            if symlink is not None:
                info = zipfile.ZipInfo("fss_manifests/link.json")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, symlink)
            archive.writestr("SHA256SUMS.json", manifest)
    return checksum(path.read_bytes())


def test_install_bundle_verifies_files_and_is_idempotent(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    archive_sha = make_bundle(
        bundle,
        {
            "fss_manifests/test.json": b'{"split":"test"}\n',
            "splits/test.json": b'{"test": ["class-a"]}\n',
        },
    )
    destination = tmp_path / "installed"

    installed = install_protocol_bundle(bundle, archive_sha, destination)

    assert installed == destination
    assert (destination / "fss_manifests/test.json").read_bytes() == b'{"split":"test"}\n'
    marker = json.loads((destination / INSTALL_MARKER).read_text())
    assert marker["archive_sha256"] == archive_sha
    assert set(marker["files"]) == {"fss_manifests/test.json", "splits/test.json"}
    assert install_protocol_bundle(bundle, archive_sha, destination) == destination


def test_wrong_archive_hash_is_rejected_before_install(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    make_bundle(bundle)
    with pytest.raises(ProtocolBundleError, match="archive SHA-256 mismatch"):
        install_protocol_bundle(bundle, "0" * 64, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize(
    "name",
    ["../escape.json", "/absolute.json", "fss_manifests/../../escape.json"],
)
def test_unsafe_paths_are_rejected(tmp_path: Path, name: str) -> None:
    bundle = tmp_path / "protocol.zip"
    archive_sha = make_bundle(bundle, {name: b"{}"})
    with pytest.raises(ProtocolBundleError, match="Unsafe ZIP path"):
        install_protocol_bundle(bundle, archive_sha, tmp_path / "installed")
    assert not (tmp_path.parent / "escape.json").exists()


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    archive_sha = make_bundle(bundle, symlink="../outside")
    with pytest.raises(ProtocolBundleError, match="Symbolic links"):
        install_protocol_bundle(bundle, archive_sha, tmp_path / "installed")


def test_duplicate_member_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    archive_sha = make_bundle(bundle, duplicate="fss_manifests/test.json")
    with pytest.raises(ProtocolBundleError, match="Duplicate ZIP path"):
        install_protocol_bundle(bundle, archive_sha, tmp_path / "installed")


def test_member_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    make_bundle(
        bundle,
        {"fss_manifests/test.json": b'{"split":"test"}\n'},
        manifest_override=b'{"fss_manifests/test.json":"' + b"0" * 64 + b'"}\n',
    )
    archive_sha = checksum(bundle.read_bytes())
    with pytest.raises(ProtocolBundleError, match="Checksum mismatch"):
        install_protocol_bundle(bundle, archive_sha, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


def test_unrelated_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    archive_sha = make_bundle(bundle)
    destination = tmp_path / "installed"
    destination.mkdir()
    marker_file = destination / "keep.txt"
    marker_file.write_text("preserve")

    with pytest.raises(ProtocolBundleError, match="without a matching installation record"):
        install_protocol_bundle(bundle, archive_sha, destination)

    assert marker_file.read_text() == "preserve"


def test_modified_installed_file_is_not_silently_accepted(tmp_path: Path) -> None:
    bundle = tmp_path / "protocol.zip"
    archive_sha = make_bundle(bundle)
    destination = tmp_path / "installed"
    install_protocol_bundle(bundle, archive_sha, destination)
    (destination / "fss_manifests/test.json").write_text("modified")

    with pytest.raises(ProtocolBundleError, match="has changed"):
        install_protocol_bundle(bundle, archive_sha, destination)
