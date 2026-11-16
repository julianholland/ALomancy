"""A sage_lib storage copied back from a remote job (issue #99).

ExPyRe's polling rsyncs a running job's files back without --delete. A
-wal copied mid-run stays in the local copy after the job's last
connection merged and deleted its own, and SQLite replays it over the
finished database: "database disk image is malformed", or silently
fewer rows. The writer here is a separate process, as on the remote."""

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from alomancy.database.global_database import (
    StaleWalError,
    drop_stale_sqlite_sidecars,
    finalize_sqlite,
    read_storage,
)

# Adds 20 structures, waits for a line on stdin (the mid-run copy), adds
# 200 more and exits, closing its connection like a finished remote job.
_WRITER = r"""
import sys
import numpy as np
from ase import Atoms
from alomancy.database.global_database import GlobalDatabase

db = GlobalDatabase(sys.argv[1])

def batch(n):
    out = []
    for i in range(n):
        a = Atoms("Cu2", positions=[[0, 0, 0], [1.1 + 0.01 * i, 0, 0]],
                  cell=[5, 5, 5], pbc=True)
        a.info.update(config_type="t", REF_energy=-1.0 - i)
        a.arrays["REF_forces"] = np.zeros((2, 3))
        out.append(a)
    return out

db.add_structures(batch(20), split="train", skip_duplicates=False)
print("snapshot", flush=True)
sys.stdin.readline()
for _ in range(5):
    db.add_structures(batch(40), split="train", skip_duplicates=False)
"""
_N_FINAL = 220


def _copied_back(tmp_path: Path, finalize: bool) -> Path:
    """The local copy of a finished job's storage, with the -wal/-shm a
    mid-run sync copied and the final sync never removed."""
    run = tmp_path / "run"
    store = run / "hybrid_local"
    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER, str(run)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert writer.stdout.readline().strip() == "snapshot"
        mid_run = tmp_path / "mid_run"
        shutil.copytree(store, mid_run)
        assert (mid_run / "db_index.sqlite-wal").exists()
        writer.stdin.write("\n")
        writer.stdin.flush()
        assert writer.wait(timeout=300) == 0
    finally:
        writer.kill()
    assert not (store / "db_index.sqlite-wal").exists()
    if finalize:
        finalize_sqlite(store)
    for suffix in ("-wal", "-shm"):
        shutil.copy(mid_run / f"db_index.sqlite{suffix}", store)
    return store


@pytest.mark.integration
def test_a_stale_wal_corrupts_the_finished_database(tmp_path):
    """The failure itself, with plain SQLite: the database file alone is
    complete; with the old -wal it is not."""
    store = _copied_back(tmp_path, finalize=False)
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(store / "db_index.sqlite", alone)
    con = sqlite3.connect(alone / "db_index.sqlite")
    assert con.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == _N_FINAL
    con.close()

    con = sqlite3.connect(f"file:{store / 'db_index.sqlite'}?mode=ro", uri=True)
    try:
        intact = con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        rows = con.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    except sqlite3.DatabaseError:
        intact, rows = False, None
    finally:
        con.close()
    assert not (intact and rows == _N_FINAL)


@pytest.mark.integration
def test_a_finalized_database_is_read_whole_and_its_leftovers_removed(tmp_path):
    store = _copied_back(tmp_path, finalize=True)

    assert len(read_storage(store)) == _N_FINAL
    assert not (store / "db_index.sqlite-wal").exists()
    assert not (store / "db_index.sqlite-shm").exists()


@pytest.mark.integration
def test_an_unfinalized_database_with_a_stale_wal_is_refused(tmp_path):
    """A database written by a remote without this fix: never read as if
    it were whole; the -wal is kept for the user to inspect."""
    store = _copied_back(tmp_path, finalize=False)

    with pytest.raises(StaleWalError, match="move the -wal and -shm aside"):
        read_storage(store)
    assert (store / "db_index.sqlite-wal").exists()


def _wal_db(path: Path) -> Path:
    path.mkdir()
    con = sqlite3.connect(path / "db_index.sqlite")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE objects (id INTEGER PRIMARY KEY)")
    con.executemany("INSERT INTO objects VALUES (?)", [(i,) for i in range(10)])
    con.commit()
    con.close()
    return path


def _journal_byte(path: Path) -> int:
    return (path / "db_index.sqlite").read_bytes()[18]


@pytest.mark.unit
def test_finalize_sqlite_leaves_one_rollback_mode_file(tmp_path):
    db = _wal_db(tmp_path / "db")
    assert _journal_byte(db) == 2

    finalize_sqlite(db)

    assert _journal_byte(db) == 1
    assert sorted(p.name for p in db.iterdir()) == ["db_index.sqlite"]
    con = sqlite3.connect(db / "db_index.sqlite")
    assert con.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 10
    con.close()


@pytest.mark.unit
def test_finalize_sqlite_raises_while_another_connection_reads(tmp_path):
    db = _wal_db(tmp_path / "db")
    holder = sqlite3.connect(db / "db_index.sqlite")
    holder.execute("BEGIN")
    holder.execute("SELECT * FROM objects").fetchall()
    try:
        with pytest.raises(RuntimeError, match="out of WAL mode"):
            finalize_sqlite(db, timeout=0.1)
    finally:
        holder.close()


@pytest.mark.unit
def test_sidecars_are_dropped_only_next_to_a_rollback_mode_database(tmp_path):
    db = _wal_db(tmp_path / "db")
    for suffix in ("-wal", "-shm"):
        (db / f"db_index.sqlite{suffix}").write_bytes(b"x")

    assert drop_stale_sqlite_sidecars(db) == []
    assert (db / "db_index.sqlite-wal").exists()

    (db / "db_index.sqlite-wal").unlink()
    (db / "db_index.sqlite-shm").unlink()
    finalize_sqlite(db)
    for suffix in ("-wal", "-shm"):
        (db / f"db_index.sqlite{suffix}").write_bytes(b"x")

    removed = drop_stale_sqlite_sidecars(db)

    assert sorted(p.name for p in removed) == [
        "db_index.sqlite-shm",
        "db_index.sqlite-wal",
    ]
    assert sorted(p.name for p in db.iterdir()) == ["db_index.sqlite"]
