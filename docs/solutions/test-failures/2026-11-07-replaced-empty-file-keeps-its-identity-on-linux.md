---
title: "A file replaced at the same path keeps its (dev, inode) identity on Linux: the freed inode is reused"
category: test-failures
date: 2026-11-07
tags: [go, ixmux, journal, inode, file-identity, linux, ci]
symptom: "go-modules (ubuntu-latest) red in TestReplacedEmptyJournalIsStale (journal/hardening_test.go): an empty journal A removed and replaced by B does not give stale-incarnation; green on Windows"
root_cause: "The journal incarnation hashed the file identity (dev+inode on Unix, volume+file index on Windows) plus the first record. An empty file has no first record, so its identity was the inode alone; Linux ext4/tmpfs hands the freed inode to the next file created, so B had A's identity. Windows file indexes are not reused this way"
---

# A replaced file can keep its identity on Linux

## Problem

The journal names a file by an incarnation, so a reader's checkpoint taken on one file is refused
(`stale-incarnation`) on another file put at the same path. The test removed an empty journal A,
created B at the same path and expected A's checkpoint to be stale. Green on Windows, red on Linux.

## Root cause

`(dev, inode)` identifies a live file, not a file over time: after `unlink`, Linux reuses the
inode number, often immediately for the next `create` in the same directory. With content, the
first record also enters the hash and tells A from B; an empty file has nothing else. Ctime
changes on every append (so it would break a growing journal), and birth time (statx) or the
generation number (`FS_IOC_GETVERSION`) are not in `syscall` and depend on the filesystem.

## Fix

Commit 52397f57: an empty journal names no file. Every empty journal has
`journal.EmptyIncarnation`; a checkpoint taken on it (offset 0, no scan hint) is read from 0 on
whatever file is at the path, and `PromotedFrom` says so. Nothing is lost or repeated, since no
record had completed. `TestEmptyJournalCheckpointPromisesNothing` binds it.

The limit that remains is written in the README and in `Identify`'s doc (an `@ai:assumption`
with `U:uncertain`): two files at one path that share a reused inode are told apart only by
their first record, so a checkpoint on an incomplete first record may be read from 0 on the new
file instead of refused. The fake session's file identity (`server/fileid_unix.go`) has the same
limit.

## How to find it again

```
# a Linux kernel is needed
go test -count=1 -run TestEmptyJournalCheckpointPromisesNothing ./journal/
```

Any "same file?" check built on `(dev, inode)` across a delete and re-create is this note.
