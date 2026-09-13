# Gaia S1 R2 — Bundle manifest and fixed point

**Status:** exact subject declaration for an unreviewed candidate
**Date:** 2026-08-15
**Revision:** R17 repair of the R16 candidate (ordinal aggregate digest `0b421ac7679937a13e35dc8f2f43456efa72d7f2611217076569bf3d5f996060`, internal thirteen-file digest `cc903de69840e335581e38c06ba0d9ec8849a59bbb4f32897c0a15352ec9a015`, 14 files, 632,720 bytes), repairing exactly two blockers, `S1R16-B01` and `S1R16-B02`, both reported on the Spec axis — **the R16 Standards axis returned `APPROVE` and is the first axis verdict in this lineage that was not `REQUEST_CHANGES`; it closes nothing on its own, its own author said so, and this revision claims nothing from it**
**Milestone:** M1 established; M2-M5 absent; **NOT INTEGRATED**

This manifest declares the exact bytes an independent reviewer should bind. It is not a verdict, not an acceptance, and not an approval.

**These are neither the R2, the R3, the R4, the R5, the R6, the R7, the R8, the R10, the R12, the R14 nor the R16 bytes.** The R2 candidate's fourteen-file ordinal aggregate digest was `5b3299f982f0813682fca2f7f20be7c5542ca6482fcab7e20e9fd5ce9f6e51e2` and its internal thirteen-file digest was `ff24dda314790768e5b88ae4124508b1e01e3272329a3d2b63262f382bf8d7c7`. The R3 candidate's fourteen-file ordinal aggregate digest was `97a3a8552016b9278c29a4e595677be84e4872a67b350d355a5273ecd21feb73` and its internal thirteen-file digest was `1c0160e1fd325e3a6aa57b121b0d45d7c08d5c7ae09e5cc3335e6944516878fc`. The R4 candidate's fourteen-file ordinal aggregate digest was `cb312cb5e1856fe3594eb42f65689b29f8194c843b275b96a9893107cd16a2ac` and its internal thirteen-file digest was `ed2246b2b56405f68baa9f2a38e8aaa1ee029fa85845c3f9087a86065db08709`. The R5 candidate's fourteen-file ordinal aggregate digest was `eb4deb3a1a92ae9061b7c8ece4798ac726b08f41d89ffb1d4dc59153f6b08ff2` and its internal thirteen-file digest was `adee9ede801621d84a8024fab5118f5ff5bb2503ca28ec514be41d1b42c4b5d9`. The R6 candidate's fourteen-file ordinal aggregate digest was `0df9e510415b1ffd04e5b312f7f9f89bf650102b217928f7c801cf444f000182` and its internal thirteen-file digest was `20e8d27a193488cd148e180da61e8a175cf38c95b6a4dd9161be532558ff84ed`. The R7 candidate's fourteen-file ordinal aggregate digest was `0b67158b7e2c169bb157d0c59fa736cd8abd23428d5271df6993601dfae39dbb` and its internal thirteen-file digest was `6af3510ff7ddb3ec7f767278fea9fc9391183ae7799f45a9e9292b37b7aad470`. The R8 candidate's fourteen-file ordinal aggregate digest was `deb16e5ca3a7d4726733d9a2956c9aa1bdb8808f75844fbe21fc7c4ebb913e6c` and its internal thirteen-file digest was `7de5c6c73fab1ade4cfc8bb536e33e66debd46b06a6bf6a5fcf576adff267363`. The R10 candidate — the bundle its own manifest declared as **R9** — had fourteen-file ordinal aggregate digest `208d0cacef94182d68f3f21c3e307e035b05540ccaa6b484b8ba54d6da5a041c` and internal thirteen-file digest `34a8345e73a61da7cdd3e4da63e36bdac54c621245cb2dbdb068d1e0182cfaab`. The R12 candidate — the bundle its own manifest declared as **R11** — had fourteen-file ordinal aggregate digest `596224b221e9463f49d04cb3909076c26ade0412d916c5f496d2c29e606fdb78` and internal thirteen-file digest `efa4db50eb799356fb1bfdd0de2dad38cc979fe7ae04d66b4968b0a22856509b`, over 14 files and 547,795 bytes. The R14 candidate — the bundle its own manifest declared as **R13** — had fourteen-file ordinal aggregate digest `b219cfff1135575ed1ebd3d46a4049cfe0cff52adbdad76c59a0a0e8de8b00fd` and internal thirteen-file digest `b2a2ecda6ef4459897845e24098a4fb03c0f6382c6533730d0eec5a23cd24790`, over 14 files and 583,299 bytes. The R16 candidate — the bundle its own manifest declared as **R15**, and the direct entry subject of this revision — had fourteen-file ordinal aggregate digest `0b421ac7679937a13e35dc8f2f43456efa72d7f2611217076569bf3d5f996060` and internal thirteen-file digest `cc903de69840e335581e38c06ba0d9ec8849a59bbb4f32897c0a15352ec9a015`, over 14 files and 632,720 bytes. All eleven of those subjects were independently reviewed, as was the S1 R1 subject before them, and **every one of the twelve returned at least one `REQUEST_CHANGES`, so not one of them is closed; each of those verdicts remains valid over its own bytes and none is superseded by this bundle.** Through R14 both axes returned `REQUEST_CHANGES` on every subject; **the R16 subject is the first on which the two axes differed** — Standards `APPROVE`, Spec `REQUEST_CHANGES` — and that `APPROVE` is recorded exactly as its own author framed it: one axis over one subject, granting no approval, no acceptance and no freshness, and unable to close S1 alone. The two R16 reviews blocked on two defects between them, both on the Spec axis — `S1R16-B01`, five change-ledger citations in a comma-list form no check read, carrying twenty line numbers of which sixteen were wrong, with an out-of-range citation passing the entire suite at exit 0; and `S1R16-B02`, a sentence in §6 of this manifest asserting that the ledger's census claims were derived from a read over the shipped file when the ledger said 210 and the shipped file said 212 — and this revision repairs exactly those two. The change ledger carries one section per repaired blocker group in the established shape, twenty-seven in all. **This revision edits the specification in seven lines**, all in place: `spec-v0.2:10`, `:16`, `:18`, `:474` and `:484` re-stamp the revision block, the review lineage, the repair list, the S1 gate row and the next-action paragraph for this candidate, `:571` states the measured anchor and citation-census figures for `S1R16-B01`, and `:574` states the measured suite figures. No line was added and none removed, so the file stands at 579 lines and every citation and load-bearing anchor is undisturbed; every edit narrows no normative requirement and adds no authority. **`S1R16-B02` required no specification byte at all** and is repaired in §6 of this file. The filenames and the `v0.2` version string are deliberately unchanged so that every `spec-v0.2:` / `ebnf-v0.2:` citation and every validator file constant stays valid; the two subjects are distinguished by digest, not by name. A reviewer must bind the digests in §3, not the filenames.

## 1. Ordering and digest construction

**Ordinal filename order** means ascending comparison of the UTF-8 code units of the file name, with no locale, case folding, or path-segment special casing. All names are 7-bit ASCII, so this is plain byte order.

The manifest lists every file in the bundle **except itself**. A manifest cannot contain its own digest.

The **aggregate digest** is constructed as: one `name|bytes|sha256` record per file, in ordinal filename order, joined with LF (`\n`), **with no trailing newline**, then SHA-256 over the UTF-8 encoding of that string.

## 2. Inventory

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `gaia-artifact-workgraph-context-v0.1.md` | 10,956 | `a5e3492c94feafd6eeb3a243aed1359ce121417c5069cb59be0e17f99aa0c57a` |
| `gaia-artifact-workgraph-staleness-design-v0.1.md` | 53,037 | `7b9d0cd64ad29e897f6b294b2a2c91e1a5bcb708a89a1a4871a14eba926fd219` |
| `gaia-consolidated-mission-room-factory-spec-v0.2.md` | 77,673 | `93b836e75b765963f138f4d01017ba3c6fcf834e25e0729ccb91538fa2e67660` |
| `gaia-engineering-doctrine-v0.1.md` | 10,866 | `f27c54b52c0a6a3b0479052f048223fe1a5c4894b0091897e94f8ee65cb858a0` |
| `gaia-mission-room-domain-context-v0.1.md` | 4,859 | `4aaf104d8a80ed3ee8a8d312a8a40ecd401cc51a0e5a901f355a431c57262bad` |
| `gaia-mission-room-protocol-v0.2.ebnf` | 27,843 | `a52a4141c6ab7777c8033df7e091ee40d591fe497bc8ddbac591cc5cef11342a` |
| `gaia-multiaxis-uncertainty-analysis-design-v0.1.md` | 10,999 | `65a32cb0ddd5a579efb17c2d5fd66c25ffa876faa9afe8dd3f192cc67ea8c6a7` |
| `gaia-multiaxis-uncertainty-math-primary-research.md` | 11,568 | `06da04bd64d8e092d71f3900ccc87806602909b1839ce46855910da79c6d8c2c` |
| `gaia-s1-r2-change-ledger.md` | 206,581 | `8a641eae3d40bddb0215cff4dd21da1f97acd9a9faae6d27172bd3d560829416` |
| `gaia-s1-r2-third-party-notices.md` | 9,595 | `96c93ef40abbeecd8ce909a0eba61c38b184074723b2bcba46b28b109b31c4be` |
| `gaia-s1-r2-validator-output.txt` | 23,567 | `6e0f9f7338bf724a43d775d78cbf75f99d67c4c2ad922dbf7f59b3d0f75462a3` |
| `gaia-s1-r2-validator.py` | 181,714 | `35180a551fd64ae581d753ddfd279baa6f3a51fd4d310638e800553b360b7955` |
| `gaia-uncertainty-grammar-v0.1.ebnf` | 10,322 | `ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b7394517` |

- **files:** 13, plus this manifest
- **total bytes:** 639,580
- **directory contains no fourteenth entry besides this manifest**, no subdirectory, no hidden file, and no cache, build, or crash artifact

## 3. Aggregate digest

```
970bb50c3f40aac8da50f5822fc8e6ad48f7e27643dd67176591a7788f0b311c
```

Discriminating variants are recorded so an independent lane can confirm it reproduced the construction rather than a coincidence. Only the first reproduces the declared value.

| Construction | Digest | Declared |
|---|---|---|
| LF-joined, **no** trailing newline | `970bb50c3f40aac8da50f5822fc8e6ad48f7e27643dd67176591a7788f0b311c` | yes |
| LF-joined, trailing newline | `3c53ad575c81f7c0220a112eacd359810949a0b71e4b6d9e4ff74717883d8d74` | no |
| CRLF-joined, trailing newline | `457e745ba91e21f93330296bfbf9d5715d3b665a99178e66cd0fe3338dfc8bdb` | no |
| Concatenated, no separator | `ea3b98f1d47f72b2282fba0b066054a3b914a2f727e78899573f6205b36223e6` | no |

**The construction is pinned by a subject this revision did not produce, and this lane re-measured that pinning rather than inheriting it.** Run over the R16 subject — the fourteen files this revision entered on, which this lane holds — the same code reproduces its declared internal thirteen-file digest `cc903de69840e335581e38c06ba0d9ec8849a59bbb4f32897c0a15352ec9a015`, all three of its declared discriminating variants (`f3c904a2…`, `0074feba…`, `7ab8d0be…`), its declared totals of 612,319 and 632,720 bytes, and its fourteen-file ordinal aggregate `0b421ac7679937a13e35dc8f2f43456efa72d7f2611217076569bf3d5f996060` — the value the two R16 reviews bound. The equivalent statements earlier manifests made about the R8, R10, R12 and R14 subjects are **carried forward unverified by this lane**; they are recorded as those rounds' measurements, not as this one's.

## 4. What each file is

| File | Role | Reviewed before? |
|---|---|---|
| `gaia-consolidated-mission-room-factory-spec-v0.2.md` | consolidated specification — **seven lines edited in place at R17**: `spec-v0.2:10`, `:16`, `:18`, `:474` and `:484` re-stamp the revision block, the review lineage, the repair list, the S1 gate row and the next-action paragraph for this candidate, including the fact that the previous revision of `:484` predicted the ninth escape in the wrong reader; `:571` states the measured load-bearing-anchor and ledger-citation counts and records that the census is now the ledger's content rather than a pattern's yield (`S1R16-B01`); `:574` states the measured suite figures. `:573` is unchanged — the thirteen controls in four families it names are still accurate. No line was added and none removed, so the file stands at 579 lines and all 239 ledger line citations and all 45 load-bearing anchors are undisturbed; no normative requirement was narrowed, no verb, lattice, refusal code, authority or subsystem was added or removed, and `**Status:**`, `**Milestone:**`, `**Authority:**` and every gate row other than S1 are byte-identical | no — new bytes. The other 572 lines are identical to bytes the R16 subject carried |
| `gaia-mission-room-protocol-v0.2.ebnf` | Mission-Room protocol grammar — **byte-identical to the R4 candidate in its entirety**, comments included, and untouched by every repair from R5 through R17; against R3 every production is byte-identical and only normative comment text changed | yes — identical bytes were in the R4 subject |
| `gaia-engineering-doctrine-v0.1.md` | normative engineering input, **byte-exact copy** | yes — in the S1 R1 subject, unchanged |
| `gaia-mission-room-domain-context-v0.1.md` | canonical glossary, **byte-exact copy** | yes — in the S1 R1 subject, unchanged |
| `gaia-artifact-workgraph-context-v0.1.md` | candidate glossary input, **byte-exact copy** | no |
| `gaia-artifact-workgraph-staleness-design-v0.1.md` | candidate design input, **byte-exact copy** | no |
| `gaia-uncertainty-grammar-v0.1.ebnf` | candidate grammar input, **byte-exact copy** | no |
| `gaia-multiaxis-uncertainty-math-primary-research.md` | pinned primary-source research, **byte-exact copy** | no |
| `gaia-multiaxis-uncertainty-analysis-design-v0.1.md` | candidate design input, **byte-exact copy** | no |
| `gaia-s1-r2-change-ledger.md` | S1-B1..B6, S1R2-B01..B02, S1R3-B01..B02, S1R4-B01..B02, S1R5-B01..B02, S1R6-B01, S1R7-B01..B02, S1R8-B01, S1R10-B01 on both axes, S1R10-B02, S1R12-B01 on both axes, S1R14-B01 on both axes with S1R14-B02 and S1R14-B03, and S1R16-B01 and S1R16-B02 on the Spec axis — old text, new text, rationale, verification, one `**Verification.**` paragraph per group, twenty-seven in all. Its five comma-list citation spans are normalised into the canonical syntax and twenty-one wrong line-number instances are corrected, and the superseded numbers are written as text rather than as backticked spans so a stale reference cannot wear the notation of a live claim. Its preamble now declares the round-scoping convention: each round section reports the figures measured for its own round, and the current figures live in this manifest and in the captured run | no — new bytes. It carries the two sections this round owes |
| `gaia-s1-r2-third-party-notices.md` | copyright and source ledger — **byte-identical to the R7, R8, R10, R12, R14 and R16 candidates** | restatement of an axis R1 and the R2 review both passed |
| `gaia-s1-r2-validator.py` | offline static validator: 58 checks, 95 embedded negative fixtures, 13 disclosed-boundary positive controls in four families, two independent EBNF readers. R17 adds one check, `LEDGER_CITATIONS_CANONICAL`, and one fixture, `NEG-L-e`, and adds nineteen rows to the load-bearing-anchor table, which now stands at 45 | no — new bytes |
| `gaia-s1-r2-validator-output.txt` | captured validator run over these exact bytes | no |

The doctrine and domain context are **copied rather than referenced** so the review is mechanically complete inside the bundle: the specification declares them normative inputs and states that conflict with either is an S1 Spec defect, so a reviewer must hold their bytes.

Two provenance artifacts are **referenced by digest rather than copied** — `UPSTREAM-MAP.md` and `THIRD_PARTY_NOTICES.md` — because they were not among this lane's trusted read-only inputs, at R2 or at R3. Their digests are in `gaia-s1-r2-third-party-notices.md` §3.

## 5. Reproducing this fixed point

```bash
cd gaia-s1-r17-bundle-work-20260815T170044Z
for f in $(ls | grep -v gaia-s1-r2-bundle-manifest.md | LC_ALL=C sort); do
  printf "%s|%s|%s\n" "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
done | head -c -1 | sha256sum
```

Then re-run the validator from these exact bytes, with bytecode generation disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B gaia-s1-r2-validator.py
PYTHONDONTWRITEBYTECODE=1 python -B gaia-s1-r2-validator.py --verify-manifest
```

The directory name above is the one these bytes were assembled in; revisions before R11 named the *previous* round's directory, which the R11 repair corrected along with the rest of `S1R10-B02`. The name is a convenience for the recipe only — nothing in the bundle depends on it, and a reviewer may run the two commands from any directory holding these fourteen files.

The first command must reproduce `gaia-s1-r2-validator-output.txt` byte for byte. The second verifies every row of §2 against the files on disk and reports any unlisted file.

**Do not redirect the validator's stdout into the bundle directory.** Doing so creates a fifteenth file, which `NO_DEBRIS` correctly reports as unexpected, and the run exits 1 for a reason that has nothing to do with the bundle. The R3 independent reviewer hit exactly this on a first attempt and recorded it as `S1R3-N06`; the commands above do not redirect, and a reviewer who wants a captured copy should redirect **outside** the directory. This is disclosed rather than repaired: the recipe as written is correct, and the trap is in the obvious next step.

## 6. Line endings

**Corrected at R3, corrected again at R11, re-measured at R13, R15 and R17, and the scope of its final sentence corrected at R17 (`S1R16-B02`).** The R2 manifest stated that every file uses CRLF. That was wrong, and no review caught it. Measured over the raw bytes of these fourteen files:

- **thirteen files use LF only and contain zero CR bytes** — including all seven byte-exact copies, which arrived that way, and including this manifest;
- **one file uses CRLF** — `gaia-s1-r2-validator-output.txt`, with 214 CRLF pairs and no bare LF, because it is captured by redirecting Python's stdout on Windows, where text-mode stdout translates a newline written as `\n` into the two bytes `\r\n`.

The figure is one per printed line, so it moves whenever the suite does: it was 200 over the R9 output, 206 over the R11 output, 210 over the R13 output, 212 over the R15 output, and 214 here, because this round adds one check and one negative fixture and no control, which print one line each. **The R11 change ledger still said `200` in both places it stated this census, against an R11 output measuring 206 and an R11 manifest §6 saying 206** — the defect the R12 Spec review reported as `S1R12-B01`.

**The scope of the sentence that stood here is corrected at R17, and the divergence it papered over is stated instead (`S1R16-B02`).** That sentence read: *"Every figure in this section and in the ledger's two census claims is derived from a single read over the shipped file in one pass, and each round records its correction as a correction."* Over the R16 bytes that was measurably false — the shipped output carried 212 CRLF pairs, this section said 212, and the change ledger's census claims at ledger:551, :555 and :645 said **210**. The correct statement, and the one this section now makes, has three parts. **First:** every figure in this section is derived from a single read over the shipped file in one pass, and each round records its correction as a correction. **Second:** the ledger's census claims are **round-scoped** under the convention its own line 10 declares — each round section reports the figures measured for its own round and they are not restated when a later round moves them — so those three statements stand at 210 as the R13 measurement, they are correct as dated measurements, they are **not** the current figure, and they have deliberately not been rewritten. **Third:** the current figure is 214 and it lives here and in the captured run, which is exactly what ledger:10 says of the two places current figures live. Editing those three dated ledger figures to agree with this one was considered and refused: agreement bought by re-stamping history is not agreement, and it would destroy the record that lets a reviewer see when each figure moved.

The escape sequences in that sentence are written as **text**, deliberately.

**The nuance recorded by the R3 review as `S1R3-N05(b)` is closed here rather than left open, and the round that broke it is named.** Every revision through R8 illustrated the translation with a code span containing a *literal* CRLF, which made this manifest a mixed-ending file by raw measurement while the per-file list above classified it as an LF file. The R9 re-stamp normalized that byte away without noticing, leaving this section asserting "this manifest contains one literal CRLF" and "by raw measurement this manifest is a mixed-ending file" over bytes that contained **zero** CR — the defect the R10 Spec review reported as `S1R10-B01` on its axis. Two repairs were available: restore the literal CRLF, or keep the bytes pure LF and make the prose describe the file as it is. Re-arming a single stray CR byte that no check can see, that no consumer preserves deliberately, and that an ordinary edit has already destroyed once is the weaker of the two, so the illustration now uses escapes as text and **this manifest is a pure-LF file**: `open(path,'rb').read().count(b'\r')` returns `0` over these bytes, and the list above says so. A reviewer measuring rather than classifying now sees exactly what this section claims. Nothing mechanical enforces that — no check in the validator binds this manifest's prose to this manifest's bytes — so this remains a claim a reviewer must measure, and the change ledger records it as an open residual rather than as a closed mechanism.

This matters for reproduction, not just for accuracy. A reviewer who re-captures the validator output on a platform without that translation will get an LF file whose SHA-256 differs from the row in §2 while every printed character is identical. Compare the *displayed* run against the captured file, or normalize line endings before hashing, before concluding the output has drifted. The validator normalizes line endings for text analysis only; all hashes in §2 and §3 are over raw bytes.

## 7. Authority

This manifest declares bytes. It grants no approval, no acceptance, no freshness, and no authority. **S1 remains `Pending`.** The S1 R1 verdict over aggregate digest `f44ecac23f0c9d180f1bcdcbacb920163a89dc0d04cebbbe6825dbc556c32832`, the S1 R2 verdict over ordinal aggregate digest `5b3299f982f0813682fca2f7f20be7c5542ca6482fcab7e20e9fd5ce9f6e51e2`, the S1 R3 verdict over ordinal aggregate digest `97a3a8552016b9278c29a4e595677be84e4872a67b350d355a5273ecd21feb73`, the two independent S1 R4 reviews over ordinal aggregate digest `cb312cb5e1856fe3594eb42f65689b29f8194c843b275b96a9893107cd16a2ac`, the two independent S1 R5 reviews over ordinal aggregate digest `eb4deb3a1a92ae9061b7c8ece4798ac726b08f41d89ffb1d4dc59153f6b08ff2`, the two independent S1 R6 reviews over ordinal aggregate digest `0df9e510415b1ffd04e5b312f7f9f89bf650102b217928f7c801cf444f000182`, the two independent S1 R7 reviews over ordinal aggregate digest `0b67158b7e2c169bb157d0c59fa736cd8abd23428d5271df6993601dfae39dbb`, the two independent S1 R8 reviews over ordinal aggregate digest `deb16e5ca3a7d4726733d9a2956c9aa1bdb8808f75844fbe21fc7c4ebb913e6c`, the two independent S1 R10 reviews over ordinal aggregate digest `208d0cacef94182d68f3f21c3e307e035b05540ccaa6b484b8ba54d6da5a041c`, the two independent S1 R12 reviews over ordinal aggregate digest `596224b221e9463f49d04cb3909076c26ade0412d916c5f496d2c29e606fdb78`, the two independent S1 R14 reviews over ordinal aggregate digest `b219cfff1135575ed1ebd3d46a4049cfe0cff52adbdad76c59a0a0e8de8b00fd`, and the two independent S1 R16 reviews over ordinal aggregate digest `0b421ac7679937a13e35dc8f2f43456efa72d7f2611217076569bf3d5f996060` — the two verdicts that authorized this round — each remain valid over their own subject and none is superseded by this bundle. **Twelve subjects have been independently reviewed and every one of them returned at least one `REQUEST_CHANGES`, so not one is closed**: the R1, R2 and R3 subjects by a single independent review each, and every subject from R4 onward by two independent reviews. Twenty-three of the twenty-four axis verdicts were `REQUEST_CHANGES`; the twenty-fourth, the R16 Standards axis, returned `APPROVE` over its own axis and its own subject and expressly stated that it grants no approval, no acceptance and no freshness and cannot alone close S1. Nothing in this bundle claims otherwise. Listing those verdicts is recording them, not producing one. No T1/M2 authorization follows from anything here, and this author grants no approval to these bytes.

`GAIA_S1_R2_BUNDLE_MANIFEST_COMPLETE`
