# ============================================================
# Command line: run the pipeline for one or many tracks
# ============================================================
"""
Examples
--------
    is2retreat --config configs/north_slope.toml --track 0129
    is2retreat --config configs/north_slope.toml --tracks-file configs/tracks_filtered.txt
    is2retreat --config ... --track 0129 --outdir /tmp/test --set SIZE_LIMIT_M=90
"""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
import traceback
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .pipeline import run_track
from .utils import TrackSkipped, format_track_id


def _parse_override(text: str):
    """KEY=VALUE with a TOML value (0.5, 180, "x", [0.1, 0.2]); bare text is a string."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set expects KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    try:
        parsed = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value
    return key.strip(), parsed


def _read_tracks(args) -> list[str]:
    tracks = list(args.track or [])
    if args.tracks_file:
        for line in Path(args.tracks_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tracks.append(line)
    return [format_track_id(t) for t in tracks]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="is2retreat",
        description="ICE-BEAM: Arctic bluff retreat from ICESat-2 (SlideRule ATL06) tracks.",
    )
    p.add_argument("--config", required=True, help="TOML config with [paths] and optional [params].")
    p.add_argument("--track", action="append", help="Track (RGT) id; repeat for several.")
    p.add_argument("--tracks-file", help="Text file with one track id per line.")
    p.add_argument("--source", choices=["auto", "cache", "sliderule"], default="auto",
                   help="Beam data: cached GeoPackage if present (auto), cache only, or always SlideRule.")
    p.add_argument("--outdir", help="Override [paths] outdir.")
    p.add_argument("--no-save-cache", action="store_true",
                   help="Don't write fresh SlideRule pulls to the cache.")
    p.add_argument("--set", action="append", type=_parse_override, default=[], metavar="KEY=VALUE",
                   help="Override a Params value, e.g. --set SIZE_LIMIT_M=90.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    tracks = _read_tracks(args)
    if not tracks:
        print("No tracks given (use --track or --tracks-file).", file=sys.stderr)
        return 2

    paths, params = load_config(args.config, **dict(args.set))
    if args.outdir:
        paths = replace(paths, outdir=Path(args.outdir).resolve())

    verbose = not args.quiet
    done, skipped, failed = [], [], []

    for i, track_id in enumerate(tracks, start=1):
        print(f"\n[{i}/{len(tracks)}] TRACK_ID={track_id}")
        start = time.time()
        try:
            result = run_track(
                track_id, paths, params,
                source=args.source,
                save_cache=not args.no_save_cache,
                verbose=verbose,
            )
        except TrackSkipped as e:
            print(f"SKIP: {e}")
            skipped.append(track_id)
            continue
        except Exception:
            traceback.print_exc()
            failed.append(track_id)
            continue

        n_gie = 0 if result.gie is None else len(result.gie.summary_df)
        print(
            f"OK: {len(result.dsas_summary)} DSAS clusters, {n_gie} GIE clusters "
            f"({time.time() - start:.0f}s)"
        )
        done.append(track_id)

    print(f"\nFinished: {len(done)} ok, {len(skipped)} skipped, {len(failed)} failed.")
    if failed:
        print("Failed tracks: " + " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
