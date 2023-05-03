#!/usr/bin/env python3
"""Script."""

import argparse

from cartotb_satellite import satellite


def main(args):
    """Do the main."""
    # transform a population count to boundaries
    if "threshold" in args:
        satellite.find_cities(args.input, threshold=args.threshold, outfile=args.output)

    # stick together all tiles to geotiff
    if "format" in args:
        satellite.merge_tiles(args.boundaries, tilename_fmt=args.format, outfile_fmt=args.output)

    # read geotiff and compute TB risk
    if "input_geotif" in args:
        satellite.satellite(imagery=args.input_geotif, output=args.output)


def parse():
    """Parse Args."""
    args = argparse.ArgumentParser(
        prog=__file__, description="Compute disease risk from satellite images."
    )
    subcommands = args.add_subparsers(help="Sub command help")

    # find cities
    args_cities = subcommands.add_parser("cities")
    args_cities.add_argument(
        "--threshold", type=float, default=6000, help="population count threshold (per km^2)."
    )
    args_cities.add_argument(
        "-o",
        "--output",
        type=str,
        help="filename for the output files (default: `cities.geojson`)",
        default="cities.geojson",
    )
    args_cities.add_argument("input", type=str, help="path to population count file (GeoTIFF).")

    # merge tiles to geotif
    args_tiles = subcommands.add_parser("tiles")
    args_tiles.add_argument(
        "-b",
        "--boundaries",
        required=True,
        help="Geojson file with city boundaries as features.",
    )
    args_tiles.add_argument(
        "-f", "--format", required=True, help="File path as `path/to/tile_{x}_{y}_{z}.png`"
    )
    args_tiles.add_argument(
        "-o",
        "--output",
        type=str,
        help="filename for the output files (default: `city_{}.geotif`)",
        default="city_{}.geotif",
    )

    # find risk
    args_risk = subcommands.add_parser("risk")
    args_risk.add_argument(
        "-o",
        "--output",
        type=str,
        help="filename for the output files",
        default="city_risk.geotif",
    )
    args_risk.add_argument("input_geotif", type=str, help="Saltellite image (GeoTIFF).")

    main(args.parse_args())
    # main(args)
