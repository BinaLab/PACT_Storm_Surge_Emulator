"""Argument parsers shared by training and inference."""

import argparse


def parse_bool_int(value):
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return 1
    if value in ("0", "false", "no", "n", "off"):
        return 0
    raise argparse.ArgumentTypeError("expected one of: 0/1, true/false, yes/no, on/off")
