import argparse
import re

UNIT = 110000

def parse_yen(value: str) -> int:
    # Remove currency symbols, commas, spaces, and other separators
    cleaned = re.sub(r"[^\d]", "", value)
    if cleaned == "":
        raise argparse.ArgumentTypeError(f"Invalid amount: {value}")
    return int(cleaned)

parser = argparse.ArgumentParser()
parser.add_argument("--total",type=parse_yen)
parser.add_argument("--center",type=parse_yen)
args = parser.parse_args()

total = ((args.total + UNIT - 1) // UNIT) * UNIT
units = total / 275
center = args.center + (total - args.total)

print(f"units : {units}")
print(f"center: {center}")
print(f"total : {total}")
