import argparse
import gzip
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()

    user_deg = Counter()
    item_deg = Counter()

    total = 0
    bad_lines = 0

    print(f"[input] {args.reviews}")

    with gzip.open(args.reviews, "rt", encoding="utf-8") as f:
        for line in f:
            total += 1

            try:
                obj = json.loads(line)
                user = obj.get("reviewerID")
                item = obj.get("asin")

                if not user or not item:
                    bad_lines += 1
                    continue

                user_deg[user] += 1
                item_deg[item] += 1

            except Exception:
                bad_lines += 1

            if total % args.progress_every == 0:
                print(
                    f"[progress] lines={total:,} "
                    f"users={len(user_deg):,} "
                    f"items={len(item_deg):,}"
                )

    print("\n========== RAW TOYS STATS ==========")
    print(f"reviews          : {total:,}")
    print(f"valid reviews    : {sum(user_deg.values()):,}")
    print(f"bad lines        : {bad_lines:,}")
    print(f"unique users     : {len(user_deg):,}")
    print(f"unique items     : {len(item_deg):,}")

    print("\n---------- User degree ----------")
    for k in [1, 2, 3, 4, 5, 10, 20]:
        count = sum(v >= k for v in user_deg.values())
        print(f"users degree >= {k:2d}: {count:,}")

    print("\n---------- Item degree ----------")
    for k in [1, 2, 3, 4, 5, 10, 20]:
        count = sum(v >= k for v in item_deg.values())
        print(f"items degree >= {k:2d}: {count:,}")

    user_values = sorted(user_deg.values())
    item_values = sorted(item_deg.values())

    if user_values:
        print("\nuser degree min/max:",
              user_values[0], user_values[-1])

    if item_values:
        print("item degree min/max:",
              item_values[0], item_values[-1])


if __name__ == "__main__":
    main()
