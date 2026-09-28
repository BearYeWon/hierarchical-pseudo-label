import argparse
import gzip
import json
from collections import defaultdict, Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()

    # user -> unique items
    user_items = defaultdict(set)

    total = 0

    print(f"[input] {args.reviews}")

    with gzip.open(args.reviews, "rt", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)

            user = obj.get("reviewerID")
            item = obj.get("asin")

            if user and item:
                user_items[user].add(item)

            if total % args.progress_every == 0:
                print(
                    f"[progress] lines={total:,} "
                    f"users={len(user_items):,}"
                )

    print("\n[1] Building unique user-item pairs...")

    user_deg = Counter()
    item_deg = Counter()

    unique_pairs = 0

    for user, items in user_items.items():
        user_deg[user] = len(items)
        unique_pairs += len(items)

        for item in items:
            item_deg[item] += 1

    print("\n========== UNIQUE PAIR STATS ==========")
    print(f"raw reviews       : {total:,}")
    print(f"unique pairs      : {unique_pairs:,}")
    print(f"duplicate reviews : {total - unique_pairs:,}")
    print(f"unique users      : {len(user_deg):,}")
    print(f"unique items      : {len(item_deg):,}")

    print("\n---------- User degree ----------")
    for k in [2, 3, 4, 5, 6, 7, 8, 10, 20]:
        n_users = sum(v >= k for v in user_deg.values())
        interactions = sum(
            v for v in user_deg.values()
            if v >= k
        )

        retained_items = set()

        for user, items in user_items.items():
            if len(items) >= k:
                retained_items.update(items)

        print(
            f"user >= {k:2d}: "
            f"users={n_users:,} "
            f"pairs={interactions:,} "
            f"items={len(retained_items):,}"
        )

    print("\n---------- Item degree ----------")
    for k in [2, 3, 4, 5, 10, 20]:
        n_items = sum(v >= k for v in item_deg.values())
        print(f"item >= {k:2d}: {n_items:,}")


if __name__ == "__main__":
    main()
