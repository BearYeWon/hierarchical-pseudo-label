import argparse
import gzip
import json
from collections import Counter


def first_nonempty(value):
    if isinstance(value, list):
        for x in value:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return None

    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def get_image_url(meta):
    for key in ["imageURLHighRes", "imageURL", "image"]:
        url = first_nonempty(meta.get(key))
        if url:
            return url
    return None


def load_valid_items(meta_path):
    valid_items = set()

    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)

            asin = obj.get("asin")
            if not asin:
                continue

            title = str(obj.get("title") or "").strip()
            image = get_image_url(obj)

            if title and image:
                valid_items.add(str(asin))

            if i % 100_000 == 0:
                print(
                    f"[metadata] lines={i:,}, "
                    f"valid_items={len(valid_items):,}"
                )

    return valid_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    print("[1] Loading valid multimodal items...")
    valid_items = load_valid_items(args.metadata)

    print(f"[metadata] valid title+image items={len(valid_items):,}")

    print("\n[2] Loading and filtering review events...")

    events = []
    seen = set()

    raw = 0
    multimodal = 0

    with gzip.open(args.reviews, "rt", encoding="utf-8") as f:
        for line in f:
            raw += 1

            obj = json.loads(line)

            user = obj.get("reviewerID")
            asin = obj.get("asin")
            ts = obj.get("unixReviewTime")

            if not user or not asin or ts is None:
                continue

            user = str(user)
            asin = str(asin)
            ts = int(ts)

            if asin not in valid_items:
                continue

            multimodal += 1

            key = (user, asin, ts)

            if key in seen:
                continue

            seen.add(key)
            events.append(key)

            if raw % 250_000 == 0:
                print(
                    f"[reviews] raw={raw:,}, "
                    f"dedup_kept={len(events):,}"
                )

    user_counts = Counter(user for user, _, _ in events)

    print("\n==============================================")
    print(" USER MIN-SEQUENCE THRESHOLD COMPARISON")
    print("==============================================")

    print(f"Raw reviews          : {raw:,}")
    print(f"Multimodal reviews   : {multimodal:,}")
    print(f"Deduplicated events  : {len(events):,}")

    print()

    TARGET_USERS = 135_748
    TARGET_ITEMS = 47_520
    TARGET_TRAIN = 887_106
    TARGET_TOTAL = TARGET_TRAIN + 2 * TARGET_USERS

    print(
        f"Friend target before split: "
        f"users={TARGET_USERS:,}, "
        f"items={TARGET_ITEMS:,}, "
        f"interactions={TARGET_TOTAL:,}"
    )

    print()

    for k in range(3, 11):
        valid_users = {
            user
            for user, count in user_counts.items()
            if count >= k
        }

        kept = [
            e for e in events
            if e[0] in valid_users
        ]

        items = {
            asin
            for _, asin, _ in kept
        }

        users = len(valid_users)
        interactions = len(kept)

        # leave-one-out validation + test
        train = interactions - 2 * users

        print(
            f"k={k:2d} | "
            f"users={users:7,d} | "
            f"items={len(items):6,d} | "
            f"total={interactions:9,d} | "
            f"pred_train={train:9,d} | "
            f"diff_train={train - TARGET_TRAIN:+9,d}"
        )


if __name__ == "__main__":
    main()
