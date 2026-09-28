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


def load_valid_metadata(path):
    valid_items = set()

    total = 0

    print("[1] Loading metadata...")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            total += 1

            obj = json.loads(line)

            asin = obj.get("asin")
            if not asin:
                continue

            title = str(obj.get("title") or "").strip()
            image = get_image_url(obj)

            # Same rule as preprocess_amazon2018.py
            if title and image:
                valid_items.add(str(asin))

            if total % 100_000 == 0:
                print(
                    f"[metadata] lines={total:,} "
                    f"valid_items={len(valid_items):,}"
                )

    print(f"[metadata] total lines       : {total:,}")
    print(f"[metadata] title+image items : {len(valid_items):,}")

    return valid_items


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reviews", required=True)
    parser.add_argument("--metadata", required=True)

    args = parser.parse_args()

    valid_items = load_valid_metadata(args.metadata)

    print("\n[2] Filtering interactions by valid multimodal items...")

    events = []

    total_reviews = 0
    multimodal_reviews = 0

    # Exact duplicates:
    # (user, asin, timestamp)
    seen_events = set()

    with gzip.open(args.reviews, "rt", encoding="utf-8") as f:
        for line in f:
            total_reviews += 1

            obj = json.loads(line)

            user = obj.get("reviewerID")
            asin = obj.get("asin")
            timestamp = obj.get("unixReviewTime")

            if not user or not asin or timestamp is None:
                continue

            user = str(user)
            asin = str(asin)

            if asin not in valid_items:
                continue

            multimodal_reviews += 1

            event_key = (user, asin, int(timestamp))

            if event_key in seen_events:
                continue

            seen_events.add(event_key)
            events.append(event_key)

            if total_reviews % 250_000 == 0:
                print(
                    f"[reviews] lines={total_reviews:,} "
                    f"kept={len(events):,}"
                )

    print("\n[3] User sequence filtering: >= 3")

    user_counts = Counter(user for user, _, _ in events)

    valid_users = {
        user
        for user, count in user_counts.items()
        if count >= 3
    }

    filtered_events = [
        event
        for event in events
        if event[0] in valid_users
    ]

    users = set()
    items = set()

    for user, asin, _ in filtered_events:
        users.add(user)
        items.add(asin)

    interactions = len(filtered_events)

    print("\n========== PREPROCESS MATCH CHECK ==========")

    print(f"Raw 5-core reviews        : {total_reviews:,}")
    print(f"After multimodal filter   : {multimodal_reviews:,}")
    print(f"After exact dedup          : {len(events):,}")

    print()
    print(f"Users                      : {len(users):,}")
    print(f"Items                      : {len(items):,}")
    print(f"Interactions               : {interactions:,}")

    # Assume leave-one-out validation + test
    predicted_train = interactions - (2 * len(users))

    print("\n---------- Leave-two-out prediction ----------")
    print(f"Predicted train interactions : {predicted_train:,}")
    print(f"Predicted valid interactions : {len(users):,}")
    print(f"Predicted test interactions  : {len(users):,}")

    print("\n---------- Friend reference ----------")
    print("Users              : 135,748")
    print("Items              : 47,520")
    print("Train interactions : 887,106")

    print("\n---------- Difference ----------")
    print(f"User diff  : {len(users) - 135748:+,}")
    print(f"Item diff  : {len(items) - 47520:+,}")
    print(f"Train diff : {predicted_train - 887106:+,}")


if __name__ == "__main__":
    main()
