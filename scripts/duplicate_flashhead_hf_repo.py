import argparse

from huggingface_hub import duplicate_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-id", default="Soul-AILab/SoulX-FlashHead-1_3B")
    parser.add_argument("--to-id", default="pkam24100/aivatar-flashhead-model")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    private = False if args.public else True

    result = duplicate_repo(
        from_id=args.from_id,
        to_id=args.to_id,
        repo_type=args.repo_type,
        private=private,
        exist_ok=args.exist_ok,
    )
    print(result)


if __name__ == "__main__":
    main()
