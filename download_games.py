"""Download games from any Lichess player and save as PGN for pretraining.

Usage
-----
    python download_games.py
    python download_games.py --user DrNykterstein --perf blitz --max 5000
"""

import argparse
import os
import sys
from pathlib import Path

GAMES_DIR = Path(__file__).parent / "chess_rl" / "games"
ENV_FILE  = Path(__file__).parent / ".env"

PERF_TYPES = {"1": "bullet", "2": "blitz", "3": "rapid", "4": "classical"}


def _load_token() -> str:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("LICHESS_TOKEN"):
                token = line.split("=", 1)[1].strip()
                if token and token != "paste_your_token_here":
                    return token
    token = os.environ.get("LICHESS_TOKEN", "")
    if not token:
        print("No token found in .env — add LICHESS_TOKEN=your_token to .env")
        sys.exit(1)
    return token


def download(username: str, perf: str | None, max_games: int) -> Path:
    try:
        import berserk
    except ImportError:
        print("berserk not installed. Run: pip install berserk")
        sys.exit(1)

    token  = _load_token()
    client = berserk.Client(berserk.TokenSession(token))

    # Verify user exists before downloading
    try:
        user = client.users.get_public_data(username)
        username = user["username"]   # use exact casing from Lichess
    except Exception:
        print(f"User '{username}' not found on Lichess.")
        sys.exit(1)

    perf_label = perf or "all"
    out_path   = GAMES_DIR / f"{username.lower()}_{perf_label}.pgn"
    GAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nPlayer : {username}")
    print(f"Type   : {perf_label}")
    print(f"Max    : {max_games} games")
    print(f"Output : {out_path}\n")

    kwargs = {"as_pgn": True, "max": max_games}
    if perf:
        kwargs["perf_type"] = perf

    count = 0
    with open(out_path, "w") as f:
        for game in client.games.export_by_player(username, **kwargs):
            f.write(game + "\n\n")
            count += 1
            if count % 100 == 0:
                print(f"  {count} games downloaded...")

    print(f"\nDone — {count} games saved to {out_path}")
    print(f"\nNext step:")
    print(f"  python pretrain.py --pgn {out_path} --epochs 3")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Lichess games for pretraining.")
    p.add_argument("--user", default=None, help="Lichess username")
    p.add_argument("--perf", default=None,
                   choices=["bullet", "blitz", "rapid", "classical"],
                   help="Game type (omit for all types)")
    p.add_argument("--max",  default=5000, type=int, help="Max games to download")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    username = args.user or input("Lichess username: ").strip()
    if not username:
        print("Username required.")
        sys.exit(1)

    if args.perf:
        perf = args.perf
    else:
        print("Game type: [1] bullet  [2] blitz  [3] rapid  [4] classical  [5] all")
        choice = input("Choose (1-5) [2]: ").strip() or "2"
        perf = PERF_TYPES.get(choice, None)   # None = all types

    download(username, perf, args.max)
