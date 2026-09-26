#!/usr/bin/env python3
"""Draw the stat graphics from made-up numbers, so the theme can be previewed.

scripts/generate_stats.py wants a GITHUB_TOKEN and a real account; this wants
neither. It builds a fixed sample summary and pushes it through the very same
drawing functions, which is useful twice over:

  * after editing LIGHT / DARK in generate_stats.py, to see the new palette
    without waiting for the nightly run;
  * on a brand-new profile repository, so the four SVGs the README points at
    exist and render from the first commit instead of 404ing until the first
    workflow run.

Nothing here is real, and the first scheduled (or manually dispatched) run
overwrites all four files with the true figures. portrait.svg is never touched:
that one is drawn from a photo, by scripts/make_portrait.py.

    python3 scripts/preview_stats.py [OUT_DIR]      # default: repository root
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import generate_stats as gen


def weeks_of(today):
    """53 whole weeks of dates, Sunday first — the API's `weekday` is 0=Sunday."""
    last_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
    first = last_sunday - timedelta(days=52 * 7)
    return [[first + timedelta(days=w * 7 + i) for i in range(7)]
            for w in range(53)]


def pattern(n):
    """A fixed pseudo-random walk of contribution counts — no `random`, so two
    runs on two machines draw byte-identical SVGs."""
    out, x = [], 7
    for i in range(n):
        x = (1103515245 * x + 12345) % (1 << 31)
        quiet = i % 11 in (3, 7)          # roughly two rest days in eleven
        out.append(0 if quiet else 1 + x % 12)
    for i in range(len(out) - 12, len(out)):
        out[i] = max(out[i], 1)           # leave a current streak standing
    return out


def sample():
    """A summary dict in exactly the shape summarise() returns."""
    today = datetime.now(timezone.utc).date()
    days = []
    for week in weeks_of(today):
        for i, day in enumerate(week):
            days.append(dict(date=day.isoformat(), weekday=i,
                             contributionCount=0))
    for day, count in zip(days, pattern(len(days))):
        day["contributionCount"] = count

    weekly = [sum(d["contributionCount"] for d in days[i:i + 7])
              for i in range(0, len(days), 7)]
    current, longest = gen.streaks(days)
    return dict(
        total=sum(weekly),
        active=sum(1 for d in days if d["contributionCount"] > 0),
        best_week=max(weekly),
        weekly=weekly,
        weeks=[days[i:i + 7] for i in range(0, len(days), 7)],
        current=current, longest=longest,
        by_size=[("Python", 412_800), ("TypeScript", 268_400),
                 ("Java", 141_900), ("JavaScript", 96_300), ("C", 41_200)],
        by_repo=[("Python", 6), ("TypeScript", 4), ("Java", 2),
                 ("JavaScript", 1), ("C", 1)])


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else root

    s = sample()
    files = {"stats.svg": gen.draw_stats(s), "streak.svg": gen.draw_streak(s),
             "langs.svg": gen.draw_langs(s), "year.svg": gen.draw_year(s)}
    for word in gen.HEADINGS:
        files[f"hd-{word.replace(' ', '-')}.svg"] = gen.draw_heading(word)

    changed = [n for n, svg in files.items()
               if gen.write(os.path.join(out_dir, n), svg)]
    print("sample data, not your account: " + (", ".join(sorted(changed))
                                               if changed else "nothing changed"))
    print("the next workflow run replaces these with the real figures")


if __name__ == "__main__":
    main()
