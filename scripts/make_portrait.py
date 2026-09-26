#!/usr/bin/env python3
"""Turn a photo into portrait.svg — a self-typing ASCII portrait.

This is the generator that produced the portrait at the top of the README.
Run it once; it is not on a schedule, unlike scripts/generate_stats.py.

    pip install pillow numpy opencv-python-headless rembg onnxruntime
    python3 scripts/make_portrait.py photo.png --crop 400,110,910,790 --model u2net
    python3 scripts/embed_portrait_font.py      # inline the font, see below

The first run downloads a background-removal model, once. rembg's own default is
the 1 GB BRIA RMBG 2.0; `--model u2net` is the cheap way in at 176 MB, and the
ASCII ramp has only 13 brightness levels, so it does not need more than that.

Two things decide whether the output is any good, and neither is a parameter:

  * The photo. ASCII draws with shadow, not detail — about 13 brightness levels
    in total. You need side light (a window at ~45°, everything else off), a
    tight crop from chin to just above the hair, and real resolution. A 320px
    headshot fails: thin features like glasses frames are averaged away on
    downscale. Flat frontal light renders the face as a hole.
  * The darkening curve below. Without it the face comes out washed out and
    featureless — brows, glasses and lips all dissolve.

The grid bakes in an advance width of exactly 0.600 em (CHAR_W / FONT_SIZE), so
after generating, run scripts/embed_portrait_font.py to inline JetBrains Mono.
Otherwise a viewer whose default monospace is narrower — Consolas is ≈0.55 —
sees the portrait about 7% too narrow.

Motion is SMIL, because GitHub strips <script> from READMEs: each row is
revealed by a clipPath wipe with a cursor block riding its edge, staggered top
to bottom, frozen at the end so it prints once and stops.
"""
import argparse
import os
import sys
from PIL import Image

RAMP = " .`:-=+*cs#%@"     # bright/sparse -> dark/dense; leading space = blank
COLS = 90                  # below ~88 the face muddies; far above it dominates
CLAHE_CLIP = 3.0           # higher amplifies skin texture into noise
GAMMA = 1.0                # ramp mapping exponent
CURVE = 1.7                # the darkening curve — the difference-maker
ALPHA_MIN = 8              # rembg cutout floor — only near-fully-transparent
                           # pixels are lifted to paper white, so dark but
                           # opaque shirt/neck rows are never stripped out
CROP_BOTTOM = 0.0          # fraction to trim off the bottom (torso, chair)
                           # keep at 0.0 — bottom rows are part of the portrait
BOTTOM_PAD = 6             # extra SVG breathing room below the last row so
                           # descenders and the final wipe are never clipped
ROW_RATIO = 0.48           # monospace cells are about twice as tall as wide
CURVE_LOW = 1.35           # gentler curve for the lower third (neck/shirt)
KEEP_LOWER_FRAC = 0.30     # bottom fraction where rembg is bypassed
FADE_FRAC = 0.15           # bottom fraction faded linearly to paper
MIN_ROWS = 60              # floor on vertical rows; pad if the crop is short

FG_LIGHT = "#0077b5"       # LinkedIn blue — the accent, on GitHub light
FG_DARK = "#38bdf8"        # bright cyan — its dark-mode step, on #0d1117
CHAR_W = 7.74              # 0.600 em at FONT_SIZE — keep these in step
FONT_SIZE = 12.9
LINE_H = 15
ROW_DELAY = 0.09           # per-row stagger, seconds
FAMILY = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def prep(path, crop=None, model=None):
    """Cut out the background, even the local contrast, then darken.

    `model` is a rembg model name — "u2net" for the 176 MB one, "birefnet-portrait"
    for a sharper cut at ten times the download. None means rembg's own default.

    The four heavy imports sit here rather than at module scope, so the drawing
    half of this file — to_lines and build_svg — stays importable with nothing
    but the standard library. That is what lets a placeholder portrait be drawn
    on a machine with no rembg installed.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # rembg pulls in pymatting, whose numba kernels cache themselves inside
    # site-packages. Under the Windows Store build of Python that directory is
    # not writable — the JIT cache write raises FileNotFoundError and the import
    # dies before it ever reaches the model, which looks like a silent hang.
    # Numba is only used by rembg's optional alpha matting, which this script
    # does not enable, so turning the JIT off costs nothing.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from rembg import remove, new_session

    src = Image.open(path).convert("RGBA")
    if crop:
        src = src.crop(crop)

    session = new_session(model) if model else None
    cut = remove(src, session=session)
    kept = np.array(src.convert("L"))
    alpha = np.array(cut.split()[-1])

    # Composite onto white so everything outside the subject maps to the blank
    # end of the ramp. Skip this and the background fills with @ and %.
    white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    gray = np.array(Image.alpha_composite(white, cut).convert("L"))

    # Combine rembg's mask with the original frame: bypass the cutout on the
    # lower KEEP_LOWER_FRAC so dark shirt/neck pixels are never read as
    # background noise and stripped to transparent/white. Anchor rows come
    # from the original photo; only the upper region trusts rembg.
    h0, w0 = kept.shape[0], kept.shape[1]
    keep_y = int(h0 * (1.0 - KEEP_LOWER_FRAC))
    gray[keep_y:h0, :w0] = kept[keep_y:h0, :w0]

    gray = cv2.bilateralFilter(gray, 11, 50, 50)      # smooth skin, keep edges
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(8, 8)).apply(gray)

    # Darkening curve: harsh CURVE up top for feature definition, gentler
    # CURVE_LOW across the lower third so dark neck/shirt pixels map to
    # low-ramp texture (`.`, `:`, `-`) instead of crushing or washing out.
    norm = gray.astype(np.float32) / 255.0
    low_y = h0 - h0 // 3
    gray = np.vstack([255.0 * np.power(norm[:low_y], CURVE),
                      255.0 * np.power(norm[low_y:], CURVE_LOW)]).astype("uint8")

    # Only near-transparent matte goes to white outside the kept band; the
    # lower band stays anchored to the photo even if rembg flagged it.
    outside = np.ones_like(alpha, dtype=bool)
    outside[keep_y:h0, :] = False
    gray[outside & (alpha < ALPHA_MIN)] = 255

    # Feathered fade across the bottom FADE_FRAC: chest/shoulders dissolve
    # into paper naturally instead of stopping on a hard truncated edge.
    fade_n = max(int(h0 * FADE_FRAC), 1)
    fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)[:, None]
    fade = np.broadcast_to(fade, (fade_n, w0))
    tail = gray[h0 - fade_n:h0, :].astype(np.float32)
    gray[h0 - fade_n:h0, :] = (tail * fade + 255.0 * (1.0 - fade)).astype("uint8")
    return Image.fromarray(gray)


def to_lines(img, cols=COLS, gamma=GAMMA):
    w, h = img.size
    if CROP_BOTTOM:
        img = img.crop((0, 0, w, int(h * (1 - CROP_BOTTOM))))
        w, h = img.size

    rows = max(int(cols * (h / w) * ROW_RATIO), MIN_ROWS)
    img = img.resize((cols, rows), Image.LANCZOS)
    px = list(img.getdata())
    n = len(RAMP)

    out = []
    for r in range(rows):
        out.append("".join(
            RAMP[min(n - 1, int((1 - px[r * cols + c] / 255.0) ** gamma * n))]
            for c in range(cols)
        ).rstrip())

    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def build_svg(lines, cols=COLS):
    pad = 14
    width = int(cols * CHAR_W + pad * 2)
    height = len(lines) * LINE_H + pad * 2 + BOTTOM_PAD

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}" '
         f'font-family="{FAMILY}">',
         f'<style>.a{{fill:{FG_LIGHT}}}'
         f'@media(prefers-color-scheme:dark){{.a{{fill:{FG_DARK}}}}}</style>']

    for i, line in enumerate(lines):
        y = pad + i * LINE_H
        begin = f"{i * ROW_DELAY:.2f}s"
        end = f"{(i + 1) * ROW_DELAY:.2f}s"
        w = max(len(line), 1) * CHAR_W
        safe = (line.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))

        p.append(f'<clipPath id="c{i}"><rect x="{pad}" y="{y}" '
                 f'height="{LINE_H}" width="0">'
                 f'<animate attributeName="width" from="0" to="{w:.1f}" '
                 f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
                 f'</rect></clipPath>')
        p.append(f'<g clip-path="url(#c{i})"><text xml:space="preserve" '
                 f'x="{pad}" y="{y + 11.2:.1f}" class="a" '
                 f'font-size="{FONT_SIZE}">{safe}</text></g>')
        # the cursor: a small block riding the wipe edge, gone once the row lands
        p.append(f'<rect y="{y + 1}" width="6" height="12" class="a" '
                 f'opacity="0">'
                 f'<animate attributeName="x" from="{pad}" to="{pad + w:.1f}" '
                 f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
                 f'<set attributeName="opacity" to="0.8" begin="{begin}"/>'
                 f'<set attributeName="opacity" to="0" begin="{end}"/></rect>')

    p.append("</svg>")
    return "".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("out", nargs="?", default="portrait.svg")
    ap.add_argument("--crop", help="left,top,right,bottom, applied first — crop "
                                   "tight to the head so the whole grid goes to "
                                   "the face")
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--model", default=None,
                    help="rembg model for the cutout — 'u2net' is a 176 MB "
                         "download against rembg's 1 GB default")
    ap.add_argument("--preview", action="store_true",
                    help="print the ASCII to the terminal as well")
    args = ap.parse_args()

    crop = None
    if args.crop:
        parts = [int(v) for v in args.crop.split(",")]
        if len(parts) != 4:
            sys.exit("--crop needs four numbers: left,top,right,bottom")
        crop = tuple(parts)

    lines = to_lines(prep(args.photo, crop, args.model), cols=args.cols)
    if args.preview:
        print("\n".join(lines))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_svg(lines, cols=args.cols))
    print(f"wrote {args.out} — {len(lines)} rows, {args.cols} columns")
    print("next: python3 scripts/embed_portrait_font.py")


if __name__ == "__main__":
    main()
