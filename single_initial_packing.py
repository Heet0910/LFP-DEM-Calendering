"""single_initial_packing.py — Publication-quality single panel of initial settled packing."""
from PIL import Image, ImageDraw, ImageFont
import os

SRC = r"c:\Users\heetv\Desktop\FINAL THIESES\relaxed bonds.png"
OUT = r"c:\Users\heetv\Desktop\VS Code\figures_ptfe\Fig_initial_settled_packing.png"

def font(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

SCALE   = 3
IMG_H   = 600 * SCALE
MARGIN  = 40  * SCALE
TITLE_H = 70  * SCALE
LEG_H   = 80  * SCALE
CAP_H   = 55  * SCALE
DARK    = (13, 30, 94)
GRAY    = (70, 70, 70)

# Load & resize OVITO image
raw = Image.open(SRC).convert("RGBA")
w0, h0 = raw.size
IMG_W = int(w0 * IMG_H / h0)
img = raw.resize((IMG_W, IMG_H), Image.LANCZOS)

total_w = IMG_W + 2 * MARGIN
total_h = MARGIN + TITLE_H + IMG_H + LEG_H + CAP_H + MARGIN

canvas = Image.new("RGB", (total_w, total_h), "white")
draw   = ImageDraw.Draw(canvas)

F_TITLE = font("C:/Windows/Fonts/calibrib.ttf", 38 * SCALE // 3)
F_BODY  = font("C:/Windows/Fonts/calibri.ttf",  22 * SCALE // 3)
F_LEG   = font("C:/Windows/Fonts/calibri.ttf",  21 * SCALE // 3)
F_CAP   = font("C:/Windows/Fonts/calibri.ttf",  18 * SCALE // 3)

# ── Title ──
draw.text((total_w // 2, MARGIN + TITLE_H // 2),
          "Initial Settled Packing — 3-Phase DEM Electrode",
          font=F_TITLE, fill=DARK, anchor="mm")

# ── Separator ──
draw.line([(MARGIN, MARGIN + TITLE_H), (total_w - MARGIN, MARGIN + TITLE_H)],
          fill=(200, 200, 200), width=SCALE)

# ── Paste image ──
y_img = MARGIN + TITLE_H
bg = Image.new("RGB", img.size, "white")
bg.paste(img, mask=img.split()[3])
canvas.paste(bg, (MARGIN, y_img))
draw.rectangle([MARGIN, y_img, MARGIN + IMG_W, y_img + IMG_H],
               outline=(160,160,160), width=SCALE)

# ── Legend ──
y_leg = y_img + IMG_H
draw.line([(MARGIN, y_leg), (total_w - MARGIN, y_leg)], fill=(200,200,200), width=SCALE)

items = [
    ((210, 60,  60),  "LFP Active Material (AM)  —  r = 1.2–1.8 µm,  ρ = 3.60 g/cm³"),
    (( 50,100, 205),  "Carbon Black (CB)  —  r = 0.75 µm,  ρ = 1.95 g/cm³"),
    ((215,195,  30),  "PTFE Binder  —  r = 0.70 µm,  ρ = 2.17 g/cm³"),
]
sw  = 22 * SCALE // 3
gap = 18 * SCALE // 3
y0  = y_leg + LEG_H // 2 - (len(items) * (sw + gap)) // 2

for col, txt in items:
    x0 = MARGIN + 20 * SCALE // 3
    draw.rounded_rectangle([x0, y0, x0+sw, y0+sw], radius=3, fill=col, outline=(80,80,80))
    draw.text((x0 + sw + 10 * SCALE // 3, y0 + sw // 2), txt,
              font=F_LEG, fill=(40,40,40), anchor="lm")
    y0 += sw + gap

# ── Stats box (top-right of image) ──
stats = [
    "Formulation: LFP 94 : CB 3 : PTFE 3  (wt%)",
    "N_total = 1318  (644 AM + 320 CB + 354 PTFE)",
    "Domain: 18 × 18 µm  |  Bed height ≈ 58.6 µm",
    "Initial porosity ε₀ ≈ 0.390",
]
bx0 = MARGIN + IMG_W - 420 * SCALE // 3
by0 = y_img + 8 * SCALE // 3
bx1 = MARGIN + IMG_W - 4 * SCALE // 3
by1 = by0 + (len(stats) * 26 + 14) * SCALE // 3
draw.rounded_rectangle([bx0, by0, bx1, by1], radius=6 * SCALE // 3,
                        fill=(255,255,255,220), outline=(150,150,150))
ty = by0 + 8 * SCALE // 3
for s in stats:
    draw.text((bx0 + 8 * SCALE // 3, ty), s, font=F_BODY, fill=(40,40,40))
    ty += 26 * SCALE // 3

# ── Figure caption ──
y_cap = y_leg + LEG_H
draw.line([(MARGIN, y_cap), (total_w-MARGIN, y_cap)], fill=(200,200,200), width=SCALE)
cap = ("Figure 3.1 — Initial gravity-settled 3-phase DEM electrode structure before calendering. "
       "Red spheres = LFP active material; blue nodes = carbon black aggregates; "
       "yellow nodes = PTFE binder. Simulation: CR = 0% (pre-compression), areal loading = 11.36 mg/cm².")
draw.text((total_w // 2, y_cap + CAP_H // 2), cap,
          font=F_CAP, fill=GRAY, anchor="mm")

canvas.save(OUT, dpi=(300, 300))
print(f"Saved: {OUT}  ({os.path.getsize(OUT)//1024} KB)  {canvas.size}")
