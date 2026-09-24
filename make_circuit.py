#!/usr/bin/env python3
"""Draw the circuit figure for the writeup.

The KiCad schematic (../lumen_thermo.kicad_sch) is built with label-on-pin connectivity so
that the exported netlist can be diffed against an intended connection map. That makes it
authoritative but unreadable as a figure - no wires are drawn. This emits a clean,
hand-laid-out version of the same circuit.

Run:  python3 make_circuit.py     ->  images/circuit.svg + images/circuit.png

The PNG is what the write-up embeds. It is rendered from the SVG with rsvg-convert if that is
installed; without it the SVG is still written and the existing PNG is left alone, with a warning.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images", "circuit.svg")

W, H = 980, 536
INK, MUTE, ACC = "#1a1a1a", "#777", "#c1440e"
FONT = "Helvetica, Arial, sans-serif"
p = []


def wire(x1, y1, x2, y2, color=INK, w=1.6):
    p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
             f'stroke-width="{w}" stroke-linecap="round"/>')


def poly(pts, color=INK, w=1.6):
    d = " ".join(f"{x},{y}" for x, y in pts)
    p.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{w}" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')


def dot(x, y, r=4.5, color=INK):
    p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')


def txt(x, y, s, size=13, color=INK, anchor="middle", bold=False, italic=False):
    st = f'font-family:{FONT};font-size:{size}px;fill:{color}'
    if bold:
        st += ";font-weight:600"
    if italic:
        st += ";font-style:italic"
    p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{st}">{s}</text>')


def res_h(x, y, label, sub=None, wd=54, ht=19):
    """Horizontal resistor centred at (x, y)."""
    p.append(f'<rect x="{x-wd/2}" y="{y-ht/2}" width="{wd}" height="{ht}" rx="2.5" '
             f'fill="#fff" stroke="{INK}" stroke-width="1.6"/>')
    txt(x, y - ht / 2 - 8, label, 13, INK, bold=True)
    if sub:
        txt(x, y + ht / 2 + 15, sub, 12, MUTE)


def res_v(x, y, label, sub=None, wd=19, ht=54):
    """Vertical resistor centred at (x, y)."""
    p.append(f'<rect x="{x-wd/2}" y="{y-ht/2}" width="{wd}" height="{ht}" rx="2.5" '
             f'fill="#fff" stroke="{INK}" stroke-width="1.6"/>')
    txt(x + wd / 2 + 8, y - 3, label, 13, INK, anchor="start", bold=True)
    if sub:
        txt(x + wd / 2 + 8, y + 13, sub, 12, MUTE, anchor="start")


def cap_v(x, y, label, sub=None, w=30, gap=7):
    """Vertical capacitor centred at (x, y); plates horizontal."""
    wire(x - w / 2, y - gap / 2, x + w / 2, y - gap / 2, INK, 2.4)
    wire(x - w / 2, y + gap / 2, x + w / 2, y + gap / 2, INK, 2.4)
    txt(x + w / 2 + 8, y - 1, label, 13, INK, anchor="start", bold=True)
    if sub:
        txt(x + w / 2 + 8, y + 15, sub, 12, MUTE, anchor="start")


def ground(x, y):
    for i, half in enumerate((11, 7, 3.5)):
        wire(x - half, y + i * 4.5, x + half, y + i * 4.5, INK, 2.0)


def amp(x, y, gain, h=46, w=52):
    """Triangle amplifier, input at left edge, output at right tip."""
    p.append(f'<polygon points="{x},{y-h/2} {x},{y+h/2} {x+w},{y}" fill="#fffbe9" '
             f'stroke="{INK}" stroke-width="1.6" stroke-linejoin="round"/>')
    txt(x + w * 0.36, y + 5, gain, 13, INK, bold=True)


def channel(y, tag, rsrc, rval, u, out_label, accent):
    """One front-end channel: Johnson source -> x11 -> x101 -> output."""
    x_src, x_a1, x_a2 = 96, 150, 246
    wire(x_src, y, x_a1, y)                       # source node into stage 1
    dot(x_src, y)
    wire(x_src, y, x_src, y + 34)                 # down through the source resistor
    res_v(x_src, y + 61, rsrc, rval)
    wire(x_src, y + 88, x_src, y + 104)
    ground(x_src, y + 104)
    amp(x_a1, y, "&#215;11")
    wire(x_a1 + 52, y, x_a2, y)
    amp(x_a2, y, "&#215;101")
    wire(x_a2 + 52, y, 334, y)
    txt(x_a1 + 26, y - 34, u, 12, MUTE)
    txt(x_src, y - 18, tag, 12, accent, bold=True)
    txt(298, y - 14, out_label, 12, MUTE)


# ---------------------------------------------------------------- frame
p.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

YA, YB = 118, 356
N1X, N2X = 452, 452

# ---- the two front ends
channel(YA, "thermal noise", "R11", "9.883 k&#8486;", "OPA2189 &#183; both halves", "N_OUT1", ACC)
channel(YB, "thermal noise", "R12", "9.911 k&#8486;", "OPA2189 &#183; both halves", "N_OUT2", ACC)

# ---- R1 / R2 into the nodes
res_h(384, YA, "R1", "4.69 k&#8486;")
wire(411, YA, N1X, YA)
res_h(384, YB, "R2", "4.68 k&#8486;")
wire(411, YB, N2X, YB)

# ---- the coupling resistor between the nodes
dot(N1X, YA); dot(N2X, YB)
wire(N1X, YA, N1X, YA + 92)
res_v(N1X, 237, "R3", "6.83 k&#8486;")
wire(N1X, 264, N1X, YB)

# ---- node 1: C1 to ground
wire(N1X, YA, 556, YA)
dot(556, YA)
wire(556, YA, 556, YA + 42)
cap_v(556, YA + 52, "C1", "23.50 nF")
wire(556, YA + 62, 556, YA + 86)
ground(556, YA + 86)

# ---- node 2: C2 and R4 to ground
wire(N2X, YB, 556, YB)
dot(556, YB)
wire(556, YB, 556, YB + 42)
cap_v(556, YB + 52, "C2", "23.51 nF")
wire(556, YB + 62, 556, YB + 86)
ground(556, YB + 86)
wire(556, YB, 660, YB)
dot(660, YB)
wire(660, YB, 660, YB + 34)
res_v(660, YB + 61, "R4", "9.90 k&#8486;")
wire(660, YB + 88, 660, YB + 104)
ground(660, YB + 104)

# ---- scope taps
for y, lab in ((YA, "v&#8321;"), (YB, "v&#8322;")):
    xt = 790 if y == YA else 790
    wire(556 if y == YA else 660, y, xt, y, MUTE, 1.4)
    dot(xt, y, 5, ACC)
    txt(xt + 16, y + 5, lab, 16, ACC, anchor="start", bold=True)
    txt(xt + 42, y + 5, "scope", 12, MUTE, anchor="start")

txt(N1X, YA - 22, "node 1", 13, INK, bold=True)
txt(N2X, YB - 22, "node 2", 13, INK, bold=True)

# ---- captions
txt(172, 36, "front end &#8212; two independent Johnson sources", 13, MUTE)
txt(600, 36, "the network whose G is being inverted", 13, MUTE)
wire(346, 20, 346, 474, "#ddd", 1.4)
txt(W / 2, 512, "&#931; = c &#183; G&#8315;&#185;   —   the covariance of v&#8321; and v&#8322; is the inverse of the conductance matrix", 13.5, INK, italic=True)

svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
       f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">\n' + "\n".join(p) + "\n</svg>\n")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(svg)

# The write-up embeds the PNG, so render it here rather than leaving it as an undocumented
# manual export that nothing in the repo can reproduce.
import shutil
import subprocess
_png = OUT[:-4] + ".png"
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-z", "2", "-o", _png, OUT], check=True)
    print(f"wrote {OUT} and {_png}")
else:
    print(f"wrote {OUT}; rsvg-convert not found, {_png} left as-is "
          f"(install librsvg to regenerate it)")
