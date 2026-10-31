"""Shared plotting style, panel export (one PDF and one editable SVG per plot) and SVG composition."""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, PURPLE = '#2a78d6', '#eb6834', '#1baf7a', '#8a5fd1'   # categorical, fixed order
INK, INK2, MUTED, GRID = '#1f1e1c', '#52514e', '#8a8984', '#d9d8d4'
SURFACE = '#ffffff'
MM = 1 / 25.4

def set_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.2,
        'legend.frameon': False, 'figure.dpi': 150, 'savefig.transparent': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    })

def panel(w_mm=58, h_mm=44, **kw):
    kw.setdefault('layout', 'constrained')
    fig, ax = plt.subplots(figsize=(w_mm * MM, h_mm * MM), **kw)
    return fig, ax

def save_panel(fig, name, outdir='../fig/panels'):
    """Save one plot as <name>.pdf (for the record) and <name>.svg (editable text, for composition)."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + '.pdf'), bbox_inches=None)
    fig.savefig(os.path.join(outdir, name + '.svg'), bbox_inches=None)
    plt.close(fig)
    return os.path.join(outdir, name + '.svg')

def compose(out_svg, width_mm, height_mm, panels, labels=(), texts=(), render=True):
    """Compose panel SVGs into one editable SVG figure.

    panels: list of (svg_path, x_mm, y_mm) -- panels are placed at their native size (mm).
    labels: list of (letter, x_mm, y_mm) panel letters.  texts: list of (string, x_mm, y_mm, size_pt, weight).
    If render, also writes <out>.pdf and <out>.png next to the SVG (via cairosvg) for the manuscript.
    """
    import svgutils.transform as st
    pt = 72 / 25.4                                                       # matplotlib SVGs are in pt
    fig = st.SVGFigure(f'{width_mm}mm', f'{height_mm}mm')
    fig.root.set('viewBox', f'0 0 {width_mm * pt:.2f} {height_mm * pt:.2f}')
    elems = []
    bg = st.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="{width_mm * pt:.2f}" '
                       f'height="{height_mm * pt:.2f}" fill="#ffffff"/></svg>').getroot()
    elems.append(bg)
    for path, x, y in panels:
        root = st.fromfile(path).getroot(); root.moveto(x * pt, y * pt); elems.append(root)
    for letter, x, y in labels:
        elems.append(st.TextElement(x * pt, y * pt, letter, size=9, weight='bold', font='Arial, Helvetica, sans-serif'))
    for s_, x, y, size, weight in texts:
        elems.append(st.TextElement(x * pt, y * pt, s_, size=size, weight=weight, font='Arial, Helvetica, sans-serif'))
    fig.append(elems)
    fig.save(out_svg)
    if render:
        render_svg(out_svg, width_mm)
    return out_svg


def render_svg(svg, width_mm=None, formats=('pdf', 'png')):
    """Render an SVG to PDF/PNG with cairosvg. The rendering copy uses DejaVu Sans, which carries all glyphs used
    here; the SVG itself keeps Arial/Helvetica for editing."""
    import cairosvg, re
    txt = open(svg).read()
    txt = re.sub(r"font-family: '[^;\"]*", "font-family: 'DejaVu Sans', sans-serif", txt)
    txt = txt.replace('font-family="Arial, Helvetica, sans-serif"', 'font-family="DejaVu Sans"')
    for fmt in formats:
        out = svg[:-4] + '.' + fmt
        if fmt == 'pdf':
            cairosvg.svg2pdf(bytestring=txt.encode(), write_to=out)
        else:
            kw = {'output_width': int(width_mm / 25.4 * 300)} if width_mm else {'scale': 3}
            cairosvg.svg2png(bytestring=txt.encode(), write_to=out, **kw)
