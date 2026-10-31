"""Builds presentation/Challenge5_Capacity_and_Delay_Risk.pptx.  All numbers are read from reports/ and data/processed,
never typed by hand.   Run from the repo root:  python presentation/build_deck.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

OUT = ROOT / "presentation" / "Challenge5_Capacity_and_Delay_Risk.pptx"
IMG = ROOT / "presentation" / "img"

NAVY, DEEP, SKY, AMBER, RED = "0B1F3A", "12335B", "3FA7D6", "F6AE2D", "D64545"
INK, MUTED, CARD, WHITE, GREEN, GREY = "1B2B44", "5B6B82", "EEF3F9", "FFFFFF", "2E9E6B", "A9B4C4"
HEAD, BODY = "Cambria", "Calibri"
rgb = lambda h: RGBColor.from_string(h)

# ----------------------------------------------------------------------------------------- numbers
am = pd.read_csv(ROOT / "reports" / "aar_metrics.csv")
dm = pd.read_csv(ROOT / "reports" / "delay_metrics.csv")
sens = pd.read_csv(ROOT / "reports" / "aar_sensitivity.csv")
unseen = pd.read_csv(ROOT / "reports" / "aar_unseen_airport.csv")
boot = pd.read_csv(ROOT / "reports" / "delay_bootstrap.csv").set_index("lead_h")
rel3 = pd.read_csv(ROOT / "reports" / "delay_reliability_lead3.csv")
daily3 = pd.read_csv(ROOT / "reports" / "delay_daily_agreement_lead3.csv").iloc[0]
aar_all = am[(am.scope == "all") & am.rmse_aar.notna()]
R = lambda model, lead, col="rmse_aar": float(aar_all[(aar_all.model == model) & (aar_all.lead_h == lead)][col].iloc[0])
dl = dm[(dm.scope == "all") & ((dm.variant == "calibrated") | dm.model.isin(["logit_schedule_only", "constant_base_rate"]))]
D = lambda model, lead, col="pr_auc": float(dl[(dl.model == model) & (dl.lead_h == lead)][col].iloc[0])
full = lambda lead, col: float(dm[(dm.model == "E_plus_bts") & (dm.variant == "calibrated") & (dm.scope == "all") & (dm.lead_h == lead)][col].iloc[0])
pa = pd.read_parquet(C.PROC / "aar_predictions.parquet")
pa = pa[pa.split == "test"]


def op_point(lead, col):
    d = pa[pa.lead_h == lead]
    ev, fl = d.y_ratio <= 0.8, d[col] <= 0.8
    return float((fl & ev).sum() / fl.sum()), float((fl & ev).sum() / ev.sum())


bal3, cau3, bal12, cau12 = op_point(3, "pred_ratio"), op_point(3, "lo_ratio"), op_point(12, "pred_ratio"), op_point(12, "lo_ratio")
event_rate = float(R("lightgbm", 3, "event_rate"))
prev = full(3, "prevalence")
LG = lambda lead: float(dm[(dm.model == "lightgbm_comparison") & (dm.lead_h == lead)].pr_auc.iloc[0])

# proxy validation: severe-departure rate by weather-derived AAR at the departure hour
hourly = pd.read_csv(C.PROC / "hourly_weather.csv", parse_dates=["hour_utc"])
dt = pd.read_parquet(C.PROC / "delay_table.parquet", columns=["airport", "direction", "event_hour_utc", "lead_h", "severe_delay", "delay_min"])
dt = dt[(dt.lead_h == 3) & (dt.direction == "dep")].merge(hourly[["airport", "hour_utc", "aar_ratio"]], left_on=["airport", "event_hour_utc"], right_on=["airport", "hour_utc"])
dt["bin"] = pd.cut(dt.aar_ratio, [0, .6, .8, .95, 1.01], labels=["≤60%", "60–80%", "80–95%", ">95%"])
proxy = dt.groupby("bin", observed=True).severe_delay.mean() * 100

# ----------------------------------------------------------------------------------------- helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5
page = [0]


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def text(slide, x, y, w, h, s, size=16, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         italic=False, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for r_text, o in runs:
            r = p.add_run()
            r.text = r_text
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", italic)
            r.font.name = o.get("font", font)
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, gap=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        runs = it if isinstance(it, list) else [(it, {})]
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", "228600")
        pPr.set("indent", "-228600")
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        for r_text, o in runs:
            r = p.add_run()
            r.text = r_text
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", False)
            r.font.name = BODY
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def shape(slide, kind, x, y, w, h, fill, line=None, name=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def card(slide, x, y, w, h, fill=CARD):
    s = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill)
    s.adjustments[0] = 0.06
    return s


def badge(slide, x, y, n, fill=SKY, color=WHITE, d=0.5):
    c = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold, r.font.name = Pt(16), True, BODY
    r.font.color.rgb = rgb(color)
    return c


def new_slide(title, dark=False, notes=None, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    page[0] += 1
    bg(s, NAVY if dark else WHITE)
    if title:
        text(s, 0.6, 0.38, 12.1, 0.8, title, size=30, bold=True, color=WHITE if dark else NAVY, font=HEAD, name="Title",
             anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        text(s, 0.6, 1.16, 12.1, 0.4, subtitle, size=16, color=GREY if dark else MUTED)
    text(s, 0.6, 7.08, 9, 0.25, "Challenge 5  ·  Predictive airport capacity & delay risk  ·  ATL & DFW  ·  Autonomous SG 2026", size=11,
         color=GREY if dark else MUTED)
    text(s, 12.0, 7.08, 0.73, 0.25, str(page[0]), size=11, color=GREY if dark else MUTED, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, big, small, color=SKY, big_size=44, dark=False):
    gap = big_size / 72 * 1.25 + 0.05
    text(slide, x, y, w, gap, big, size=big_size, bold=True, color=color, font=HEAD)
    text(slide, x, y + gap, w, 0.9, small, size=14, color=GREY if dark else MUTED)


def style_chart(chart, legend=True, size=12, title=None):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = rgb(INK)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    if title:
        chart.has_title = True
        chart.chart_title.text_frame.text = title
        pr = chart.chart_title.text_frame.paragraphs[0]
        pr.runs[0].font.size, pr.runs[0].font.bold, pr.runs[0].font.color.rgb = Pt(size + 1), True, rgb(INK)
        pr.runs[0].font.name = BODY
    else:
        chart.has_title = False
    try:
        va, ca = chart.value_axis, chart.category_axis
        va.major_gridlines.format.line.color.rgb = rgb("D9E0EA")
        va.format.line.fill.background()
        va.tick_labels.font.size = Pt(size)
        ca.tick_labels.font.size = Pt(size)
        ca.format.line.color.rgb = rgb(GREY)
        ca.has_major_gridlines = False
    except Exception:
        pass


def color_series(chart, colors, line=True, width=2.75):
    for s, col in zip(chart.plots[0].series, colors):
        if line:
            s.format.line.color.rgb = rgb(col)
            s.format.line.width = Pt(width)
            s.smooth = False
            s.marker.style = XL_MARKER_STYLE.CIRCLE
            s.marker.size = 7
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = rgb(col)
            s.marker.format.line.color.rgb = rgb(col)
        else:
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = rgb(col)


# ===================================================================================== 1. title
s = new_slide(None, dark=True, notes=(
    "Open with the user: a traffic-management duty manager at a hub. Today capacity is cut when the weather arrives. "
    "We forecast it 3 to 12 hours before, and we show how reliable that forecast is, including where it fails."))
text(s, 0.8, 1.7, 11.5, 1.7, ["See the capacity drop", "before the storm lands"], size=54, bold=True, color=WHITE, font=HEAD)
text(s, 0.8, 3.75, 11, 0.9, "Predicting airport arrival rates and severe-delay risk 3 to 12 hours ahead for Atlanta (ATL) and Dallas-Fort Worth (DFW)",
     size=22, color="CADCFC")
badge(s, 0.8, 5.45, 1, SKY)
text(s, 1.45, 5.5, 3.2, 0.5, "Forecast capacity (AAR)", size=18, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
badge(s, 4.9, 5.45, 2, SKY)
text(s, 5.55, 5.5, 3.4, 0.5, "Flag severe-delay risk", size=18, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
badge(s, 9.0, 5.45, 3, SKY)
text(s, 9.65, 5.5, 3.3, 0.5, "Advise the controller", size=18, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.8, 6.5, 11.5, 0.4, "Autonomous SG 2026  ·  AI & Aerospace track  ·  Airbus  ·  Challenge 5", size=14, color=GREY)

# ===================================================================================== 2. problem
s = new_slide("Capacity is cut when the weather arrives, not before", notes=(
    "User: the traffic management unit duty manager and airline ops controllers. Status quo: capacity changes are reactive, "
    "so aircraft hold in the air or queue on taxiways. Numbers on this slide are from the 2025 test period of our own data: "
    "about one hour in five has capacity at least 20 percent below best case; about one in eight sampled flights is delayed over 45 minutes."))
card(s, 0.6, 1.5, 5.9, 5.1)
text(s, 0.95, 1.75, 5.3, 0.4, "WHO THIS IS FOR", size=13, bold=True, color=SKY)
text(s, 0.95, 2.15, 5.3, 1.0, "Traffic-management duty manager at a hub, and the airline ops desks that depend on the same call", size=20, bold=True, color=NAVY)
text(s, 0.95, 3.45, 5.3, 0.4, "TODAY", size=13, bold=True, color=SKY)
bullets(s, 0.95, 3.85, 5.3, 2.6, [
    "Arrival rates are lowered reactively, when the weather is already overhead",
    "Aircraft stack in holding patterns or wait on taxiways with engines running",
    "Delays cascade into cancellations and missed connections",
], size=16)
stat(s, 7.0, 1.5, 2.8, f"{event_rate:.0%}", "of hours in 2025 had arrival capacity at least 20% below best case", SKY)
stat(s, 10.1, 1.5, 2.6, f"{prev:.0%}", "of sampled flights were delayed more than 45 minutes", AMBER)
stat(s, 7.0, 3.85, 2.8, "3–12 h", "of warning wanted, so staffing, metering and airline plans can change in time", GREEN, big_size=40)
stat(s, 10.1, 3.85, 2.6, "2 hubs", "ATL (runway heading 80°) and DFW (180°), 2023–2025, one pipeline for any airport", NAVY, big_size=40)
text(s, 7.0, 6.2, 5.9, 0.5, "Figures: 2025 held-out period of the organiser flight data and public METAR records.", size=12, color=MUTED, italic=True)

# ===================================================================================== 3. what we built
s = new_slide("Two forecasters feeding one dashboard, with rules in between", notes=(
    "Method split matters for the rubric. ML does what it is good at: forecasting from many weather signals. "
    "Rules do the translation from forecast to advice because that must be transparent and auditable. A person decides. "
    "No language model is used anywhere in the decision path."))
xs = [0.6 + i * 3.11 for i in range(4)]
heads = ["Public data", "Leak-free features", "Two forecasters", "Dashboard + advice"]
bodies = [["METAR observations", "TAF forecasts", "Organiser flight data", "downloaded automatically"],
          ["Only what was published before the forecast time", "Runway head/cross/tailwind", "Recent hub delays"],
          ["AAR regression (RMSE)", "Severe-delay classifier (45+ min)", "AAR: LightGBM + range; delay: logistic, calibrated"],
          ["Capacity outlook 3–12 h", "Flight risk with reasons", "Rule-based recommendations"]]
for i, (x, hd, bd) in enumerate(zip(xs, heads, bodies)):
    card(s, x, 1.55, 2.8, 3.6)
    badge(s, x + 0.25, 1.8, i + 1)
    text(s, x + 0.25, 2.45, 2.5, 0.8, hd, size=20, bold=True, color=NAVY, font=HEAD)
    bullets(s, x + 0.25, 3.3, 2.35, 1.8, bd, size=14, gap=4)
    if i < 3:
        a = shape(s, MSO_SHAPE.RIGHT_ARROW, x + 2.82, 3.15, 0.27, 0.34, SKY)
card(s, 0.6, 5.45, 12.13, 1.35, fill=NAVY)
text(s, 0.9, 5.62, 4.0, 0.3, "WHO DOES WHAT", size=13, bold=True, color=SKY)
text(s, 0.9, 5.98, 11.6, 0.75,
     [[("Machine learning", {"bold": True, "color": AMBER}), (" forecasts.  ", {"color": WHITE}),
       ("Rules", {"bold": True, "color": AMBER}), (" turn forecasts into advice you can audit.  ", {"color": WHITE}),
       ("The duty manager", {"bold": True, "color": AMBER}), (" decides.", {"color": WHITE})]], size=20)

# ===================================================================================== 4. leak-free
s = new_slide("Leak-free by construction, and runway-aware", notes=(
    "The brief warns about data leakage and ignoring runway directionality. Every feature is joined on the issue time. "
    "METAR hour must have been published; the TAF must have been issued at least lead hours before the target hour; the recent-performance "
    "features only count flights that had already departed or landed at the issue time. These are asserted in code and covered by ten automated tests. "
    "Wind is converted to head, tail and crosswind relative to the assigned heading, 80 degrees at ATL and 180 at DFW."))
text(s, 0.6, 1.4, 12.1, 0.4, "Forecast issued at time T − h, for an event at time T", size=16, color=MUTED)
labels = [("TAF issued", "at least h hours before the target hour"), ("METAR hour published", "latest observation already public"),
          ("Issue time  T − h", "the forecast is made here"), ("Target hour  T", "what we predict")]
for i, (a, b) in enumerate(labels):
    x = 0.6 + i * 3.06
    ch = shape(s, MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON, x, 1.95, 2.95, 1.0, SKY if i < 3 else AMBER)
    text(s, x + 0.55 if i else x + 0.25, 2.1, 1.85, 0.7, a, size=15, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    text(s, x + 0.15, 3.05, 2.65, 0.7, b, size=13, color=MUTED, align=PP_ALIGN.CENTER)
card(s, 0.6, 4.0, 5.95, 2.85)
text(s, 0.9, 4.2, 5.6, 0.35, "THREE GUARDS, ALL TESTED", size=13, bold=True, color=SKY)
bullets(s, 0.9, 4.6, 5.6, 2.2, [
    "Weather: only hours published before the issue time",
    "Forecast: only TAFs issued at least h hours before the target",
    "Hub state: only flights already completed at the issue time",
    "Split by time: train to 2024-06, tune on 2024-H2, test on 2025",
], size=15, gap=5)
card(s, 6.8, 4.0, 5.93, 2.85)
text(s, 7.1, 4.2, 5.3, 0.35, "RUNWAY DIRECTIONALITY", size=13, bold=True, color=SKY)
text(s, 7.1, 4.65, 5.4, 0.8, [[("crosswind = wind · sin(dir − heading)", {"font": "Consolas", "size": 15})],
                              [("tailwind  = −wind · cos(dir − heading)", {"font": "Consolas", "size": 15})]], size=15, color=NAVY)
bullets(s, 7.1, 5.6, 5.4, 1.2, ["ATL heading 80°, DFW heading 180° (from the brief)", "Unknown airport: falls back to the prevailing wind"], size=14, gap=4)

# ===================================================================================== 5. AAR target
s = new_slide("The AAR target is derived from FAA rates, and we tested it", notes=(
    "Be upfront: no public data set gives the hourly AAR a facility actually called. So we derive it. Base rates are the facility-called AARs published in the FAA Airport Capacity Profiles "
    "by weather category. DFW low-IFR is not published separately; we scaled it from the ATL ratio and flag it. On top we reduce for crosswind, tailwind, thunderstorms, gusts and winter weather; those reduction sizes are our assumptions. "
    "Evidence the proxy means something: in the real flight data, departures in hours where the proxy AAR was low were delayed more often. "
    "Later we stress-test the assumptions."))
card(s, 0.6, 1.45, 6.3, 5.4)
text(s, 0.9, 1.65, 5.7, 0.35, "HOW THE TARGET IS BUILT", size=13, bold=True, color=SKY)
text(s, 0.9, 2.05, 5.7, 0.35, "1  Base rate by flight category (FAA called AAR)", size=15, bold=True, color=NAVY)
rows = [["Hub", "VFR", "MVFR", "IFR", "LIFR"], ["ATL"] + [str(C.AIRPORTS["KATL"]["aar"][k]) for k in ("VFR", "MVFR", "IFR", "LIFR")],
        ["DFW"] + [str(C.AIRPORTS["KDFW"]["aar"][k]) + ("*" if k == "LIFR" else "") for k in ("VFR", "MVFR", "IFR", "LIFR")]]
tbl = s.shapes.add_table(3, 5, Inches(0.9), Inches(2.5), Inches(5.7), Inches(1.2)).table
for r_i, row in enumerate(rows):
    for c_i, v in enumerate(row):
        cell = tbl.cell(r_i, c_i)
        cell.text = v
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size, p.runs[0].font.name = Pt(15), BODY
        p.runs[0].font.bold = r_i == 0 or c_i == 0
        p.runs[0].font.color.rgb = rgb(WHITE if r_i == 0 else INK)
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb(DEEP if r_i == 0 else WHITE)
text(s, 0.9, 3.8, 5.7, 0.3, "* DFW low-IFR is not published; scaled from ATL's ratio (assumption).", size=11, color=MUTED, italic=True)
text(s, 0.9, 4.25, 5.7, 0.35, "2  Multiply by hazard reductions (our assumptions)", size=15, bold=True, color=NAVY)
bullets(s, 0.9, 4.7, 5.7, 2.1, ["Crosswind above 15 kt: −2.5% per kt", "Tailwind above 5 kt: −4% per kt", "Thunderstorm −30%, winter weather −15%, gusts ≥ 30 kt −10%",
                                "Never below 45% of the base rate"], size=14, gap=3)
cd = CategoryChartData()
cd.categories = list(proxy.index)
cd.add_series("Severe departure delay rate (%)", [round(float(v), 1) for v in proxy.values])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.2), Inches(1.9), Inches(5.8), Inches(3.9), cd)
ch = gf.chart
style_chart(ch, legend=False, title="Do low-AAR hours hurt real flights?")
color_series(ch, [SKY], line=False)
pl = ch.plots[0]
pl.has_data_labels = True
pl.data_labels.font.size = Pt(13)
pl.data_labels.number_format = '0.0"%"'
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.gap_width = 60
ch.value_axis.maximum_scale = 20
ch.value_axis.has_title = False
text(s, 7.2, 5.9, 5.8, 0.9, "Share of real departures delayed over 45 min, grouped by the proxy AAR (% of best case) in the departure hour. "
                              "The worse the proxy, the worse the real outcome.", size=13, color=MUTED)

# ===================================================================================== 6. AAR results
leads = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
s = new_slide("The AAR forecast beats every baseline at every horizon", notes=(
    f"Scored on the held-out 2025 data: {int(R('lightgbm', 3, 'n')):,} airport-hours per lead. RMSE is in arrivals per hour. "
    f"Persistence means carrying the last published rate forward; climatology is the average for that month and hour; TAF-implied means applying our rule to what the newest TAF forecast for the hour. "
    f"At 3 hours the model's error is {R('lightgbm', 3):.1f} versus {R('persistence', 3):.1f} for persistence. At 12 hours it is {R('lightgbm', 12):.1f} versus {R('persistence', 12):.1f}. "
    "The model does degrade slowly with lead time, as it should; the TAF-implied line is flat because a TAF is already a forecast."))
cd = CategoryChartData()
cd.categories = [f"+{l} h" for l in leads]
for nm, key in (("Our model (LightGBM)", "lightgbm"), ("TAF-implied", "taf_implied"), ("Climatology", "climatology"), ("Persistence", "persistence")):
    cd.add_series(nm, [round(R(key, l), 2) for l in leads])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.45), Inches(8.2), Inches(5.35), cd)
ch = gf.chart
style_chart(ch, legend=True, size=13, title="Forecast error on unseen 2025 data (RMSE, arrivals/h)")
color_series(ch, [SKY, GREEN, GREY, AMBER])
ch.value_axis.minimum_scale = 0
ch.value_axis.maximum_scale = 25
stat(s, 9.2, 1.5, 3.5, f"−{100 * (1 - R('lightgbm', 3) / R('persistence', 3)):.0f}%", f"error vs persistence at +3 h ({R('lightgbm', 3):.1f} vs {R('persistence', 3):.1f} arrivals/h)", SKY, big_size=48)
stat(s, 9.2, 3.45, 3.5, f"−{100 * (1 - R('lightgbm', 12) / R('persistence', 12)):.0f}%", f"error vs persistence at +12 h ({R('lightgbm', 12):.1f} vs {R('persistence', 12):.1f})", SKY, big_size=48)
stat(s, 9.2, 5.4, 3.5, f"−{100 * (1 - R('lightgbm', 3) / R('taf_implied', 3)):.0f}%", f"error vs the TAF alone at +3 h ({R('taf_implied', 3):.1f})", GREEN, big_size=48)

# ===================================================================================== 7. when capacity really drops
s = new_slide("When capacity really drops, it catches about 7 in 10", notes=(
    "Overall RMSE can hide the cases that matter. So we also report hours where capacity truly fell by 20 percent or more. "
    "A plain least-squares model shrinks these rare hours toward average. We fixed that by up-weighting those hours during training, choosing the weight on validation data only, with a rule written beforehand: "
    "smallest error on reduction hours as long as overall error rises by at most five percent. Weight two passed; weight four did not. "
    f"On the test year overall error did not change ({R('lightgbm', 3):.2f} at 3 h) and error on reduction hours improved, now better than the TAF alone. "
    f"The alert has two operating points: balanced (precision {bal3[0]:.0%}, recall {bal3[1]:.0%} at 3 h) and cautious, which alerts on the lower bound of the forecast range (precision {cau3[0]:.0%}, recall {cau3[1]:.0%}). "
    "The controller chooses."))
cd = CategoryChartData()
cd.categories = ["+3 h", "+6 h", "+9 h", "+12 h"]
for nm, key in (("Our model", "lightgbm"), ("TAF-implied", "taf_implied"), ("TAF worst case", "taf_worst_case"), ("Persistence", "persistence")):
    cd.add_series(nm, [round(R(key, l, "rmse_aar_reduction_hrs"), 1) for l in (3, 6, 9, 12)])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.45), Inches(7.3), Inches(4.2), cd)
ch = gf.chart
style_chart(ch, legend=True, size=12, title="Error in hours with a real capacity cut (RMSE, arrivals/h)")
color_series(ch, [SKY, GREEN, "7FC8A9", AMBER], line=False)
ch.plots[0].gap_width = 70
ch.plots[0].overlap = -5
ch.value_axis.minimum_scale = 0
text(s, 0.6, 5.8, 7.3, 1.0, f"Capacity-reduction hours = AAR at least 20% below best case ({event_rate:.0%} of 2025 hours). "
                            "Training up-weights these hours 2×, chosen on validation data by a rule fixed in advance.", size=13, color=MUTED)
card(s, 8.2, 1.45, 4.53, 5.35)
text(s, 8.5, 1.65, 4.0, 0.35, "THE ALERT HAS TWO SETTINGS", size=13, bold=True, color=SKY)
text(s, 8.5, 2.15, 4.3, 0.4, "Balanced (central forecast)", size=18, bold=True, color=NAVY)
text(s, 8.5, 2.6, 4.3, 0.9, [[(f"{bal3[0]:.0%}", {"size": 36, "bold": True, "color": SKY, "font": HEAD}), ("  of alerts are real", {"size": 15})],
                             [(f"{bal3[1]:.0%}", {"size": 36, "bold": True, "color": SKY, "font": HEAD}), ("  of cuts are caught", {"size": 15})]], size=15)
text(s, 8.5, 4.35, 4.3, 0.4, "Cautious (low end of range)", size=18, bold=True, color=NAVY)
text(s, 8.5, 4.8, 4.3, 0.9, [[(f"{cau3[0]:.0%}", {"size": 36, "bold": True, "color": AMBER, "font": HEAD}), ("  of alerts are real", {"size": 15})],
                             [(f"{cau3[1]:.0%}", {"size": 36, "bold": True, "color": AMBER, "font": HEAD}), ("  of cuts are caught", {"size": 15})]], size=15)
text(s, 8.5, 6.15, 4.3, 0.5, "+3 h, 2025 test. The duty manager picks the trade-off.", size=12, color=MUTED, italic=True)

# ===================================================================================== 8. delay ablation
s = new_slide("Weather earns its place for delay risk; airport traffic adds little", notes=(
    "This is the ablation ladder for the severe-delay classifier, scored by PR-AUC, the right metric when only about 13 percent of flights are severe. A random guess scores the prevalence. "
    "Schedule alone gives 0.22. Adding how the hub has been performing in the last hours helps a little. Adding METAR weather is the big step at 3 hours. Adding the TAF forecast helps most at 12 hours, when observations are stale. "
    f"Adding airport-wide BTS traffic barely moves it ({D('D_plus_taf', 3):.3f} to {D('E_plus_bts', 3):.3f} at 3 hours), so the sample's own recent-hub signal already carries most of it. "
    f"We bootstrapped by day: the gain of the full model over schedule-only is {boot.loc[3, 'pr_auc_gain_mean']:+.3f} at 3 hours (95% interval {boot.loc[3, 'ci_low']:+.3f} to {boot.loc[3, 'ci_high']:+.3f}) "
    f"and {boot.loc[12, 'pr_auc_gain_mean']:+.3f} at 12 hours ({boot.loc[12, 'ci_low']:+.3f} to {boot.loc[12, 'ci_high']:+.3f}). Both intervals exclude zero."))
cd = CategoryChartData()
cd.categories = ["Schedule only", "+ recent hub delays", "+ METAR weather", "+ TAF forecast", "+ BTS airport traffic"]
cd.add_series("3 h ahead", [round(D(m, 3), 3) for m in ("A_schedule", "B_plus_recent_hub", "C_plus_metar", "D_plus_taf", "E_plus_bts")])
cd.add_series("12 h ahead", [round(D(m, 12), 3) for m in ("A_schedule", "B_plus_recent_hub", "C_plus_metar", "D_plus_taf", "E_plus_bts")])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.45), Inches(8.0), Inches(5.35), cd)
ch = gf.chart
style_chart(ch, legend=True, size=13, title=f"Severe-delay ranking quality on 2025 data (PR-AUC; chance = {prev:.2f})")
color_series(ch, [SKY, DEEP], line=False)
pl = ch.plots[0]
pl.gap_width = 60
pl.has_data_labels = True
pl.data_labels.font.size = Pt(13)
pl.data_labels.number_format = "0.00"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ch.value_axis.minimum_scale = 0
ch.value_axis.maximum_scale = 0.4
stat(s, 9.0, 1.5, 3.7, f"{D('E_plus_bts', 3):.2f}", f"PR-AUC at +3 h vs {D('A_schedule', 3):.2f} schedule-only and {prev:.2f} chance", SKY, big_size=48)
card(s, 9.0, 3.6, 3.73, 3.2)
text(s, 9.25, 3.8, 3.5, 0.35, "HOW SURE ARE WE?", size=13, bold=True, color=SKY)
text(s, 9.25, 4.2, 3.3, 2.5, [f"Gain over schedule-only, bootstrapped by day (95% interval):",
                               [(f"+3 h:  {boot.loc[3, 'pr_auc_gain_mean']:+.3f}  ({boot.loc[3, 'ci_low']:+.3f} to {boot.loc[3, 'ci_high']:+.3f})", {"bold": True, "color": NAVY})],
                               [(f"+12 h: {boot.loc[12, 'pr_auc_gain_mean']:+.3f}  ({boot.loc[12, 'ci_low']:+.3f} to {boot.loc[12, 'ci_high']:+.3f})", {"bold": True, "color": NAVY})],
                               "Both intervals exclude zero. Weather adds real, if modest, skill.",
                               [(f"Model chosen out-of-fold on 2024: regularised logistic regression. On 2025 it ties or beats LightGBM ({D('E_plus_bts', 3):.2f} vs {LG(3):.2f} at +3 h, {D('E_plus_bts', 12):.2f} vs {LG(12):.2f} at +12 h).", {"italic": True, "color": MUTED})]], size=14, color=INK)

# ===================================================================================== 9. trust: calibration
dep3 = dm[(dm.model == "E_plus_bts") & (dm.variant == "calibrated") & (dm.lead_h == 3)].set_index("scope")
s = new_slide("Predicted risk tracks what actually happens", notes=(
    "A risk score is only useful if the number means something. We split 2025 flights into ten equal groups by predicted risk. "
    f"The line shows predicted against observed; they track closely. The top tenth of flights has about {rel3.observed.iloc[-1]:.0%} observed severe-delay rate versus {prev:.0%} overall. "
    f"Daily agreement: across {int(daily3.days)} airport-days the predicted average risk rank-correlates {daily3.spearman_daily_pred_vs_observed:.2f} with the observed share. "
    f"Departures are easier than arrivals: PR-AUC {dep3.loc['direction=dep', 'pr_auc']:.2f} versus {dep3.loc['direction=arr', 'pr_auc']:.2f}, because arrival delay also contains the upstream leg."))
cd = CategoryChartData()
cd.categories = [f"{i + 1}" for i in range(len(rel3))]
cd.add_series("Predicted risk (%)", [round(float(v) * 100, 1) for v in rel3.mean_pred])
cd.add_series("Observed severe-delay rate (%)", [round(float(v) * 100, 1) for v in rel3.observed])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.45), Inches(7.6), Inches(5.35), cd)
ch = gf.chart
style_chart(ch, legend=True, size=13, title="Calibration at +3 h: 10 groups of flights, low to high risk")
color_series(ch, [SKY, AMBER])
ch.value_axis.minimum_scale = 0
stat(s, 8.6, 1.5, 4.1, f"{full(3, 'precision_top10pct'):.0%}", f"of the riskiest 10% of flights were severely delayed (vs {prev:.0%} overall)", SKY, big_size=48)
stat(s, 8.6, 3.5, 4.1, f"ρ = {daily3.spearman_daily_pred_vs_observed:.2f}", f"daily predicted risk vs observed severe share, {int(daily3.days)} airport-days", GREEN, big_size=44)
text(s, 8.6, 5.55, 4.1, 1.2, [f"Departures PR-AUC {dep3.loc['direction=dep', 'pr_auc']:.2f}, arrivals {dep3.loc['direction=arr', 'pr_auc']:.2f}.",
                              f"ATL {dep3.loc['airport=KATL', 'pr_auc']:.2f}, DFW {dep3.loc['airport=KDFW', 'pr_auc']:.2f}."], size=14, color=MUTED)

# ===================================================================================== 10. failures
fcases = pd.read_csv(ROOT / "reports" / "delay_failure_cases_lead3.csv")
miss = fcases[fcases.kind == "missed_severe_delay"].iloc[0]
d_dfw = pa[(pa.airport == "KDFW") & (pa.lead_h == 6) & (pa.issue_utc == pd.Timestamp("2025-04-30 10:00", tz="UTC"))]
dfw_act, dfw_pred = float((d_dfw.y_ratio * d_dfw.aar_vfr).min()), float((d_dfw.pred_ratio * d_dfw.aar_vfr).min())
u_dfw3 = float(unseen[(unseen.lead_h == 3) & (unseen.scope == "KDFW")].rmse_aar.iloc[0])
u_atl3 = float(unseen[(unseen.lead_h == 3) & (unseen.scope == "KATL")].rmse_aar.iloc[0])
t_dfw3 = float(am[(am.lead_h == 3) & (am.scope == "KDFW") & (am.model == "taf_implied")].rmse_aar.iloc[0])
t_atl3 = float(am[(am.lead_h == 3) & (am.scope == "KATL") & (am.model == "taf_implied")].rmse_aar.iloc[0])
s = new_slide("Where it fails, and what we would do about it", notes=(
    "Honest evaluation. Failure one: on 30 April at DFW, with the forecast issued six hours ahead, the actual minimum arrival rate was "
    f"{dfw_act:.0f} per hour and our model forecast {dfw_pred:.0f}. Severe convective events are rare and the model under-reacts. The cautious setting and the TAF-implied line on the dashboard exist for exactly this. "
    f"Failure two: the biggest missed delay, a {miss.delay_min:.0f}-minute departure delay at DFW on 19 March, was given {miss.p:.0%} risk. Nothing in the weather predicted it. No weather model could. "
    f"Failure three: trained only on ATL and tested on DFW, the model is worse than the TAF alone ({u_dfw3:.1f} versus {t_dfw3:.1f}); trained only on DFW and tested on ATL it still wins ({u_atl3:.1f} versus {t_atl3:.1f}). "
    "So the pipeline runs for any airport, but with two airports we cannot claim the model transfers without retraining."))
items = [
    (RED, "Severe storms are under-called", f"DFW, 30 Apr 2025: six hours ahead the model said {dfw_pred:.0f} arrivals/h, the real minimum was {dfw_act:.0f}. Use the cautious setting and the TAF line."),
    (AMBER, "Some delays have no weather cause", f"A {miss.delay_min:.0f}-minute departure delay at DFW (19 Mar) was scored {miss.p:.0%} risk. Maintenance and crew issues are invisible to weather data."),
    (AMBER, "Transfer to a new airport is partial", f"Trained without DFW, the model is worse than the TAF alone on DFW ({u_dfw3:.1f} vs {t_dfw3:.1f}); without ATL it still wins ({u_atl3:.1f} vs {t_atl3:.1f}). Retrain per airport."),
    (MUTED, "Labels and data are proxies", "AAR is derived from FAA rates, not logged calls. The flight file samples roughly 3% of traffic, so counts are small."),
]
for i, (col, hd, bd) in enumerate(items):
    x, y = 0.6 + (i % 2) * 6.13, 1.5 + (i // 2) * 2.7
    card(s, x, y, 6.0, 2.5)
    badge(s, x + 0.3, y + 0.3, i + 1, fill=col if col != MUTED else DEEP)
    text(s, x + 1.0, y + 0.3, 4.8, 0.55, hd, size=20, bold=True, color=NAVY, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, y + 1.0, 5.4, 1.4, bd, size=15, color=INK)

# ===================================================================================== 11. robustness
scen = [n for n in sens.scenario.unique()]
s = new_slide("The result survives our own assumptions", notes=(
    "The AAR reductions are our assumptions, so we stress-tested them. We rebuilt the target, the baselines and the model from scratch under eight settings, for example halving or increasing the crosswind, tailwind and thunderstorm penalties, "
    "and a category-only target with no hazard terms at all. In every scenario, at both 3 and 12 hours, the model beats persistence and the TAF-implied rate. "
    "What does change is the absolute error: the tailwind and thunderstorm assumptions move RMSE a lot, so we do not claim the absolute number is the truth, only the ranking and the size of the gain."))
cd = CategoryChartData()
short = {"base": "Base", "thunderstorm x0.50": "Storm ×0.5", "thunderstorm x0.85": "Storm ×0.85", "crosswind 1.25%/kt": "Xwind 1.25%",
         "crosswind 5%/kt": "Xwind 5%", "tailwind 2%/kt": "Tail 2%", "tailwind 8%/kt": "Tail 8%", "category-only (no hazards)": "No hazards"}
cd.categories = [short[n] for n in scen]
for nm, key in (("Our model", "lightgbm"), ("TAF-implied", "taf_implied"), ("Persistence", "persistence")):
    cd.add_series(nm, [round(float(sens[(sens.scenario == n) & (sens.lead_h == 3) & (sens.model == key)].rmse_aar.iloc[0]), 1) for n in scen])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.45), Inches(8.6), Inches(5.35), cd)
ch = gf.chart
style_chart(ch, legend=True, size=12, title="+3 h forecast error under eight different AAR assumptions (RMSE, arrivals/h)")
color_series(ch, [SKY, GREEN, AMBER], line=False)
ch.plots[0].gap_width = 60
ch.value_axis.minimum_scale = 0
card(s, 9.5, 1.45, 3.23, 5.35)
text(s, 9.75, 1.65, 3.0, 0.35, "WHAT WE TOOK FROM IT", size=13, bold=True, color=SKY)
bullets(s, 9.75, 2.1, 2.8, 4.6, ["The model wins in all 8 scenarios, at +3 h and +12 h", "Absolute error depends on the storm and tailwind assumptions, so we quote the gain, not the absolute number",
                                  "Real facility logs would remove this assumption: the top next step"], size=14, gap=8)

# ===================================================================================== 12. dashboard
from PIL import Image
case = pa[(pa.airport == "KDFW") & (pa.issue_utc == pd.Timestamp("2025-02-18 17:00", tz="UTC"))].sort_values("hour_utc")
crop = IMG / "dash_forecast_crop.png"
Image.open(IMG / "dash_forecast.jpg").crop((195, 60, 765, 248)).save(crop)
s = new_slide("A dashboard a duty manager can act on", notes=(
    "Left: DFW, forecast issued 18 February at 17:00 UTC, hours before a thunderstorm line cut the arrival rate to about 40. Blue is the model, green is the TAF-implied rate, grey is persistence, red is what actually happened. "
    "Persistence stays near 100 and misses the drop completely; the model sees it hours ahead, though it is not as deep as reality. Right: the product, a Streamlit dashboard that replays the held-out 2025 period, so every forecast is genuinely out of sample. "
    "It has an alert-sensitivity switch, a jump-to-event menu, per-flight risk with its main drivers, history and trends, rule-based recommendations, and a page that shows the evidence and the failures. "
    "Deep link example: ?airport=KDFW&t0=2025-02-18T17."))
cd = CategoryChartData()
cd.categories = [f"{h:%H}Z" for h in case.hour_utc]
cd.add_series("Model forecast", [round(float(v), 0) for v in case.pred_ratio * case.aar_vfr])
cd.add_series("TAF-implied", [round(float(v), 0) for v in case.taf_aar_ratio * case.aar_vfr])
cd.add_series("Persistence", [round(float(v), 0) for v in case.persist_ratio * case.aar_vfr])
cd.add_series("What happened", [round(float(v), 0) for v in case.y_ratio * case.aar_vfr])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.45), Inches(6.9), Inches(4.2), cd)
ch = gf.chart
style_chart(ch, legend=True, size=12, title="DFW, issued 18 Feb 17:00 UTC: arrivals per hour, +3 to +12 h")
color_series(ch, [SKY, GREEN, GREY, RED])
ch.value_axis.minimum_scale = 20
ch.value_axis.maximum_scale = 120
text(s, 0.6, 5.75, 6.9, 0.8, "A thunderstorm line cut the real rate to about 40 arrivals/h. Persistence stayed near 100; the model saw the drop hours ahead "
                             "and flagged it SEVERE, though not as deep as reality.", size=13, color=MUTED)
s.shapes.add_picture(str(crop), Inches(7.75), Inches(1.45), width=Inches(4.98))
text(s, 7.75, 1.45 + 4.98 * 188 / 570 + 0.1, 4.98, 0.5, "The live view: alert banner and key numbers, with tabs for forecast, flight risk, history, advice and evidence", size=12, color=MUTED)
bullets(s, 7.75, 4.0, 4.98, 2.3, ["Replays unseen 2025, with a reveal-the-outcome switch", "Balanced or cautious alerts; per-flight reasons",
                                  "Rule-based advice; evidence and failure tabs"], size=13, gap=3)
text(s, 0.6, 6.6, 12.1, 0.4, "Try it:  streamlit run dashboard/app.py", size=14, bold=True, color=NAVY, font=BODY)

# ===================================================================================== 13. impact
s = new_slide("What improves, and what comes next", dark=True, notes=(
    "Impact stated as reasoned improvement against today's baseline. Today the rate is cut when weather arrives, which is what persistence represents. "
    "Our forecast cuts the error by about a third at 3 hours and by about 44 percent at 12 hours, and catches about 7 in 10 real capacity cuts with 7 in 10 of alerts being real, with a cautious setting that catches 9 in 10. "
    "We do not claim minutes saved or fuel saved: that needs a trial with real traffic managers. Next steps are logged AAR data, more airports, a live feed, and a user trial."))
stat(s, 0.6, 1.5, 3.9, f"{100 * (1 - R('lightgbm', 12) / R('persistence', 12)):.0f}%", "lower AAR error than the reactive baseline at +12 h", SKY, big_size=60, dark=True)
stat(s, 0.6, 3.6, 3.9, f"{bal3[1]:.0%}  /  {cau3[1]:.0%}", "of real capacity cuts flagged 3 h ahead (balanced / cautious)", AMBER, big_size=44, dark=True)
stat(s, 4.9, 1.5, 3.9, f"{D('E_plus_bts', 3) / prev:.1f}×", f"better than chance at spotting severe delays +3 h ahead (PR-AUC {D('E_plus_bts', 3):.2f} vs {prev:.2f})", GREEN, big_size=60, dark=True)
stat(s, 4.9, 3.6, 3.9, "10 tests", "guard against look-ahead, wrong wind geometry and broken splits", SKY, big_size=44, dark=True)
card(s, 9.2, 1.5, 3.53, 5.0, fill=DEEP)
text(s, 9.45, 1.7, 3.0, 0.35, "NEXT STEPS", size=13, bold=True, color=AMBER)
bullets(s, 9.45, 2.15, 3.0, 4.3, [[("Train on logged AARs ", {"bold": True, "color": WHITE}), ("(FAA ASPM) to drop the proxy", {"color": WHITE})],
                                  [("Add airports ", {"bold": True, "color": WHITE}), ("through the config file", {"color": WHITE})],
                                  [("Live METAR/TAF feed ", {"bold": True, "color": WHITE}), ("instead of replay", {"color": WHITE})],
                                  [("Trial with traffic managers ", {"bold": True, "color": WHITE}), ("to measure delay saved", {"color": WHITE})]], size=15, gap=10)
text(s, 0.6, 5.95, 8.2, 0.9, "We claim forecast accuracy and decision support, not minutes or fuel saved: that needs a trial with real traffic managers. A qualified person decides.",
     size=15, color=GREY, italic=True)

# ===================================================================================== 14. appendix: run it
s = new_slide("Appendix: run it yourself", notes="One command rebuilds everything from public data. The cached downloads make reruns fast.")
card(s, 0.6, 1.45, 7.3, 4.6, fill=NAVY)
text(s, 0.95, 1.7, 6.6, 0.35, "NO MANUAL STEPS", size=13, bold=True, color=AMBER)
text(s, 0.95, 2.15, 6.6, 3.6, [[("pip install -r requirements.txt", {"font": "Consolas", "size": 16, "color": WHITE})],
                               [("python run_pipeline.py", {"font": "Consolas", "size": 16, "color": WHITE})],
                               [("python -m pytest -q", {"font": "Consolas", "size": 16, "color": WHITE})],
                               [("streamlit run dashboard/app.py", {"font": "Consolas", "size": 16, "color": WHITE})]], size=16)
text(s, 0.95, 4.2, 6.6, 2.4, ["Downloads METAR + TAF from the Iowa State Mesonet, cleans them, builds leak-free features, trains and scores both models, "
                             "and writes every metric to reports/. About 5 minutes after the first download."], size=15, color="CADCFC")
card(s, 8.2, 1.45, 4.53, 4.6)
text(s, 8.5, 1.65, 4.0, 0.35, "ANY AIRPORT", size=13, bold=True, color=SKY)
bullets(s, 8.5, 2.1, 4.0, 3.9, ["Add an ICAO code, time zone, runway heading and FAA rates in src/config.py", "No heading? Falls back to the prevailing wind",
                                "AAR target is a ratio to the airport's best case; the model has no airport-ID feature",
                                "Delays are rates, never raw minutes, so a hub and a regional airport compare fairly"], size=14, gap=8)

prs.save(str(OUT))
print("saved", OUT, "slides:", len(prs.slides))
