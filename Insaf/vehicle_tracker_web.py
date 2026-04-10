"""
=============================================================
  نظام رصد المركبات v6.0
  ✦ YOLOv8  — كشف دقيق للسيارات
  ✦ Optical Flow — سرعة دقيقة
  ✦ 4 ألوان: أحمر، أزرق، أصفر، أخضر
  ✦ تقرير تفصيلي لكل سيارة
=============================================================
التثبيت:
    pip install opencv-python numpy flask ultralytics

التشغيل:
    python vehicle_tracker_web.py
    python vehicle_tracker_web.py --video "video.mp4"

افتح: http://localhost:5000
=============================================================
"""

import cv2, numpy as np, math, time, threading, argparse, os, tempfile
from collections import deque
from flask import Flask, Response, jsonify, request

# ══════════════════════════════════════════════════════
#  إعدادات
# ══════════════════════════════════════════════════════
SPEED_GREEN   = 60    # km/h أخضر
SPEED_YELLOW  = 100   # km/h أصفر
SPEED_MAX     = 220

# معايرة السرعة
# px_per_m يُحسب مرة واحدة عند أول ظهور السيارة ويُثبَّت
CAR_REAL_LEN_M = 4.5   # متر
SMOOTH_N       = 8     # إطارات تنعيم (أكثر = أهدأ)
LK_MAX_PTS     = 60    # نقاط Lucas-Kanade لكل سيارة

HISTORY        = 35
LANE_DEV_THR   = 80   # بكسل
COLLISION_DIST = 60   # بكسل
MAX_MISSING    = 15
MIN_LIFE       = 3    # إطارات قبل اعتبار السيارة مؤكدة

YOLO_CONF      = 0.35 # حد الثقة
YOLO_CLASSES   = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ألوان OpenCV (BGR)
CV_GREEN  = (30, 185, 30)
CV_YELLOW = (0,  205, 225)
CV_RED    = (30,  30, 205)
CV_BLACK  = (0,   0,   0)
CV_WHITE  = (220, 220, 220)

# ══════════════════════════════════════════════════════
#  تحميل YOLOv8
# ══════════════════════════════════════════════════════
_yolo_model = None
_yolo_lock  = threading.Lock()

def get_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
                _yolo_model = YOLO("yolov8n.pt")
                print("[YOLO] Model loaded ✓")
            except Exception as e:
                print(f"[YOLO] Failed: {e}")
                _yolo_model = False
    return _yolo_model

# ══════════════════════════════════════════════════════
#  HTML
# ══════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>رصد المركبات v6</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#080e08;--card:#0d160d;--g0:#0c1c0c;--g1:#162816;
  --border:rgba(45,95,45,.48);--bhi:rgba(70,150,70,.6);
  --accent:#4e9e4e;--acc-lo:rgba(78,158,78,.09);
  --green:#4e9e4e;--yellow:#b89818;--red:#a02828;--cyan:#2e7a6e;
  --text:#96bc96;--tdim:#3e5e3e;--thi:#bee0be;
  --mono:'IBM Plex Mono',monospace;--sans:'Cairo',sans-serif;--r:5px
}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(40,90,40,.045) 1px,transparent 1px),
  linear-gradient(90deg,rgba(40,90,40,.045) 1px,transparent 1px);background-size:44px 44px}
.root{position:relative;z-index:1;padding:12px;display:flex;flex-direction:column;gap:10px}

/* Header */
.hdr{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
  padding:10px 18px;background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);position:relative;overflow:hidden;gap:10px}
.hdr::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.28}
.brand{display:flex;align-items:center;gap:9px}
.bico{width:33px;height:33px;border-radius:50%;border:1px solid var(--bhi);
  display:flex;align-items:center;justify-content:center;background:var(--g0);font-size:.95rem}
.bname{font-weight:700;font-size:.95rem;color:var(--thi)}
.bver{font-family:var(--mono);font-size:.52rem;color:var(--tdim);margin-top:1px}
.pill{display:flex;align-items:center;gap:5px;padding:3px 11px;border-radius:20px;
  border:1px solid var(--bhi);background:var(--acc-lo);
  font-family:var(--mono);font-size:.58rem;color:var(--accent)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent);
  animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1;box-shadow:0 0 4px var(--accent)}50%{opacity:.2;box-shadow:none}}
.hclock{display:flex;flex-direction:column;align-items:flex-end;gap:1px}
.htime{font-family:var(--mono);font-size:.8rem;color:var(--thi)}
.hdate{font-family:var(--mono);font-size:.52rem;color:var(--tdim)}

/* Layout */
.main{display:grid;grid-template-columns:1fr 280px;gap:10px}
.lcol,.rcol{display:flex;flex-direction:column;gap:10px}

/* Card */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
.ch{display:flex;align-items:center;justify-content:space-between;
  padding:7px 13px;border-bottom:1px solid var(--border);background:rgba(255,255,255,.01)}
.chlbl{font-family:var(--mono);font-size:.58rem;color:var(--tdim);letter-spacing:.12em;
  text-transform:uppercase;display:flex;align-items:center;gap:5px}
.chlbl::before{content:'';width:5px;height:5px;border-radius:1px;background:var(--accent);opacity:.6}
.bdg{font-family:var(--mono);font-size:.54rem;padding:1px 7px;border-radius:20px;
  background:var(--g0);border:1px solid var(--border);color:var(--tdim)}

/* Video */
.vwrap{position:relative;background:#000;aspect-ratio:16/9;overflow:hidden}
#stream{width:100%;height:100%;object-fit:contain;display:block}
.vc{position:absolute;width:15px;height:15px;border-color:var(--accent);border-style:solid;opacity:.48}
.vc.tl{top:7px;left:7px;border-width:1px 0 0 1px}
.vc.tr{top:7px;right:7px;border-width:1px 1px 0 0}
.vc.bl{bottom:7px;left:7px;border-width:0 0 1px 1px}
.vc.br{bottom:7px;right:7px;border-width:0 1px 1px 0}
.vtags{position:absolute;bottom:8px;left:8px;display:flex;gap:4px}
.vtag{font-family:var(--mono);font-size:.55rem;padding:1px 6px;border-radius:2px;
  background:rgba(0,0,0,.72);border:1px solid var(--border);color:var(--tdim)}
.vprog{position:absolute;bottom:0;left:0;right:0;height:2px;background:rgba(0,0,0,.4)}
.vprog-fill{height:100%;background:var(--accent);transition:width .5s linear;width:0%}

/* Done overlay */
.done-overlay{display:none;position:absolute;inset:0;background:rgba(8,14,8,.88);
  align-items:center;justify-content:center;flex-direction:column;gap:10px;
  animation:fadeIn .6s ease}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.done-overlay.show{display:flex}
.done-ico{font-size:2.5rem}
.done-txt{font-family:var(--mono);font-size:.72rem;color:var(--accent);letter-spacing:.12em}

/* Summary panel */
.sum-panel{display:none;flex-direction:column;gap:0;
  border:1px solid var(--bhi);border-radius:var(--r);overflow:hidden;
  animation:sdn .5s ease}
@keyframes sdn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
.sum-panel.show{display:flex}
.sum-hdr{padding:11px 15px;background:var(--g0);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between}
.sum-title{font-family:var(--mono);font-size:.62rem;color:var(--accent);letter-spacing:.12em;text-transform:uppercase}
.sum-ts{font-family:var(--mono);font-size:.55rem;color:var(--tdim)}
.sum-body{padding:14px;background:var(--card);display:flex;flex-direction:column;gap:13px}
.sum-sec{display:flex;flex-direction:column;gap:5px}
.sum-sec-t{font-family:var(--mono);font-size:.57rem;color:var(--tdim);letter-spacing:.12em;
  text-transform:uppercase;padding-bottom:5px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:5px}
.sum-sec-t::before{content:'';width:4px;height:4px;border-radius:1px;background:var(--accent)}
.sum-lines{display:flex;flex-direction:column;gap:3px}
.sl{display:flex;justify-content:space-between;align-items:baseline;
  padding:4px 8px;border-radius:3px;background:var(--g0);
  font-family:var(--mono);font-size:.61rem}
.sl:hover{background:var(--g1)}
.sll{color:var(--tdim)}.slv{color:var(--thi);font-weight:600}
.slv.g{color:var(--green)}.slv.y{color:var(--yellow)}.slv.r{color:var(--red)}.slv.c{color:var(--cyan)}

/* Car report table */
.cr-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.6rem}
.cr-table th{padding:5px 8px;text-align:right;color:var(--tdim);letter-spacing:.07em;
  text-transform:uppercase;border-bottom:1px solid var(--border);background:var(--g0);font-weight:400}
.cr-table td{padding:5px 8px;border-bottom:1px solid rgba(45,95,45,.18);vertical-align:middle}
.cr-table tr:last-child td{border-bottom:none}
.cr-table tr:hover td{background:rgba(12,28,12,.6)}
.cid{color:var(--cyan);font-weight:600}
/* لون السيارة */
.clr{display:inline-block;padding:1px 6px;border-radius:2px;font-size:.52rem;text-transform:uppercase}
.clr.red  {background:rgba(160,40,40,.22);color:#c06060;border:1px solid rgba(160,40,40,.3)}
.clr.blue {background:rgba(40,80,160,.22);color:#6090c0;border:1px solid rgba(40,80,160,.3)}
.clr.yellow{background:rgba(180,150,20,.2);color:#c8a820;border:1px solid rgba(180,150,20,.3)}
.clr.green{background:rgba(40,140,40,.18);color:#5ab85a;border:1px solid rgba(40,140,40,.3)}
.clr.other{background:var(--g1);color:var(--tdim);border:1px solid var(--border)}
/* سرعة */
.spg{color:var(--green)}.spy{color:var(--yellow)}.spr{color:var(--red);font-weight:600}
.yes{color:var(--red)}.no{color:var(--tdim)}

/* Verdict */
.verdict{padding:11px 13px;border-radius:4px;border:1px solid var(--border);
  display:flex;align-items:flex-start;gap:9px;font-family:var(--mono);font-size:.62rem;line-height:1.6}
.verdict.safe  {border-color:rgba(78,158,78,.5);background:rgba(78,158,78,.06);color:var(--green)}
.verdict.warn  {border-color:rgba(184,152,24,.4);background:rgba(184,152,24,.05);color:var(--yellow)}
.verdict.danger{border-color:rgba(160,40,40,.5);background:rgba(160,40,40,.07);color:var(--red)}
.v-ico{font-size:1.3rem;flex-shrink:0}

/* Right stats */
.sgrid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border)}
.sbox{background:var(--card);padding:10px 12px;display:flex;flex-direction:column;gap:2px}
.slbl2{font-family:var(--mono);font-size:.53rem;color:var(--tdim);letter-spacing:.1em;text-transform:uppercase}
.sval{font-family:var(--mono);font-size:1.5rem;font-weight:600;line-height:1;color:var(--accent);transition:color .4s}
.sval.y{color:var(--yellow)}.sval.r{color:var(--red)}.sval.c{color:var(--cyan)}
.sval.flash{animation:fv .6s infinite}
@keyframes fv{0%,100%{opacity:1}50%{opacity:.22}}
.ssub{font-family:var(--mono);font-size:.5rem;color:var(--tdim)}

/* Bars */
.barsec{padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.br{display:flex;flex-direction:column;gap:3px}
.bl{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.53rem;color:var(--tdim)}
.bt{height:4px;background:var(--g0);border-radius:2px;overflow:hidden}
.bf{height:100%;border-radius:2px;transition:width .5s,background .5s;background:var(--accent)}

/* Objects list */
.olist{padding:6px;display:flex;flex-direction:column;gap:3px;max-height:180px;overflow-y:auto}
.olist::-webkit-scrollbar{width:2px}
.olist::-webkit-scrollbar-thumb{background:var(--g1)}
.orow{display:grid;grid-template-columns:40px 52px 65px 32px 20px;
  align-items:center;gap:5px;padding:4px 6px;background:var(--g0);border-radius:3px;
  font-family:var(--mono);font-size:.58rem;border:1px solid transparent;transition:border-color .2s}
.orow:hover{border-color:var(--bhi)}
.orow.coll{border-color:rgba(160,40,40,.5);background:rgba(160,40,40,.07)}
.oid{color:var(--cyan)}
.ctag{padding:1px 4px;border-radius:2px;text-align:center;font-size:.5rem;text-transform:uppercase}
.ctag.red   {background:rgba(160,40,40,.2);color:#c06060;border:1px solid rgba(160,40,40,.3)}
.ctag.blue  {background:rgba(40,80,160,.2);color:#6090c0;border:1px solid rgba(40,80,160,.3)}
.ctag.yellow{background:rgba(180,150,20,.2);color:#c8a820;border:1px solid rgba(180,150,20,.3)}
.ctag.green2{background:rgba(40,140,40,.18);color:#5ab85a;border:1px solid rgba(40,140,40,.3)}
.ctag.other {background:var(--g1);color:var(--tdim);border:1px solid var(--border)}

/* Alerts */
.alist{padding:6px;display:flex;flex-direction:column;gap:3px;max-height:145px;overflow-y:auto}
.alist::-webkit-scrollbar{width:2px}
.alist::-webkit-scrollbar-thumb{background:var(--g1)}
.ai{display:flex;gap:6px;padding:4px 6px;border-radius:3px;border-right:2px solid var(--border);
  background:rgba(255,255,255,.012);font-family:var(--mono);font-size:.58rem;line-height:1.3;
  animation:si .2s ease}
@keyframes si{from{opacity:0;transform:translateY(-2px)}to{opacity:1;transform:none}}
.ai.coll{border-right-color:var(--red);color:#b06060;background:rgba(160,40,40,.07)}
.ai.dev{border-right-color:var(--yellow);color:#a08030}
.ai.info{color:var(--tdim)}
.at{color:var(--tdim);white-space:nowrap;font-size:.52rem}

/* Upload */
.upbody{padding:10px;display:flex;flex-direction:column;gap:7px}
.dz{border:1px dashed var(--bhi);border-radius:var(--r);padding:14px 10px;
  text-align:center;cursor:pointer;transition:all .25s;background:var(--g0);
  display:flex;flex-direction:column;align-items:center;gap:4px}
.dz:hover,.dz.over{border-color:var(--accent);background:var(--acc-lo)}
.dz.ready{border-color:var(--cyan);border-style:solid}
.dzico{font-size:1.25rem;opacity:.38}
.dztxt{font-family:var(--mono);font-size:.6rem;color:var(--text)}
.dzsub{font-family:var(--mono);font-size:.52rem;color:var(--tdim)}
.pw{display:none;flex-direction:column;gap:3px}
.pl{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.52rem;color:var(--tdim)}

/* Controls */
.cgrid{padding:9px;display:grid;grid-template-columns:1fr 1fr;gap:5px}
.btn{padding:7px;border-radius:3px;border:1px solid var(--border);background:transparent;
  color:var(--text);font-family:var(--mono);font-size:.6rem;cursor:pointer;
  letter-spacing:.05em;transition:all .2s}
.btn:hover{border-color:var(--bhi);background:var(--acc-lo);color:var(--thi)}
.btn:active{transform:scale(.97)}
.btn.pri{border-color:var(--cyan);color:var(--cyan)}
.btn.dan{border-color:rgba(160,40,40,.4);color:#b06060}
.btn.dan:hover{background:rgba(160,40,40,.1)}
.btn.play{display:none;border-color:var(--accent);color:var(--accent);grid-column:span 2}

/* Footer */
.ftr{padding:7px 14px;border:1px solid var(--border);border-radius:var(--r);background:var(--card);
  display:flex;justify-content:space-between;align-items:center;
  font-family:var(--mono);font-size:.52rem;color:var(--tdim)}
.leg{display:flex;gap:10px;align-items:center}
.li{display:flex;align-items:center;gap:3px}
.ld{width:7px;height:4px;border-radius:1px}
</style>
</head>
<body>
<div class="root">

<!-- Header -->
<header class="hdr">
  <div class="brand">
    <div class="bico">🚗</div>
    <div>
      <div class="bname">نظام رصد المركبات</div>
      <div class="bver">YOLOv8 · Optical Flow · 4-Color · v6.0</div>
    </div>
  </div>
  <div class="pill"><div class="dot"></div><span id="liveLabel">READY</span></div>
  <div class="hclock">
    <div class="htime" id="clock">--:--:--</div>
    <div class="hdate" id="hdate">----</div>
  </div>
</header>

<!-- Main -->
<div class="main">
  <div class="lcol">

    <!-- Video -->
    <div class="card">
      <div class="ch">
        <span class="chlbl">البث المباشر</span>
        <div style="display:flex;gap:5px">
          <span class="bdg" id="fTag">FRAME 0</span>
          <span class="bdg" id="srcTag">CAMERA</span>
          <span class="bdg" id="stTag" style="color:var(--accent)">READY</span>
        </div>
      </div>
      <div class="vwrap">
        <img id="stream" src="/video_feed" alt="">
        <div class="vc tl"></div><div class="vc tr"></div>
        <div class="vc bl"></div><div class="vc br"></div>
        <div class="vtags">
          <span class="vtag" id="vCars">CARS: 0</span>
          <span class="vtag" id="vMax">MAX: -- km/h</span>
          <span class="vtag" id="vColl">COLL: 0</span>
        </div>
        <div class="vprog"><div class="vprog-fill" id="vProg"></div></div>
        <div class="done-overlay" id="doneOv">
          <div class="done-ico">✅</div>
          <div class="done-txt">انتهى التشغيل · شاهد الملخص أدناه</div>
        </div>
      </div>
    </div>

    <!-- Summary -->
    <div class="sum-panel" id="sumPanel">
      <div class="sum-hdr">
        <div class="sum-title">📋 تقرير الجلسة الكامل</div>
        <div class="sum-ts" id="sumTs">--</div>
      </div>
      <div class="sum-body">

        <!-- وقت -->
        <div class="sum-sec">
          <div class="sum-sec-t">الوقت والمدة</div>
          <div class="sum-lines">
            <div class="sl"><span class="sll">وقت البدء</span><span class="slv" id="rStart">--</span></div>
            <div class="sl"><span class="sll">وقت الانتهاء</span><span class="slv" id="rEnd">--</span></div>
            <div class="sl"><span class="sll">مدة التشغيل</span><span class="slv g" id="rDur">--</span></div>
            <div class="sl"><span class="sll">إجمالي الإطارات</span><span class="slv" id="rFrames">--</span></div>
          </div>
        </div>

        <!-- إجمالي -->
        <div class="sum-sec">
          <div class="sum-sec-t">ملخص إجمالي</div>
          <div class="sum-lines">
            <div class="sl"><span class="sll">إجمالي السيارات</span><span class="slv" id="rTotal">--</span></div>
            <div class="sl"><span class="sll">أعلى سرعة مسجّلة</span><span class="slv y" id="rMaxS">--</span></div>
            <div class="sl"><span class="sll">متوسط السرعة</span><span class="slv g" id="rAvgS">--</span></div>
            <div class="sl"><span class="sll">سيارات انحرفت</span><span class="slv y" id="rDevC">--</span></div>
            <div class="sl"><span class="sll">سيارات اصطدمت</span><span class="slv r" id="rCollC">--</span></div>
          </div>
        </div>

        <!-- جدول كل سيارة -->
        <div class="sum-sec">
          <div class="sum-sec-t">تقرير تفصيلي — كل سيارة</div>
          <div id="carTable" style="overflow-x:auto"></div>
        </div>

        <!-- حكم -->
        <div class="verdict" id="verdict">
          <div class="v-ico" id="vIco">⚪</div>
          <div id="vTxt">في انتظار انتهاء الجلسة...</div>
        </div>

      </div>
    </div>
  </div>

  <!-- Right col -->
  <div class="rcol">

    <div class="card">
      <div class="ch"><span class="chlbl">إحصائيات لحظية</span></div>
      <div class="sgrid">
        <div class="sbox"><div class="slbl2">مركبات</div><div class="sval" id="sCars">0</div><div class="ssub">نشطة</div></div>
        <div class="sbox"><div class="slbl2">أعلى سرعة</div><div class="sval y" id="sMax">0</div><div class="ssub">km/h</div></div>
        <div class="sbox"><div class="slbl2">اصطدامات</div><div class="sval r" id="sColl">0</div><div class="ssub">الآن</div></div>
        <div class="sbox"><div class="slbl2">انحرافات</div><div class="sval y" id="sDev">0</div><div class="ssub">الآن</div></div>
        <div class="sbox"><div class="slbl2">حمراء</div><div class="sval r" id="sRed">0</div><div class="ssub">سيارة</div></div>
        <div class="sbox"><div class="slbl2">زرقاء</div><div class="sval c" id="sBlue">0</div><div class="ssub">سيارة</div></div>
      </div>
    </div>

    <div class="card">
      <div class="ch"><span class="chlbl">مؤشر السرعة</span></div>
      <div class="barsec">
        <div class="br">
          <div class="bl"><span>متوسط</span><span id="bAvg">0 km/h</span></div>
          <div class="bt"><div class="bf" id="barAvg" style="width:0%"></div></div>
        </div>
        <div class="br">
          <div class="bl"><span>أقصى سرعة</span><span id="bMax">0 km/h</span></div>
          <div class="bt"><div class="bf" id="barMax" style="width:0%"></div></div>
        </div>
        <div class="br">
          <div class="bl"><span>كثافة</span><span id="bDens">0%</span></div>
          <div class="bt"><div class="bf" id="barDens" style="width:0%;background:var(--cyan)"></div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="ch"><span class="chlbl">قائمة المركبات</span><span class="bdg" id="carCnt">0</span></div>
      <div class="olist" id="oList">
        <div style="color:var(--tdim);font-size:.58rem;padding:7px;font-family:var(--mono)">لا توجد مركبات...</div>
      </div>
    </div>

    <div class="card">
      <div class="ch"><span class="chlbl">رفع فيديو</span></div>
      <div class="upbody">
        <label class="dz" id="dzEl" for="fInput">
          <div class="dzico">📂</div>
          <div class="dztxt" id="dzTxt">اسحب فيديو أو اضغط للاختيار</div>
          <div class="dzsub">MP4 · AVI · MOV · MKV</div>
        </label>
        <input type="file" id="fInput" accept="video/*" style="display:none" onchange="onFile(event)">
        <div class="pw" id="pw">
          <div class="pl"><span id="pTxt">جاري الرفع...</span><span id="pPct">0%</span></div>
          <div class="bt"><div class="bf" id="pBar" style="width:0%;background:var(--cyan)"></div></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:5px">
          <button class="btn play" id="btnPlay" onclick="uploadVid()">▶ تشغيل الفيديو</button>
          <!-- زر الكاميرا غير متاح في نسخة الويب -->
        </div>
      </div>
    </div>

    <div class="card">
      <div class="ch"><span class="chlbl">سجل الأحداث</span><span class="bdg" id="aBdg">0</span></div>
      <div class="alist" id="aList">
        <div class="ai info"><span class="at">--:--</span><span>انتظار...</span></div>
      </div>
    </div>

    <div class="card">
      <div class="ch"><span class="chlbl">التحكم</span></div>
      <div class="cgrid">
        <button class="btn" id="btnPause" onclick="ctrl('pause')">⏸ إيقاف</button>
        <button class="btn" onclick="ctrl('snapshot')">📷 لقطة</button>
        <button class="btn dan" onclick="ctrl('stop')">■ إيقاف كلي</button>
        <button class="btn pri" onclick="switchCam()">📷 كاميرا</button>
      </div>
    </div>

  </div>
</div>

<footer class="ftr">
  <span>Vehicle Tracker v6.0 · YOLOv8 + Optical Flow</span>
  <div class="leg">
    <div class="li"><div class="ld" style="background:#4e9e4e"></div>≤60 km/h</div>
    <div class="li"><div class="ld" style="background:#b89818"></div>61–100</div>
    <div class="li"><div class="ld" style="background:#a02828"></div>&gt;100 km/h</div>
  </div>
  <span id="fTime">--</span>
</footer>
</div>

<script>
let alerts=[], prev={}, paused=false, selFile=null;
let sessionDone=false, sessStart=null;

function tick(){
  const n=new Date();
  $('clock',n.toTimeString().slice(0,8));
  $('hdate',n.toLocaleDateString('ar-SA'));
  $('fTime',n.toLocaleString('ar-SA'));
}
setInterval(tick,1000);tick();

function $(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}
function sC(v){return v>100?'var(--red)':v>60?'var(--yellow)':'var(--accent)';}
function sv(id,v,cls){const e=document.getElementById(id);if(!e)return;e.textContent=v;e.className='sval'+(cls?' '+cls:'');}
function bar(id,r,col){const e=document.getElementById(id);if(!e)return;e.style.width=Math.min(r*100,100)+'%';if(col)e.style.background=col;}

async function poll(){
  try{
    const d=await(await fetch('/stats')).json();
    if(!sessStart&&d.frame>0)sessStart=new Date();

    $('fTag','FRAME '+d.frame);
    $('vCars','CARS: '+d.total);
    $('vMax','MAX: '+d.max_speed+' km/h');
    $('vColl','COLL: '+d.collisions);

    const stEl=document.getElementById('stTag');
    if(d.state==='done'){stEl.textContent='DONE';stEl.style.color='var(--green)'}
    else if(d.state==='running'){stEl.textContent='LIVE';stEl.style.color='var(--accent)';$('liveLabel','LIVE')}

    if(d.total_frames>0)document.getElementById('vProg').style.width=(d.frame/d.total_frames*100)+'%';

    sv('sCars',d.total,'');
    sv('sMax',d.max_speed,d.max_speed>100?'r':d.max_speed>60?'y':'');
    sv('sColl',d.collisions,d.collisions>0?'r flash':'r');
    sv('sDev',d.deviations,d.deviations>0?'y':'');
    sv('sRed',d.red,'r');sv('sBlue',d.blue,'c');
    $('carCnt',d.total);

    const avg=d.avg_speed||0,mx=d.max_speed||0;
    bar('barAvg',avg/220,sC(avg));$('bAvg',avg.toFixed(0)+' km/h');
    bar('barMax',mx/220,sC(mx)); $('bMax',mx.toFixed(0)+' km/h');
    const dens=Math.min(d.total/10*100,100);
    bar('barDens',dens/100,'var(--cyan)');$('bDens',dens.toFixed(0)+'%');

    renderObjs(d.objects||[]);

    if(d.collisions>0&&d.collisions!==(prev.collisions||0))addA('coll','💥 اصطدام — '+d.collisions+' حالة');
    if(d.deviations>0&&d.deviations!==(prev.deviations||0))addA('dev','↩ انحراف مسار ('+d.deviations+')');
    if(mx>150&&mx!==(prev.max_speed||0))addA('coll','⚡ سرعة خطرة: '+mx.toFixed(0)+' km/h');

    if(d.state==='done'&&!sessionDone){sessionDone=true;showSummary(d);}
    prev=d;
  }catch(e){}
}
setInterval(poll,400);poll();

// ── ألوان السيارة ──────────────────────────
const COLOR_AR={red:'حمراء',blue:'زرقاء',yellow:'صفراء',green:'خضراء',other:'أخرى'};

function renderObjs(objs){
  const el=document.getElementById('oList');
  if(!objs.length){el.innerHTML='<div style="color:var(--tdim);font-size:.58rem;padding:7px;font-family:var(--mono)">لا توجد مركبات...</div>';return;}
  el.innerHTML=objs.map(o=>{
    const sc=sC(o.speed);
    const cCls=o.color==='green'?'green2':o.color;
    const ct=COLOR_AR[o.color]||'أخرى';
    const w=o.collision?'💥':o.deviated?'↩':'';
    return `<div class="orow${o.collision?' coll':''}">
      <span class="oid">C-${o.id}</span>
      <span class="ctag ${cCls}">${ct}</span>
      <span style="color:${sc};font-weight:600;font-family:var(--mono)">${o.speed.toFixed(0)} km/h</span>
      <span style="color:var(--tdim);font-size:.5rem">${o.size}</span>
      <span>${w}</span></div>`;
  }).join('');
}

// ── ملخص ─────────────────────────────────
function showSummary(d){
  document.getElementById('doneOv').classList.add('show');
  $('liveLabel','DONE');

  const endTime=new Date();
  const startStr=sessStart?sessStart.toTimeString().slice(0,8):'--';
  const endStr=endTime.toTimeString().slice(0,8);
  const dur=sessStart?Math.floor((endTime-sessStart)/1000):0;
  const hh=String(Math.floor(dur/3600)).padStart(2,'0');
  const mm=String(Math.floor(dur%3600/60)).padStart(2,'0');
  const ss=String(dur%60).padStart(2,'0');
  $('sumTs',endStr);$('rStart',startStr);$('rEnd',endStr);
  $('rDur',`${hh}:${mm}:${ss}`);$('rFrames',d.frame+' إطار');

  const cars=d.session_cars||[];
  const mx=d.session_max_speed||0;
  const devC=cars.filter(c=>c.deviated).length;
  const collC=cars.filter(c=>c.collided).length;

  $('rTotal',cars.length+' سيارة');
  document.getElementById('rMaxS').textContent=mx.toFixed(0)+' km/h';
  document.getElementById('rMaxS').className='slv'+(mx>150?' r':mx>100?' y':' g');
  $('rAvgS',(d.session_avg_speed||0).toFixed(0)+' km/h');

  const devEl=document.getElementById('rDevC');
  devEl.textContent=devC===0?'لا يوجد انحراف':devC===cars.length?'كل السيارات ('+devC+')':devC+' من '+cars.length;
  devEl.className='slv'+(devC>0?' y':'');

  const collEl=document.getElementById('rCollC');
  collEl.textContent=collC===0?'لا يوجد اصطدام':collC+' سيارة';
  collEl.className='slv'+(collC>0?' r':'');

  // جدول كل سيارة
  const tbl=document.getElementById('carTable');
  if(!cars.length){
    tbl.innerHTML='<div style="color:var(--tdim);font-size:.6rem;padding:8px;font-family:var(--mono)">لم تُرصد سيارات.</div>';
  } else {
    const rows=cars.map(c=>{
      const sCls=c.max_speed>100?'spr':c.max_speed>60?'spy':'spg';
      const cAr=COLOR_AR[c.color]||'أخرى';
      const cCls=c.color==='green'?'green':c.color;
      const devTxt=c.deviated?'<span class="yes">✓ انحرفت</span>':'<span class="no">— لا</span>';
      const collTxt=c.collided?'<span class="yes">💥 اصطدام</span>':'<span class="no">— لا</span>';
      return `<tr>
        <td class="cid">C-${c.id}</td>
        <td><span class="clr ${cCls}">${cAr}</span></td>
        <td><span class="${sCls}">${c.max_speed.toFixed(0)} km/h</span></td>
        <td style="color:var(--tdim)">${c.avg_speed.toFixed(0)} km/h</td>
        <td>${devTxt}</td>
        <td>${collTxt}</td>
      </tr>`;
    }).join('');
    tbl.innerHTML=`<table class="cr-table">
      <thead><tr>
        <th>رقم</th><th>اللون</th><th>أعلى سرعة</th><th>متوسط</th><th>انحراف</th><th>اصطدام</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  }

  // حكم
  const vEl=document.getElementById('verdict');
  const viEl=document.getElementById('vIco');
  const vtEl=document.getElementById('vTxt');
  if(collC>0&&mx>130){vEl.className='verdict danger';viEl.textContent='🔴';
    vtEl.textContent=`خطر — ${collC} سيارة اصطدمت، أعلى سرعة ${mx.toFixed(0)} km/h.`;}
  else if(collC>0||devC>0||mx>100){vEl.className='verdict warn';viEl.textContent='🟡';
    vtEl.textContent=`تحذير — ${collC} اصطدام، ${devC} انحراف، أعلى سرعة ${mx.toFixed(0)} km/h.`;}
  else{vEl.className='verdict safe';viEl.textContent='🟢';
    vtEl.textContent=`مقبول — لا اصطدامات. أعلى سرعة ${mx.toFixed(0)} km/h. ${devC} انحراف.`;}

  document.getElementById('sumPanel').classList.add('show');
  setTimeout(()=>document.getElementById('sumPanel').scrollIntoView({behavior:'smooth'}),300);
  addA('info','📋 الملخص جاهز');
}

function addA(type,msg){
  const t=new Date().toTimeString().slice(0,5);
  alerts.unshift({type,msg,t});if(alerts.length>50)alerts.pop();
  $('aBdg',alerts.length);
  document.getElementById('aList').innerHTML=alerts.map(a=>
    `<div class="ai ${a.type}"><span class="at">${a.t}</span><span>${a.msg}</span></div>`
  ).join('');
}

async function ctrl(c){
  try{
    const d=await(await fetch('/control/'+c)).json();
    if(c==='pause'){paused=!paused;$('btnPause',paused?'▶ استكمال':'⏸ إيقاف');}
    if(c==='snapshot')addA('info','📷 لقطة: '+(d.filename||''));
  }catch(e){}
}

const dzEl=document.getElementById('dzEl');
dzEl.addEventListener('dragover',e=>{e.preventDefault();dzEl.classList.add('over')});
dzEl.addEventListener('dragleave',()=>dzEl.classList.remove('over'));
dzEl.addEventListener('drop',e=>{e.preventDefault();dzEl.classList.remove('over');const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('video/'))setF(f)});
function onFile(e){const f=e.target.files[0];if(f)setF(f);}
function setF(f){
  selFile=f;const mb=(f.size/1024/1024).toFixed(1);
  document.getElementById('dzTxt').textContent='✓ '+f.name+' ('+mb+' MB)';
  dzEl.classList.add('ready');document.getElementById('btnPlay').style.display='block';
  addA('info','📁 '+f.name);
}
async function uploadVid(){
  if(!selFile)return;
  sessionDone=false;sessStart=null;
  document.getElementById('doneOv').classList.remove('show');
  document.getElementById('sumPanel').classList.remove('show');
  document.getElementById('pw').style.display='flex';
  const fd=new FormData();fd.append('video',selFile);
  const xhr=new XMLHttpRequest();xhr.open('POST','/upload_video',true);
  xhr.upload.onprogress=e=>{
    if(!e.lengthComputable)return;
    const p=Math.round(e.loaded/e.total*100);
    $('pPct',p+'%');document.getElementById('pBar').style.width=p+'%';
    $('pTxt',p<100?'جاري الرفع...':'جاري التشغيل...');
  };
  xhr.onload=()=>{
    try{const r=JSON.parse(xhr.responseText);
      if(r.ok){$('srcTag','VIDEO');$('liveLabel','LIVE');addA('info','▶ '+selFile.name);}
      else addA('coll','✗ '+(r.error||''));
    }catch(e){}
    setTimeout(()=>document.getElementById('pw').style.display='none',3000);
  };
  xhr.send(fd);
}
async function switchCam(){
  try{
    await fetch('/switch_source/camera');
    selFile=null;
    document.getElementById('dzTxt').textContent='اسحب فيديو أو اضغط للاختيار';
    dzEl.classList.remove('ready');
    document.getElementById('btnPlay').style.display='none';
    document.getElementById('pw').style.display='none';
    document.getElementById('fInput').value='';
    $('srcTag','CAMERA');$('liveLabel','LIVE');
    sessionDone=false;sessStart=null;
    document.getElementById('doneOv').classList.remove('show');
    document.getElementById('sumPanel').classList.remove('show');
    addA('info','📷 تم التبديل للكاميرا');
  }catch(e){}
}
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════
#  تصنيف لون السيارة — 4 ألوان
# ══════════════════════════════════════════════════════
def classify_color(roi):
    """
    يصنّف لون السيارة: red / blue / yellow / green / other
    - نأخذ المنطقة الوسطى (60%) لتجنب الخلفية
    - نستخدم عتبة تكيفية: اللون يفوز إذا تجاوز 10% من البكسلات الملوّنة
    """
    if roi is None or roi.size == 0: return "other"

    h, w = roi.shape[:2]
    # اقتصاص المنطقة الوسطى
    y1,y2 = max(0, int(h*0.15)), min(h, int(h*0.85))
    x1,x2 = max(0, int(w*0.10)), min(w, int(w*0.90))
    crop  = roi[y1:y2, x1:x2]
    if crop.size == 0: crop = roi

    # تصغير للسرعة
    small = cv2.resize(crop, (48, 24))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s_ch  = hsv[:,:,1]  # قناة التشبع
    v_ch  = hsv[:,:,2]  # قناة الإضاءة

    # نحسب فقط على البكسلات الملوّنة (تشبع > 60 وإضاءة > 40)
    colored_mask = (s_ch > 60) & (v_ch > 40)
    n_colored = int(colored_mask.sum()) or 1
    tot = small.shape[0] * small.shape[1] or 1

    # إذا البكسلات الملوّنة أقل من 8% → أبيض/رمادي/أسود
    if n_colored / tot < 0.08:
        return "other"

    # أقنعة الألوان على البكسلات الملوّنة فقط
    def masked_count(mask):
        return int(cv2.bitwise_and(mask, mask,
                   mask=colored_mask.astype(np.uint8)).sum() // 255)

    red   = cv2.bitwise_or(
                cv2.inRange(hsv,(0,  70,50),(12, 255,255)),
                cv2.inRange(hsv,(158,70,50),(180,255,255)))
    blue  = cv2.inRange(hsv,(95,  70,50),(135,255,255))
    yell  = cv2.inRange(hsv,(18,  80,80),( 40,255,255))
    grn   = cv2.inRange(hsv,(40,  60,50),( 90,255,255))

    scores = {
        'red':    masked_count(red)  / n_colored,
        'blue':   masked_count(blue) / n_colored,
        'yellow': masked_count(yell) / n_colored,
        'green':  masked_count(grn)  / n_colored,
    }
    best, score = max(scores.items(), key=lambda x: x[1])
    # عتبة 10% من البكسلات الملوّنة
    return best if score > 0.10 else "other"


# ══════════════════════════════════════════════════════
#  كلاس المسار
# ══════════════════════════════════════════════════════
class CarTrack:
    _ctr = 0
    def __init__(self, bbox, centroid):
        CarTrack._ctr += 1
        self.id              = CarTrack._ctr
        self.bbox            = bbox
        self.centroid        = centroid
        self.history         = deque([centroid], maxlen=HISTORY)
        self.speed_buf       = deque(maxlen=SMOOTH_N)
        self.speed_kmh       = 0.0
        self.missing         = 0
        self.life            = 1
        self.confirmed       = False
        self.color_label     = "other"
        self.size_label      = "small"
        self.deviated        = False
        self.deviation_event = False
        self.max_deviation   = 0.0
        self.collision       = False
        # px_per_m مُثبَّت عند أول ظهور — لا يتغير مع المنظور
        w = bbox[2]
        self.px_per_m        = max(w, 10) / CAR_REAL_LEN_M
        # نقاط LK للتتبع الدقيق
        self.lk_pts          = None

    def update(self, bbox, centroid, fps, lk_speed_px=None):
        self.history.append(centroid)
        self.bbox      = bbox
        self.centroid  = centroid
        self.missing   = 0
        self.life     += 1
        if self.life >= MIN_LIFE: self.confirmed = True

        # السرعة: نستخدم px_per_m المُثبَّت عند أول ظهور
        if lk_speed_px is not None and lk_speed_px > 0:
            # سرعة LK: px/frame × fps → px/s → m/s → km/h
            kmh = min((lk_speed_px * fps) / self.px_per_m * 3.6, SPEED_MAX)
        elif len(self.history) >= 2:
            p1, p2 = self.history[-2], self.history[-1]
            px_s   = math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * fps
            kmh    = min(px_s / self.px_per_m * 3.6, SPEED_MAX)
        else:
            kmh = 0.0

        if kmh >= 0:
            self.speed_buf.append(kmh)
            self.speed_kmh = sum(self.speed_buf) / len(self.speed_buf)

    def check_deviation(self):
        if len(self.history) < 8: return
        pts = list(self.history)
        origin_x = sum(p[0] for p in pts[:4]) / 4
        dev = abs(self.centroid[0] - origin_x)
        if dev > self.max_deviation: self.max_deviation = dev
        prev = self.deviated
        self.deviated = dev > LANE_DEV_THR
        self.deviation_event = self.deviated and not prev

    def cv_color(self):
        if self.speed_kmh <= SPEED_GREEN:  return CV_GREEN
        elif self.speed_kmh <= SPEED_YELLOW: return CV_YELLOW
        else:                                return CV_RED


# ══════════════════════════════════════════════════════
#  Optical Flow — حساب سرعة كل مستطيل
# ══════════════════════════════════════════════════════
# إعدادات Lucas-Kanade
_LK_PARAMS  = dict(winSize=(15,15), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,0.01))
_SHI_PARAMS = dict(maxCorners=LK_MAX_PTS, qualityLevel=0.2,
                   minDistance=5, blockSize=5)

def _sample_pts_in_box(gray, x, y, w, h):
    """يستخرج نقاط Shi-Tomasi داخل bbox"""
    x1,y1 = max(x,2), max(y,2)
    x2,y2 = min(x+w, gray.shape[1]-2), min(y+h, gray.shape[0]-2)
    if x2<=x1 or y2<=y1: return None
    mask = np.zeros(gray.shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **_SHI_PARAMS)
    return pts  # shape (N,1,2) or None

def compute_lk_speeds(prev_gray, curr_gray, tracks):
    """
    Lucas-Kanade optical flow per-track.
    يرجع dict: {track_id: median_displacement_px_per_frame}
    """
    result = {}
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return result

    for tid, tr in tracks.items():
        try:
            x,y,bw,bh = tr.bbox
            # إعادة أخذ النقاط كل 5 إطارات أو عند الفقدان
            if tr.lk_pts is None or tr.life % 5 == 0:
                tr.lk_pts = _sample_pts_in_box(prev_gray, x, y, bw, bh)
            if tr.lk_pts is None or len(tr.lk_pts) < 4:
                continue

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, tr.lk_pts, None, **_LK_PARAMS)

            if next_pts is None: continue
            good_prev = tr.lk_pts[status.flatten()==1]
            good_next = next_pts[status.flatten()==1]
            if len(good_prev) < 3: continue

            disps = np.linalg.norm(good_next - good_prev, axis=2).flatten()
            # استخدم الـ median لتجنب الضجيج والحواف
            result[tid] = float(np.median(disps))
            # احتفظ بالنقاط المتتبعة
            tr.lk_pts = good_next.reshape(-1,1,2)
        except Exception:
            tr.lk_pts = None
    return result


# ══════════════════════════════════════════════════════
#  إحصائيات الجلسة
# ══════════════════════════════════════════════════════
class SessionStats:
    def __init__(self):
        self.max_speed   = 0.0
        self.speed_sum   = 0.0
        self.speed_cnt   = 0
        self.dev_count   = 0
        self.max_dev     = 0.0
        self.dev_cars    = set()
        self.coll_cars   = set()
        self.peak_cars   = 0
        self.car_records = {}   # id → {max_speed, speed_sum, cnt, deviated, collided, color}

    def update(self, tracks):
        confirmed = [t for t in tracks.values() if t.confirmed]
        n = len(confirmed)
        if n > self.peak_cars: self.peak_cars = n

        for tr in confirmed:
            if tr.id not in self.car_records:
                self.car_records[tr.id] = dict(
                    id=tr.id, max_speed=0.0, speed_sum=0.0,
                    speed_cnt=0, deviated=False, collided=False,
                    color=tr.color_label)
            rec = self.car_records[tr.id]
            rec['color'] = tr.color_label

            s = tr.speed_kmh
            if s > 0:
                self.speed_sum += s; self.speed_cnt += 1
                rec['speed_sum'] += s; rec['speed_cnt'] += 1
                if s > rec['max_speed']:  rec['max_speed']  = s
                if s > self.max_speed:    self.max_speed     = s

            if tr.deviation_event: self.dev_count += 1
            if tr.deviated:
                rec['deviated'] = True
                self.dev_cars.add(tr.id)
                if tr.max_deviation > self.max_dev: self.max_dev = tr.max_deviation

            if tr.collision:
                rec['collided'] = True
                self.coll_cars.add(tr.id)

    @property
    def avg_speed(self):
        return self.speed_sum / self.speed_cnt if self.speed_cnt else 0


# ══════════════════════════════════════════════════════
#  مطابقة المسارات
# ══════════════════════════════════════════════════════
def iou(bA, bB):
    ax,ay,aw,ah=bA; bx,by,bw,bh=bB
    inter=(max(0,min(ax+aw,bx+bw)-max(ax,bx))*
           max(0,min(ay+ah,by+bh)-max(ay,by)))
    union=aw*ah+bw*bh-inter
    return inter/union if union else 0

def match_tracks(tracks, dets):
    matched={}; used=set()
    for di,(bbox,_) in enumerate(dets):
        best,tid=0,None
        for k,tr in tracks.items():
            if k in used: continue
            s=iou(tr.bbox,bbox)
            if s>best: best,tid=s,k
        if best>0.20: matched[di]=tid; used.add(tid)
    return matched

def check_collisions(tracks):
    conf={k:v for k,v in tracks.items() if v.confirmed}
    ids=list(conf)
    for i in ids: conf[i].collision=False
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            a,b=conf[ids[i]],conf[ids[j]]
            if math.hypot(a.centroid[0]-b.centroid[0],
                          a.centroid[1]-b.centroid[1]) < COLLISION_DIST:
                a.collision=b.collision=True


# ══════════════════════════════════════════════════════
#  حالة النظام
# ══════════════════════════════════════════════════════
class State:
    def __init__(self):
        self.tracks       = {}
        self.frame_idx    = 0
        self.total_frames = 0
        self.frame_bytes  = b''
        self.video_state  = "idle"
        self.paused       = False
        self.stop_req     = False
        self.snap_req     = False
        self.session      = SessionStats()
        self.lock         = threading.Lock()

state = State()


# ══════════════════════════════════════════════════════
#  كشف السيارات: YOLO أو fallback
# ══════════════════════════════════════════════════════
def detect_cars_yolo(frame, model):
    """يكشف السيارات بـ YOLO ويرجع list of (bbox, centroid)"""
    results = model(frame, classes=YOLO_CLASSES,
                    conf=YOLO_CONF, verbose=False)[0]
    dets = []
    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        w,h = x2-x1, y2-y1
        if w < 20 or h < 15: continue
        dets.append(((x1,y1,w,h), (x1+w//2, y1+h//2)))
    return dets

def detect_cars_bg(frame, bg_sub):
    """Fallback دقيق: background subtraction + تصفية صارمة"""
    kern3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # تحسين الإطار قبل الخلفية
    blr = cv2.GaussianBlur(frame,(7,7),0)
    fg  = bg_sub.apply(blr, learningRate=0.005)
    # إزالة الظلال (قيمة 127)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    # تنظيف مورفولوجي
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kern3, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kern5, iterations=3)
    fg = cv2.dilate(fg, kern5, iterations=1)

    cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if not (4000 < area < 160000): continue
        asp = w/h if h else 0
        if not (0.85 <= asp <= 5.0): continue
        # نسبة امتلاء المحيط الفعلي
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        fill = hull_area / (area or 1)
        if fill < 0.40: continue
        dets.append(((x,y,w,h),(x+w//2,y+h//2)))
    return dets


# ══════════════════════════════════════════════════════
#  خيط الفيديو
# ══════════════════════════════════════════════════════
def video_thread(source, single_pass=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); return

    fps          = max(cap.get(cv2.CAP_PROP_FPS) or 25, 10)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    with state.lock:
        state.total_frames = total_frames
        state.video_state  = "running"
        state.session      = SessionStats()

    # تحميل YOLO
    model  = get_yolo()
    bg_sub = cv2.createBackgroundSubtractorMOG2(400, 45, True) if not model else None

    prev_gray = None

    while not state.stop_req:
        if state.paused: time.sleep(0.04); continue

        ret, frame = cap.read()
        if not ret:
            with state.lock: state.video_state = "done"
            print("[INFO] Video finished.")
            break

        # تصغير
        fh,fw = frame.shape[:2]
        sc = min(1.0, 1280/fw, 720/fh)
        if sc < 1.0: frame = cv2.resize(frame,(int(fw*sc),int(fh*sc)))

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # كشف السيارات
        if model:
            dets = detect_cars_yolo(frame, model)
        else:
            dets = detect_cars_bg(frame, bg_sub)

        # حساب Lucas-Kanade قبل تحديث المسارات (نحتاج prev_gray)
        lk_speeds = compute_lk_speeds(prev_gray, curr_gray, state.tracks)

        with state.lock:
            mm  = match_tracks(state.tracks, dets)
            mid = set(mm.values())

            for di,(bbox,cen) in enumerate(dets):
                if di in mm:
                    tid   = mm[di]
                    lkspd = lk_speeds.get(tid)
                    state.tracks[tid].update(bbox, cen, fps, lkspd)
                else:
                    tr = CarTrack(bbox, cen)
                    x,y,w,h = bbox
                    tr.color_label = classify_color(frame[y:y+h, x:x+w])
                    tr.size_label  = "large" if w*h > 20000 else "small"
                    state.tracks[tr.id] = tr

            for tid in list(state.tracks):
                if tid not in mid:
                    state.tracks[tid].missing += 1
                    if state.tracks[tid].missing > MAX_MISSING:
                        del state.tracks[tid]

            for tr in state.tracks.values(): tr.check_deviation()
            check_collisions(state.tracks)
            state.session.update(state.tracks)
            state.frame_idx += 1

        prev_gray = curr_gray

        out = draw_frame(frame.copy())

        if state.snap_req:
            fn = f"snap_{state.frame_idx:05d}.png"
            cv2.imwrite(fn, out); print(f"[SNAP] {fn}")
            state.snap_req = False

        _,buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        state.frame_bytes = buf.tobytes()

    cap.release()
    if not single_pass:
        with state.lock:
            if state.video_state != "done":
                state.video_state = "idle"


# ══════════════════════════════════════════════════════
#  رسم الإطار
# ══════════════════════════════════════════════════════

# لون bbox حسب لون السيارة المكتشف (أطار خارجي رفيع)
_COLOR_MAP = {
    'red':    (40,  40,  200),
    'blue':   (200, 100,  40),
    'yellow': (0,   200, 220),
    'green':  (40,  200,  40),
    'other':  (160, 160, 160),
}

def draw_frame(frame):
    with state.lock:
        cars = [t for t in state.tracks.values() if t.confirmed]

    for tr in cars:
        x,y,w,h = tr.bbox
        spd_col = tr.cv_color()     # لون حسب السرعة (إطار داخلي)
        car_col = _COLOR_MAP.get(tr.color_label, (160,160,160))  # لون السيارة

        # تظليل اصطدام / انحراف
        if tr.collision or tr.deviated:
            ov = frame.copy()
            c  = (0,0,60) if tr.collision else (0,40,80)
            cv2.rectangle(ov,(x,y),(x+w,y+h),c,-1)
            cv2.addWeighted(ov,.12,frame,.88,0,frame)

        # ── إطار مزدوج: خارجي = لون السيارة، داخلي = لون السرعة ──
        cv2.rectangle(frame,(x-2,y-2),(x+w+2,y+h+2), car_col, 1)   # لون السيارة
        cv2.rectangle(frame,(x,y),(x+w,y+h), spd_col, 2)             # لون السرعة

        # زوايا صغيرة
        L=8
        for (px,py,dx,dy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame,(px,py),(px+dx*L,py),spd_col,2)
            cv2.line(frame,(px,py),(px,py+dy*L),spd_col,2)

        # السرعة
        spd = f"{tr.speed_kmh:.0f} km/h"
        (sw,sh),_ = cv2.getTextSize(spd,cv2.FONT_HERSHEY_SIMPLEX,.45,1)
        ty = max(y-5, sh+4)
        cv2.rectangle(frame,(x-1,ty-sh-3),(x+sw+4,ty+2),CV_BLACK,-1)
        cv2.putText(frame,spd,(x+2,ty),cv2.FONT_HERSHEY_SIMPLEX,.45,spd_col,1,cv2.LINE_AA)

        # اسم اللون تحت السرعة
        clr_ar = {'red':'أحمر','blue':'أزرق','yellow':'أصفر','green':'أخضر','other':'?'}
        cv2.putText(frame,clr_ar.get(tr.color_label,'?'),
                    (x+2,ty-sh-6),cv2.FONT_HERSHEY_SIMPLEX,.38,car_col,1,cv2.LINE_AA)

        # تحذيرات
        if tr.collision:
            cv2.putText(frame,"COLLISION",(x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,.50,CV_RED,2,cv2.LINE_AA)
        elif tr.deviated:
            cv2.putText(frame,"DEVIATION",(x,y+h+14),cv2.FONT_HERSHEY_SIMPLEX,.42,CV_YELLOW,1,cv2.LINE_AA)

        # مسار
        pts=list(tr.history)
        for k in range(1,len(pts)):
            a=k/len(pts)
            c=tuple(int(v*a) for v in spd_col)
            cv2.line(frame,pts[k-1],pts[k],c,1)

    # شريط علوي
    with state.lock:
        n  = len([t for t in state.tracks.values() if t.confirmed])
        nc = sum(1 for t in state.tracks.values() if t.collision and t.confirmed)
        mx = max((t.speed_kmh for t in state.tracks.values() if t.confirmed),default=0)
        fi = state.frame_idx
    h2,w2=frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w2,30),CV_BLACK,-1)
    cv2.line(frame,(0,30),(w2,30),(22,50,22),1)
    cv2.putText(frame,f"Cars:{n}  Max:{mx:.0f}km/h  Coll:{nc}  Frame:{fi}",
                (7,20),cv2.FONT_HERSHEY_SIMPLEX,.43,CV_GREEN,1,cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════
#  Flask
# ══════════════════════════════════════════════════════
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500*1024*1024  # 500MB للاستضافة المجانية

@app.route('/')
def index(): return HTML

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if state.frame_bytes:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+state.frame_bytes+b'\r\n'
            time.sleep(0.033)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with state.lock:
        conf = [t for t in state.tracks.values() if t.confirmed]
        objs = [{'id':t.id,'speed':round(t.speed_kmh,1),'color':t.color_label,
                 'size':t.size_label,'collision':t.collision,'deviated':t.deviated}
                for t in conf]
        avg = round(sum(o['speed'] for o in objs)/len(objs),1) if objs else 0
        mx  = round(max((o['speed'] for o in objs),default=0),1)
        ss  = state.session
        return jsonify(
            total=len(objs),
            red  =sum(1 for o in objs if o['color']=='red'),
            blue =sum(1 for o in objs if o['color']=='blue'),
            fast =sum(1 for o in objs if o['speed']>SPEED_YELLOW),
            collisions=sum(1 for o in objs if o['collision']),
            deviations=len(set(t.id for t in conf if t.deviated)),
            avg_speed=avg, max_speed=mx,
            frame=state.frame_idx, total_frames=state.total_frames,
            state=state.video_state, objects=objs,
            session_max_speed = round(ss.max_speed,1),
            session_avg_speed = round(ss.avg_speed,1),
            session_coll_count= len(ss.coll_cars),
            session_peak_cars = ss.peak_cars,
            session_cars=[{
                'id':r['id'],'color':r['color'],
                'max_speed':round(r['max_speed'],1),
                'avg_speed':round(r['speed_sum']/r['speed_cnt'],1) if r['speed_cnt'] else 0,
                'deviated':r['deviated'],'collided':r['collided'],
            } for r in ss.car_records.values()],
        )

@app.route('/control/<cmd>')
def control(cmd):
    if cmd=='pause':    state.paused   =not state.paused
    elif cmd=='stop':   state.stop_req =True
    elif cmd=='snapshot': state.snap_req=True
    return jsonify(ok=True,paused=state.paused,
                   filename=f"snap_{state.frame_idx:05d}.png")

@app.route('/upload_video',methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify(ok=False,error='No file')
    f=request.files['video']
    ext=os.path.splitext(f.filename)[1].lower()
    if ext not in('.mp4','.avi','.mov','.mkv','.wmv','.flv','.webm'):
        return jsonify(ok=False,error='Unsupported format')
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=ext)
    f.save(tmp.name);tmp.close()
    _start(tmp.name,True)
    return jsonify(ok=True)

@app.route('/switch_source/camera')
def switch_camera():
    return jsonify(ok=False, error='الكاميرا غير متاحة في نسخة الويب — ارفع فيديو')

def _start(src,sp):
    state.stop_req=True;time.sleep(0.35)
    state.stop_req=False;state.paused=False
    with state.lock:
        state.tracks.clear();state.frame_idx=0
        state.total_frames=0;state.video_state="idle"
        state.session=SessionStats()
    CarTrack._ctr=0
    threading.Thread(target=video_thread,args=(src,sp),daemon=True).start()


# ══════════════════════════════════════════════════════
#  نقطة الدخول
# ══════════════════════════════════════════════════════
# ── السحابة: لا كاميرا — فقط رفع فيديو ──────────────────────────
# يبدأ في وضع الانتظار حتى يرفع المستخدم فيديو من الواجهة
import sys

PORT = int(os.environ.get('PORT', 5000))

if __name__=='__main__':
    print("="*52)
    print("  Vehicle Tracker v6.0 — Cloud Mode")
    print(f"  Port: {PORT}")
    print("="*52)
    # لا نبدأ أي خيط فيديو — ننتظر رفع المستخدم
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)