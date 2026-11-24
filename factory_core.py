import os, math, time, json, cv2
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
VIDEO_PATH = ''
INTERVAL_SEC = 2
MEMORY_SEC = 8
CHAIN_DEPTH = 2

# Roboflow YOLOv8 class names -> ontology classes.
YOLO_CLASS_MAP = {
    "HMI":"HMI", "box":"Box", "button":"Button", "control box":"ControlCabinet",
    "conveyor":"ConveyorBelt", "gripper":"Gripper", "logo":"Logo",
    "man":"Person", "pallet":"Pallet", "product":"Workpiece",
    "product_workpiece":"Workpiece", "robot":"Cobot", "robot arm":"Cobot",
    "text":"TextLabel", "tower light":"TowerLight", "workpiece rack":"WorkpieceRack",
}

def detect_yolo(frame, model, conf=0.25):
    """Return detections in the format consumed by the existing ontology pipeline."""
    result = model.predict(source=frame, conf=float(conf), imgsz=640, verbose=False)[0]
    dets = []
    if result.boxes is None:
        return dets
    for b in result.boxes:
        raw = str(model.names[int(b.cls.item())])
        cls = YOLO_CLASS_MAP.get(raw, raw.replace(' ', '_'))
        box = tuple(float(v) for v in b.xyxy[0].tolist())
        dets.append(dict(cls=cls, raw_cls=raw, conf=float(b.conf.item()), box=box, super=family(cls)))
    return nms_by_class(dets, thr=0.5)

def init_yolo_video(path):
    """Initialize ontology without requiring SIFT reference frames from a particular video."""
    global VIDEO_PATH, VW, VH, G_all
    VIDEO_PATH = path
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('영상을 열 수 없습니다. MP4 형식과 코덱을 확인하세요.')
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    G_all = nx.MultiDiGraph()
    for c, parent in PARENT.items():
        G_all.add_node(c, kind='class')
        G_all.add_node(parent, kind='class')
        G_all.add_edge(c, parent, label='subClassOf', t=-1.)
    for i, (_, _, name, desc) in enumerate(PROCESS_TIMELINE):
        G_all.add_node(name, kind='process', desc=desc)
        if i:
            G_all.add_edge(PROCESS_TIMELINE[i-1][2], name, label='precedes', t=-1.)
    POS.clear()

# ---------- 클래스 계층 (child -> parent) ----------
PARENT = {
    "PhysicalObject": "Entity", "Frame": "Entity", "Process": "Entity",
    "Agent": "PhysicalObject", "Person": "Agent", "Professor": "Person",
    "Equipment": "PhysicalObject", "Product": "PhysicalObject",
    "Robot": "Equipment", "Cobot": "Robot",
    "ControlDevice": "Equipment", "HMI": "ControlDevice",
    "EmergencyStop": "ControlDevice", "TowerLight": "ControlDevice",
    "TransportEquipment": "Equipment", "ConveyorBelt": "TransportEquipment",
    "ProcessStation": "Equipment", "StationModule": "ProcessStation",
    "PneumaticComponent": "Equipment", "PneumaticTube": "PneumaticComponent",
    "Enclosure": "Equipment", "ControlCabinet": "Enclosure",
    "Workpiece": "Product", "Box": "Product", "Pallet": "Product",
    "Gripper": "Equipment", "WorkpieceRack": "Equipment",
    "Button": "ControlDevice", "Logo": "PhysicalObject", "TextLabel": "PhysicalObject",
}
DETECTABLE = ["Professor","Cobot","HMI","EmergencyStop","TowerLight","ConveyorBelt",
              "PneumaticTube","StationModule","ControlCabinet","Workpiece"]
DETECT_METHOD = {
    "Professor":"Haar 얼굴검출 + 상체 확장", "Cobot":"SIFT 레퍼런스", "HMI":"SIFT 레퍼런스",
    "EmergencyStop":"SIFT + 빨간 원 색상검출", "TowerLight":"HSV 램프색 검출",
    "ConveyorBelt":"HSV 초록 벨트 + 가로로 긴 형태", "PneumaticTube":"HSV 파란색 군집",
    "StationModule":"SIFT 레퍼런스", "ControlCabinet":"SIFT 레퍼런스", "Workpiece":"어두운 사각형 + 흰 테두리",
}
FAMILY = {c: PARENT[c] for c in DETECTABLE}                 # 탐지 클래스 -> 바로 위 상위 클래스
def family(cls): return FAMILY.get(cls, cls)
def ancestors(cls):
    out = [cls]
    while out[-1] in PARENT: out.append(PARENT[out[-1]])
    return out
def is_a(cls, anc): return anc in ancestors(cls)

# ---------- 공정 타임라인 (영상 자막/내레이션 기준, 필요하면 수정) ----------
PROCESS_TIMELINE = [   # (시작초, 끝초, 이름, 설명)
    (0,   16,  "Intro",                  "영상 도입"),
    (16,  47,  "Lab_Overview",           "교수 소개 / 로지스틱스 랩 전경"),
    (47,  100, "HMI_Operation",          "SIMATIC HMI 조작, 비상정지 해제"),
    (100, 128, "P1_Pallet_Box_Supply",   "공정1: 팔레트·상자 공급"),
    (128, 168, "P2_Inspection_Reject",   "공정2: 색상·재질·높이 검사, 불량 USB 추출"),
    (168, 215, "P3_Laser_Marking",       "공정3: 그리퍼가 USB를 작업대로 이동, 레이저 각인"),
    (215, 1e9, "P4_Vision_Verification", "공정4: 산업용 카메라로 각인 글자 확인"),
]
def process_at(t):
    for s, e, name, _ in PROCESS_TIMELINE:
        if s <= t < e: return name
    return PROCESS_TIMELINE[-1][2]
PROCESS_NAMES = [p[2] for p in PROCESS_TIMELINE]

# ---------- 관계 스키마 (relation, domain, range, kind, 설명) ----------
SEMANTIC_RULES = [   # 클래스 수준 공리: (domain, relation, range, 설명). 둘 다 최근에 관측되면 인스턴스 간 관계 생성
    ("HMI",           "controls",            "ConveyorBelt",  "HMI가 컨베이어를 제어"),
    ("HMI",           "controls",            "StationModule", "HMI가 스테이션 제어"),
    ("HMI",           "controls",            "Cobot",         "HMI가 협동로봇 제어"),
    ("EmergencyStop", "part_of",             "HMI",           "비상정지 버튼은 HMI 패널의 일부"),
    ("EmergencyStop", "halts",               "StationModule", "비상정지가 설비를 정지"),
    ("TowerLight",    "indicates_status_of", "StationModule", "타워램프가 설비 상태 표시"),
    ("PneumaticTube", "actuates",            "StationModule", "공압 튜브가 실린더/그리퍼 구동"),
    ("ConveyorBelt",  "transports",          "Workpiece",     "컨베이어가 제품 이송"),
    ("StationModule", "processes",           "Workpiece",     "스테이션이 제품 가공/검사"),
    ("Cobot",         "handles",             "Workpiece",     "협동로봇이 제품 취급"),
    ("Professor",     "presents",            "Equipment",     "교수가 설비를 설명"),
]
RELATION_SCHEMA = [
    ("subClassOf",   "Class",          "Class",          "structural", "클래스 계층"),
    ("instanceOf",   "PhysicalObject", "Class",          "structural", "탐지된 객체 -> 클래스"),
    ("appearsIn",    "PhysicalObject", "Frame",          "temporal",   "객체가 해당 시점 프레임에 등장"),
    ("nextFrame",    "Frame",          "Frame",          "temporal",   "시간 순서"),
    ("inProcess",    "Frame",          "Process",        "temporal",   "프레임이 속한 공정"),
    ("participatesIn","PhysicalObject","Process",        "temporal",   "객체가 공정에 등장/참여"),
    ("precedes",     "Process",        "Process",        "temporal",   "공정 순서"),
    ("contains",     "Equipment",      "PhysicalObject", "spatial",    "A 박스가 B 박스를 포함"),
    ("on",           "Product",        "Equipment",      "spatial",    "제품이 컨베이어 위에 놓임"),
    ("adjacent_to",  "Equipment",      "Equipment",      "spatial",    "박스 간격이 가까움(서로 인접)"),
] + [(r, d, g, "semantic", desc) for d, r, g, desc in SEMANTIC_RULES]
SPATIAL_RELS  = {r[0] for r in RELATION_SCHEMA if r[3] == "spatial"}
SEMANTIC_RELS = {r[0] for r in RELATION_SCHEMA if r[3] == "semantic"}
REL_NODES     = SPATIAL_RELS | SEMANTIC_RELS

FAMILY_COLORS = {
    "Person":"#F6BD16","Robot":"#5B8FF9","ControlDevice":"#E8684A","TransportEquipment":"#61DDAA",
    "ProcessStation":"#9270CA","PneumaticComponent":"#78D3F8","Enclosure":"#8C9BB5","Product":"#FF9D4D",
}

# ---------- 레퍼런스 정의 (클래스: [(시각, 박스), ...]) ----------
REFERENCE_SPECS = {
    "Cobot":          [(43,  (720, 180, 1300, 920))],
    "ControlCabinet": [(43,  (158, 285, 607, 948))],
    "HMI":            [(62,  (37, 13, 1680, 1067))],
    "EmergencyStop":  [(62,  (1408, 395, 1700, 770))],
    "StationModule":  [(138, (791, 395, 1213, 790))],
}
MIN_INLIERS = {"Cobot":15, "ControlCabinet":12, "HMI":80, "EmergencyStop":40, "StationModule":40}  # 호모그래피 인라이어 최소 개수
MAX_PER_CLASS = {"PneumaticTube":3, "ConveyorBelt":1, "TowerLight":1}   # 장면마다 흔들리는 클래스는 대표 1~3개만 유지

# ---------- 공통 유틸 ----------
def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def wh(b): return (b[2]-b[0], b[3]-b[1])
def area(b): w, h = wh(b); return max(w, 0) * max(h, 0)
def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1]); return max(w, 0) * max(h, 0)
def iou(a, b):
    i = inter_area(a, b); u = area(a) + area(b) - i; return i / u if u > 0 else 0
def box_gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2])); dy = max(0, max(a[1], b[1]) - min(a[3], b[3])); return math.hypot(dx, dy)

def blobs(mask, min_area, close=9, dil=0):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    if dil: mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a >= min_area:
            x, y, w, h = cv2.boundingRect(c); out.append((x, y, x + w, y + h, a, c))
    return out

# ---------- 1) 교수: Haar 얼굴검출 ----------
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if hasattr(cv2, "CascadeClassifier") else None
if FACE is None: print("⚠ 이 OpenCV 버전엔 CascadeClassifier가 없어 교수 검출을 건너뜁니다 (Colab 기본 OpenCV 4.x에는 있음)")
def detect_professor(frame):
    if FACE is None: return []
    g = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    fs = FACE.detectMultiScale(g, 1.1, 6, minSize=(50, 50))
    if len(fs) == 0: return []
    x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
    x, y, w, h = x * 2, y * 2, w * 2, h * 2
    box = (max(0, x - 1.3*w), max(0, y - 0.4*h), min(VW-1, x + 2.3*w), min(VH-1, y + 6*h))
    return [dict(cls="Professor", conf=0.9, box=box)]

# ---------- 2) 색상 기반: 컨베이어 벨트 / 공압 튜브 / 타워램프 / 비상정지 ----------
def detect_color_equipment(frame):
    SC = 0.25; sm = cv2.resize(frame, None, fx=SC, fy=SC); H, W = sm.shape[:2]; fa = H * W
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    hsv[:int(H*0.20), :int(W*0.27)] = 0; hsv[:int(H*0.17), int(W*0.70):] = 0   # 좌상단 SMART FACTORY / 우상단 로고(워터마크) 제외
    up = lambda b: (b[0]/SC, b[1]/SC, b[2]/SC, b[3]/SC); res = []
    m = cv2.inRange(hsv, (95, 110, 70), (125, 255, 255))                          # 파란 공압 튜브
    for b in blobs(m, fa*0.004, close=7, dil=9): res.append(dict(cls="PneumaticTube", conf=min(0.95, b[4]/fa*20), box=up(b[:4])))
    m = cv2.inRange(hsv, (40, 70, 50), (85, 255, 255))                            # 초록 컨베이어 벨트(가로로 긴 띠)
    for x1, y1, x2, y2, a, c in blobs(m, fa*0.002, close=5):
        if (x2-x1) / max(y2-y1, 1) >= 5 and (x2-x1) > W*0.12: res.append(dict(cls="ConveyorBelt", conf=min(0.95, 0.5 + (x2-x1)/W), box=up((x1, y1, x2, y2))))   # 가장 긴 벨트를 대표로
    lamp = (cv2.inRange(hsv, (0,170,190), (10,255,255)) | cv2.inRange(hsv, (170,170,190), (180,255,255)) |
            cv2.inRange(hsv, (40,120,200), (85,255,255)) | cv2.inRange(hsv, (10,170,200), (28,255,255)))
    for x1, y1, x2, y2, a, c in blobs(lamp, fa*0.00015, close=3):                  # 타워램프(작고 밝은 빨강/초록/주황)
        w, h = x2-x1, y2-y1
        if a < fa*0.004 and 0.5 < w/max(h, 1) < 2.0:
            res.append(dict(cls="TowerLight", conf=0.6, box=up((x1-w*0.8, y1-h*1.5, x2+w*0.8, y2+h*3))))
    red = cv2.inRange(hsv, (0,150,80), (10,255,255)) | cv2.inRange(hsv, (170,150,80), (180,255,255))
    for x1, y1, x2, y2, a, c in blobs(red, fa*0.01, close=5):                      # 비상정지: 큰 빨간 원
        per = cv2.arcLength(c, True); circ = 4*math.pi*a/(per*per + 1e-6); w, h = x2-x1, y2-y1
        if circ > 0.6 and 0.7 < w/max(h, 1) < 1.4:
            res.append(dict(cls="EmergencyStop", conf=0.8, box=up((x1-w*0.25, y1-h*0.25, x2+w*0.25, y2+h*0.25))))
    return res

# ---------- 3) 제품(USB 박스): 어두운 사각형 + 흰 테두리 ----------
def detect_workpiece(frame):
    sm = cv2.resize(frame, None, fx=0.25, fy=0.25); H, W = sm.shape[:2]; fa = H * W
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    dark = cv2.morphologyEx((g < 60).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = []
    for c in cs:
        a = cv2.contourArea(c)
        if not (fa*0.003 < a < fa*0.04): continue
        x, y, w, h = cv2.boundingRect(c)
        if a/(w*h) < 0.7 or not (1.2 < w/h < 4): continue
        pad = max(2, int(0.25*min(w, h))); X1, Y1, X2, Y2 = max(0, x-pad), max(0, y-pad), min(W, x+w+pad), min(H, y+h+pad)
        ring = np.ones((Y2-Y1, X2-X1), np.uint8); ring[pad:pad+h, pad:pad+w] = 0
        v = hsv[Y1:Y2, X1:X2, 2][ring > 0]; s = hsv[Y1:Y2, X1:X2, 1][ring > 0]
        if len(v) and np.mean((v > 150) & (s < 60)) > 0.40:
            out.append(dict(cls="Workpiece", conf=0.6, box=(X1*4, Y1*4, X2*4, Y2*4)))
    return out

# ---------- 4) SIFT 레퍼런스 매칭 (협동로봇, HMI, 스테이션 ...) ----------
def detect_by_reference(frame):
    kp2, des2 = sift_feats(frame); res = []
    if des2 is None or len(des2) < 10: return res
    for cls, kp1, des1, (h, w) in REFS:
        if des1 is None: continue
        good = [m[0] for m in bf.knnMatch(des1, des2, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 12: continue
        src = np.float32([kp1[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        need = MIN_INLIERS.get(cls, 25)
        if Hm is None or int(mask.sum()) < need: continue
        corners = np.float32([[0, 0], [w*SC_SIFT, 0], [w*SC_SIFT, h*SC_SIFT], [0, h*SC_SIFT]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2) / SC_SIFT
        if not cv2.isContourConvex(proj.astype(np.float32).reshape(-1, 1, 2)): continue
        x1, y1 = proj.min(0); x2, y2 = proj.max(0)
        box = (max(0, x1), max(0, y1), min(VW-1, x2), min(VH-1, y2))
        if not (3000 < area(box) < VW*VH*1.5): continue
        res.append(dict(cls=cls, conf=min(1.0, int(mask.sum()) / (2.0 * need)), box=box))
    return res

def nms_by_class(dets, thr=0.4):
    keep = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        if all(not (k["cls"] == d["cls"] and iou(k["box"], d["box"]) > thr) for k in keep): keep.append(d)
    out, cnt = [], Counter()
    for d in keep:                                   # 클래스별 최대 개수 제한 (그래프 복잡도 관리)
        cnt[d["cls"]] += 1
        if cnt[d["cls"]] <= MAX_PER_CLASS.get(d["cls"], 3): out.append(d)
    return out

def detect_all(frame):
    dets = detect_professor(frame) + detect_color_equipment(frame) + detect_workpiece(frame) + detect_by_reference(frame)
    dets = nms_by_class(dets)
    hmi = [d for d in dets if d["cls"] == "HMI"]          # HMI 클로즈업 화면 안의 색 노이즈(파란 UI 등) 제거
    if hmi: dets = [d for d in dets if d["cls"] in ("HMI", "EmergencyStop") or inter_area(d["box"], hmi[0]["box"]) / max(area(d["box"]), 1) < 0.8]
    for d in dets: d["super"] = family(d["cls"])
    return dets
# ---------- 추적기: 2초 간격에서도 같은 객체에 같은 ID ----------
class Tracker:
    def __init__(self, max_dist=0.25, max_miss=3):
        self.tracks, self.count, self.max_dist, self.max_miss = {}, defaultdict(int), max_dist, max_miss
    def update(self, dets, diag):
        used = set()
        for d in sorted(dets, key=lambda x: -x["conf"]):
            cx, cy = center(d["box"]); best, bd = None, 1e9
            for tid, t in self.tracks.items():
                if tid in used or t["cls"] != d["cls"]: continue
                dist = math.hypot(cx - t["c"][0], cy - t["c"][1]) / diag
                if dist < bd: best, bd = tid, dist
            if best is not None and bd < self.max_dist: d["id"] = best
            else:
                self.count[d["cls"]] += 1; d["id"] = f"{d['cls']}_{self.count[d['cls']]}"
            used.add(d["id"]); self.tracks[d["id"]] = {"cls": d["cls"], "c": (cx, cy), "miss": 0}
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.max_miss: del self.tracks[tid]

# ---------- 공간 관계 (같은 프레임) ----------
NO_ADJ = {"PneumaticTube", "ConveyorBelt", "TowerLight"}   # 큰/가는 박스는 인접 관계에서 제외(노이즈 방지)
def infer_spatial(dets, diag):
    rels = []
    for A in dets:
        for B in dets:
            if A is B: continue
            a, b = A["box"], B["box"]; aa, ab = area(a), area(b)
            if aa <= 0 or ab <= 0: continue
            inter = inter_area(a, b)
            if inter > 0 and ab < aa and inter / ab > 0.85 and aa > ab * 1.5 and is_a(A["cls"], "Equipment") and A["cls"] not in NO_ADJ:
                rels.append((A["id"], "contains", B["id"])); continue
            if A["cls"] == "Workpiece" and B["cls"] == "ConveyorBelt":
                xo = min(a[2], b[2]) - max(a[0], b[0]); ha = a[3] - a[1]
                if xo > 0.5 * (a[2] - a[0]) and b[1] - 0.3*ha <= a[3] <= b[3] + 0.3*ha: rels.append((A["id"], "on", B["id"])); continue
            if (A["id"] < B["id"] and A["cls"] not in NO_ADJ and B["cls"] not in NO_ADJ and A["cls"] != B["cls"]
                    and is_a(A["cls"], "Equipment") and is_a(B["cls"], "Equipment") and inter / min(aa, ab) < 0.3 and box_gap(a, b) < 0.05 * diag):
                rels.append((A["id"], "adjacent_to", B["id"]))
    return rels

# ---------- 의미 관계: 온톨로지 규칙을 최근 관측 객체에 적용 ----------
def infer_semantic(active, last_seen):
    rels = []
    for iid, info in active.items():
        for dom, rel, rng, _ in SEMANTIC_RULES:
            if not is_a(info["cls"], dom): continue
            cands = [j for j, v in active.items() if j != iid and is_a(v["cls"], rng)]
            if not cands: continue
            c0 = last_seen[iid]["c"]
            j = min(cands, key=lambda k: math.hypot(last_seen[k]["c"][0] - c0[0], last_seen[k]["c"][1] - c0[1]))
            rels.append((iid, rel, j))
    return rels

# ---------- 누적 지식그래프 (전체 영상, 시간 속성 포함) ----------
G_all = nx.MultiDiGraph()
for c, p in PARENT.items():
    G_all.add_node(c, kind="class"); G_all.add_node(p, kind="class"); G_all.add_edge(c, p, label="subClassOf", t=-1.0)
for i, (_, _, name, desc) in enumerate(PROCESS_TIMELINE):
    G_all.add_node(name, kind="process", desc=desc)
    if i: G_all.add_edge(PROCESS_TIMELINE[i-1][2], name, label="precedes", t=-1.0)

def update_global(k, t, dets, rels, proc, prev_fid, graph=None):
    G_all = graph if graph is not None else globals()["G_all"]
    G_all.add_node(proc, kind="process")
    fid = f"Frame_{k:03d}"
    G_all.add_node(fid, kind="frame", time=float(t)); G_all.add_edge(fid, proc, label="inProcess", t=float(t))
    if prev_fid: G_all.add_edge(prev_fid, fid, label="nextFrame", t=float(t))
    for d in dets:
        if d["id"] not in G_all:
            G_all.add_node(d["id"], kind="instance", cls=d["cls"], first_seen=float(t)); G_all.add_edge(d["id"], d["cls"], label="instanceOf", t=float(t))
        G_all.nodes[d["id"]]["last_seen"] = float(t)
        G_all.add_edge(d["id"], fid, label="appearsIn", t=float(t), conf=round(d["conf"], 3), bbox=str([int(v) for v in d["box"]]))
        G_all.add_edge(d["id"], proc, label="participatesIn", t=float(t))
    for a, r, b in rels: G_all.add_edge(a, b, label=r, t=float(t), evidence="rule_candidate")
    return fid

# ---------- 현재 시점 스냅샷 ----------
def build_snapshot(t, active, rels, proc, prev_proc):
    G = nx.MultiDiGraph(); fnode = f"t={t:.2f}s"
    G.add_node(fnode, kind="frame", sup="Frame"); G.add_node(proc, kind="process", sup="Process"); G.add_edge(fnode, proc, label="inProcess")
    if prev_proc and prev_proc != proc:
        G.add_node(prev_proc, kind="process", sup="Process", status="gone"); G.add_edge(prev_proc, proc, label="precedes")
    for iid, info in active.items():
        cls, fam = info["cls"], family(info["cls"])
        G.add_node(iid, kind="instance", sup=fam, status=info["status"]); G.add_node(cls, kind="class", sup=fam)
        G.add_edge(iid, cls, label="instanceOf")
        if info["status"] == "present": G.add_edge(iid, proc, label="participatesIn")
        cur = cls
        for _ in range(CHAIN_DEPTH):
            p = PARENT.get(cur)
            if not p: break
            G.add_node(p, kind="class", sup=p); G.add_edge(cur, p, label="subClassOf"); cur = p
    for a, r, b in rels:
        if a in G and b in G: G.add_edge(a, b, label=r, evidence="rule_candidate")
    return G

# ---------- 렌더링: 영상 프레임 + 지식그래프 ----------
POS = {}
def layered_layout(G):
    """안정적인 층 구조 배치: 위=시간/공정, 가운데=인스턴스, 아래=클래스 계층"""
    pos = {}
    inst = sorted([n for n, d in G.nodes(data=True) if d["kind"] == "instance"], key=lambda n: (G.nodes[n]["sup"], n))
    m = len(inst)
    for i, n in enumerate(inst): pos[n] = ((-1 + 2 * (i + 0.5) / m) if m > 1 else 0.0, 0.18 * (i % 2))
    cur, seen, y = inst, set(inst), -0.8
    while cur:
        nxt = defaultdict(list)
        for n in cur:
            for _, v, d in G.out_edges(n, data=True):
                if d["label"] in ("instanceOf", "subClassOf") and v not in seen: nxt[v].append(pos[n][0])
        for v, xs in nxt.items(): pos[v] = (float(np.mean(xs)), y); seen.add(v)
        cur, y = list(nxt), y - 0.8
    gone = [n for n, d in G.nodes(data=True) if d["kind"] == "process" and d.get("status") == "gone"]
    live = [n for n, d in G.nodes(data=True) if d["kind"] == "process" and d.get("status") != "gone"]
    fr = [n for n, d in G.nodes(data=True) if d["kind"] == "frame"]
    for n in gone: pos[n] = (-0.65, 1.15)
    for n in live: pos[n] = (0.05, 1.15)
    for n in fr: pos[n] = (0.75, 1.15)
    return pos
EDGE_STYLE = {"taxonomy": ("#5B8FF9", "solid", 1.2), "process": ("#9AA5B1", "dotted", 1.0),
              "spatial": ("#E8A33D", "solid", 2.4), "semantic": ("#8E44AD", "solid", 2.4)}
def egroup(l):
    if l in ("instanceOf", "subClassOf"): return "taxonomy"
    if l in SPATIAL_RELS: return "spatial"
    if l in SEMANTIC_RELS: return "semantic"
    return "process"

def render(frame_bgr, dets, G, new_nodes, title):
    img = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"]); hexc = FAMILY_COLORS.get(d["super"], "#999999").lstrip("#")
        col = tuple(int(hexc[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 5); cv2.putText(img, d["id"], (x1 + 6, max(34, y1 + 34)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=80, gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax1.axis("off"); ax1.set_title("Video frame + OpenCV detections")
    pos = layered_layout(G)
    for g, (c, ls, w) in EDGE_STYLE.items():
        el = [(u, v) for u, v, dd in G.edges(data=True) if egroup(dd["label"]) == g]
        if el:
            cs = "arc3,rad=0.22" if g in ("spatial", "semantic") else "arc3,rad=0.0"
            nx.draw_networkx_edges(G, pos, edgelist=el, edge_color=c, style=ls, width=w, arrows=True, arrowsize=12, node_size=1300, connectionstyle=cs, ax=ax2)
    lab = {(u, v): dd["label"] for u, v, dd in G.edges(data=True) if egroup(dd["label"]) in ("spatial", "semantic") or dd["label"] == "precedes"}
    if lab: nx.draw_networkx_edge_labels(G, pos, edge_labels=lab, font_size=8, font_color="#6C3483", ax=ax2)
    for n, data in G.nodes(data=True):
        kind = data["kind"]; col = FAMILY_COLORS.get(data.get("sup"), "#D9D9D9")
        shape, size = {"frame": ("s", 2200), "process": ("p", 2600), "class": ("h", 1700), "instance": ("o", 1300)}[kind]
        if kind == "frame": col = "#FFD54F"
        if kind == "process": col = "#B9D6F2"
        faded, new = data.get("status") in ("gone", "recent"), n in new_nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_shape=shape, node_size=size, node_color=[col], alpha=0.3 if faded else 0.95,
                               edgecolors="red" if new else "#444444", linewidths=3.5 if new else 1, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax2)
    ax2.set_title(title); ax2.axis("off"); ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(min(p[1] for p in pos.values()) - 0.3, 1.45)
    ax2.text(0.01, 0.01, "square=time  pentagon=Process  hexagon=Class  circle=Instance | red edge=new, faded=recent\norange=spatial  purple=semantic  gray dotted=process/time",
             transform=ax2.transAxes, fontsize=8, color="#555")
    fig.tight_layout(); fig.canvas.draw(); arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    if LIVE_DISPLAY: clear_output(wait=True); display(fig)
    plt.close(fig); return arr

def init_video(path):
    global VIDEO_PATH, _cap, _fps, VW, VH, SX, SY, sift, bf, REFS, _crops, G_all, POS, grab, sift_feats, SC_SIFT
    VIDEO_PATH = path
    _cap = cv2.VideoCapture(VIDEO_PATH)
    _fps = _cap.get(cv2.CAP_PROP_FPS) or 30
    VW, VH = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SX, SY = VW / 1920.0, VH / 1080.0
    def grab(t):
        _cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * _fps)); ok, f = _cap.read(); return f if ok else None
    
    def show_grid(t, step=200):
        f = grab(t).copy()
        for x in range(0, VW, int(step * SX)):
            cv2.line(f, (x, 0), (x, VH), (0, 255, 255), 1); cv2.putText(f, str(int(x / SX)), (x + 3, 24), 0, 0.8, (0, 255, 255), 2)
        for y in range(0, VH, int(step * SY)):
            cv2.line(f, (0, y), (VW, y), (0, 255, 255), 1); cv2.putText(f, str(int(y / SY)), (3, y + 24), 0, 0.8, (0, 255, 255), 2)
        plt.figure(figsize=(12, 7)); plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); plt.title(f"t={t}s (1920x1080 좌표)"); plt.axis("off"); plt.show()
    
    # ---------- SIFT 레퍼런스 로드 ----------
    SC_SIFT = 0.5
    sift = cv2.SIFT_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    def sift_feats(img):
        g = cv2.cvtColor(cv2.resize(img, None, fx=SC_SIFT, fy=SC_SIFT), cv2.COLOR_BGR2GRAY)
        return sift.detectAndCompute(g, None)
    
    REFS = []   # (class, kp, des, (h, w))
    _crops = []
    for cls, specs in REFERENCE_SPECS.items():
        for t, (x1, y1, x2, y2) in specs:
            f = grab(t)
            if f is None: continue
            crop = f[int(y1*SY):int(y2*SY), int(x1*SX):int(x2*SX)]
            kp, des = sift_feats(crop)
            REFS.append((cls, kp, des, crop.shape[:2])); _crops.append((cls, crop))
            print(f"reference {cls:14s} t={t:>3}s  keypoints={len(kp)}")
    G_all = nx.MultiDiGraph()
    for c, p in PARENT.items():
        G_all.add_node(c, kind="class"); G_all.add_node(p, kind="class"); G_all.add_edge(c,p,label="subClassOf",t=-1.)
    for i, (_,_,name,desc) in enumerate(PROCESS_TIMELINE):
        G_all.add_node(name,kind="process",desc=desc)
        if i: G_all.add_edge(PROCESS_TIMELINE[i-1][2],name,label="precedes",t=-1.)
    POS.clear()
