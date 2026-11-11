"""xw.py: X11 helpers by window id or name. usage:
  xw.py list
  xw.py shot ID|NAME OUT.png
  xw.py click ID|NAME X Y      (window-relative)
"""
import sys, time
from Xlib import display, X
from Xlib.ext import xtest
from PIL import Image
d = display.Display(); root = d.screen().root

def windows():
    out = []
    def walk(w):
        try:
            a = w.get_attributes(); g = w.get_geometry()
            if a.map_state == X.IsViewable and g.width > 50 and g.height > 50:
                out.append((w, w.get_wm_name(), g.width, g.height))
        except Exception:
            pass
        for c in w.query_tree().children:
            walk(c)
    walk(root)
    return out

def pick(spec):
    if spec == 'root':
        return root
    ws = windows()
    if spec.isdigit():
        for w, n, a, b in ws:
            if w.id == int(spec): return w
    best = [t for t in ws if t[1] and spec in t[1]]
    best.sort(key=lambda t: -t[2] * t[3])
    return best[0][0] if best else None

def origin(w):
    t = w.translate_coords(root, 0, 0); return -t.x, -t.y

cmd = sys.argv[1]
if cmd == "list":
    for w, n, a, b in windows():
        print(w.id, repr(n), a, b, origin(w))
    sys.exit()
w = pick(sys.argv[2])
if not w: sys.exit("window not found")
if cmd == "shot":
    g = w.get_geometry()
    raw = w.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xffffffff)
    Image.frombytes("RGB", (g.width, g.height), raw.data, "raw", "BGRX").save(sys.argv[3])
    print("saved", sys.argv[3], g.width, g.height)
elif cmd == "drag":  # drag ID X1 Y1 X2 Y2
    ox, oy = origin(w); x1, y1, x2, y2 = map(int, sys.argv[3:7])
    xtest.fake_input(d, X.MotionNotify, x=ox + x1, y=oy + y1); d.sync(); time.sleep(0.2)
    xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.2)
    for k in range(1, 21):
        xtest.fake_input(d, X.MotionNotify, x=ox + x1 + (x2 - x1) * k // 20, y=oy + y1 + (y2 - y1) * k // 20)
        d.sync(); time.sleep(0.03)
    time.sleep(0.2); xtest.fake_input(d, X.ButtonRelease, 1); d.sync()
    print("dragged")
elif cmd in ("rclick", "dclick"):
    ox, oy = origin(w); x, y = int(sys.argv[3]), int(sys.argv[4])
    xtest.fake_input(d, X.MotionNotify, x=ox + x, y=oy + y); d.sync(); time.sleep(0.1)
    b, n = (3, 1) if cmd == "rclick" else (1, 2)
    for _ in range(n):
        xtest.fake_input(d, X.ButtonPress, b); xtest.fake_input(d, X.ButtonRelease, b); d.sync(); time.sleep(0.05)
    print(cmd)
elif cmd == "key":  # key ID KEYSYM[+MOD]
    from Xlib import XK
    codes = [d.keysym_to_keycode(XK.string_to_keysym(p)) for p in sys.argv[3].split("+")]
    for c in codes: xtest.fake_input(d, X.KeyPress, c)
    for c in reversed(codes): xtest.fake_input(d, X.KeyRelease, c)
    d.sync(); print("key", sys.argv[3])
elif cmd == "move":
    ox, oy = origin(w); xtest.fake_input(d, X.MotionNotify, x=ox + int(sys.argv[3]), y=oy + int(sys.argv[4])); d.sync()
elif cmd == "type":  # type ID TEXT (window id only used to make sure it exists)
    from Xlib import XK
    names = {"/": ("slash", 0), "_": ("underscore", 1), ".": ("period", 0), "-": ("minus", 0),
             " ": ("space", 0), "~": ("asciitilde", 1), ":": ("colon", 1)}
    shift = d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))
    for ch in sys.argv[3]:
        if ch in names:
            ks, sh = names[ch]
        else:
            ks, sh = ch, 1 if ch.isupper() else 0
        code = d.keysym_to_keycode(XK.string_to_keysym(ks))
        if sh: xtest.fake_input(d, X.KeyPress, shift)
        xtest.fake_input(d, X.KeyPress, code); xtest.fake_input(d, X.KeyRelease, code)
        if sh: xtest.fake_input(d, X.KeyRelease, shift)
        d.sync(); time.sleep(0.02)
    print("typed")
elif cmd == "scroll":  # scroll ID X Y N (+down/-up)
    ox, oy = origin(w); x, y, n = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    xtest.fake_input(d, X.MotionNotify, x=ox + x, y=oy + y); d.sync(); time.sleep(0.1)
    b = 5 if n > 0 else 4
    for _ in range(abs(n)):
        xtest.fake_input(d, X.ButtonPress, b); xtest.fake_input(d, X.ButtonRelease, b); d.sync(); time.sleep(0.05)
    print("scrolled", n)
elif cmd == "click":
    ox, oy = origin(w); x, y = int(sys.argv[3]), int(sys.argv[4])
    xtest.fake_input(d, X.MotionNotify, x=ox + x, y=oy + y); d.sync(); time.sleep(0.1)
    xtest.fake_input(d, X.ButtonPress, 1); xtest.fake_input(d, X.ButtonRelease, 1); d.sync()
    print("clicked", ox + x, oy + y)
