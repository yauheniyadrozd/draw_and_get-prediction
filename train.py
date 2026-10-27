import argparse
import io
import json
import os
import urllib.parse
import urllib.request

import numpy as np

# Klasy Quick Draw (nazwy angielskie, jak w zbiorze). Polskie nazwy są w index.html.
CLASSES = [
    "cat", "dog", "house", "tree", "car", "bicycle", "sun", "moon", "star",
    "flower", "fish", "apple", "banana", "clock", "umbrella", "airplane",
    "guitar", "smiley face", "cup", "key",
]

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def http_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        if r.status == 200:  # serwer zignorował Range i oddał cały plik — tniemy lokalnie
            data = data[start:end + 1]
        return data


def fetch_class(name, n, cache_dir):
    """Pobiera tylko pierwsze n obrazków (28x28) danej klasy, bez ściągania całego ~100 MB pliku."""
    path = os.path.join(cache_dir, f"{name.replace(' ', '_')}_{n}.npy")
    if os.path.exists(path):
        return np.load(path)
    url = BASE + urllib.parse.quote(name) + ".npy"
    head = http_range(url, 0, 1023)
    f = io.BytesIO(head)
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(f)
    hlen = f.tell()
    n = min(n, shape[0])
    raw = http_range(url, hlen, hlen + n * 784 - 1)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, 784).copy()
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, arr)
    return arr


def build_dataset(per_class, cache_dir, fake=False):
    xs, ys = [], []
    for i, c in enumerate(CLASSES):
        if fake:
            rng = np.random.default_rng(i)
            arr = (rng.random((per_class, 784)) * 255).astype(np.uint8)
        else:
            print(f"  pobieram {c} ...", flush=True)
            arr = fetch_class(c, per_class, cache_dir)
        xs.append(arr)
        ys.append(np.full(len(arr), i, dtype=np.int64))
    x = np.concatenate(xs).astype(np.float32) / 255.0
    y = np.concatenate(ys)
    idx = np.random.default_rng(0).permutation(len(x))
    return x[idx].reshape(-1, 1, 28, 28), y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--fake", action="store_true", help="losowe dane — tylko test potoku eksportu")
    ap.add_argument("--out", default="model.js")
    args = ap.parse_args()
    if args.fast:
        args.per_class, args.epochs = 1500, 3

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    here = os.path.dirname(os.path.abspath(__file__))
    print("Dane...")
    x, y = build_dataset(args.per_class, os.path.join(here, "data"), fake=args.fake)
    n_val = len(x) // 10
    xv, yv = torch.from_numpy(x[:n_val]), torch.from_numpy(y[:n_val])
    xt, yt = torch.from_numpy(x[n_val:]), torch.from_numpy(y[n_val:])

    class Net(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.c1 = nn.Conv2d(1, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 32, 3, padding=1)
            self.c3 = nn.Conv2d(32, 64, 3, padding=1)
            self.d1 = nn.Linear(64 * 3 * 3, 128)
            self.d2 = nn.Linear(128, k)

        def forward(self, t):
            t = F.max_pool2d(F.relu(self.c1(t)), 2)  # 14x14
            t = F.max_pool2d(F.relu(self.c2(t)), 2)  # 7x7
            t = F.max_pool2d(F.relu(self.c3(t)), 2)  # 3x3
            t = t.flatten(1)
            t = F.relu(self.d1(t))
            return self.d2(t)

    net = Net(len(CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=args.epochs * ((len(xt) + 127) // 128)
    )
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 128):
            b = perm[i:i + 128]
            xb, yb = xt[b], yt[b]
            # prosta augmentacja: losowe przesunięcie o max 2 px
            dx, dy = np.random.randint(-2, 3, size=2)
            xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            acc = (net(xv).argmax(1) == yv).float().mean().item()
        print(f"epoka {ep + 1}/{args.epochs}  loss {tot / len(xt):.3f}  val acc {acc:.3f}", flush=True)

    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    layers = {
        k: [round(float(v), 5) for v in a.reshape(-1)] for k, a in sd.items()
    }
    out = {"classes": CLASSES, "layers": layers}
    with open(os.path.join(here, args.out), "w") as f:
        f.write("window.QD_MODEL = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"Zapisano {args.out}. Otwórz index.html.")


if __name__ == "__main__":
    main()
