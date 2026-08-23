"""Functional decord shim for aarch64 (no decord wheel): imageio-backed VideoReader with
torch bridge — implements the surface VBench's utils.load_video uses."""
import types, sys
import numpy as np
import torch
import imageio.v2 as iio

_BRIDGE = {"mode": "native"}

class _Batch:
    def __init__(self, arr): self._a = arr
    def asnumpy(self): return self._a

class VideoReader:
    def __init__(self, path, num_threads=1, ctx=None, width=None, height=None):
        rd = iio.get_reader(path)
        self._frames = [f for f in rd]
        rd.close()
        self._fps = 16.0
        try:
            self._fps = float(iio.get_reader(path).get_meta_data().get("fps", 16.0))
        except Exception:
            pass
    def __len__(self): return len(self._frames)
    def get_avg_fps(self): return self._fps
    def get_batch(self, idxs):
        arr = np.stack([self._frames[int(i)] for i in idxs])
        if _BRIDGE["mode"] == "torch":
            return torch.from_numpy(arr)
        return _Batch(arr)
    def __getitem__(self, i):
        arr = self._frames[int(i)]
        return torch.from_numpy(arr) if _BRIDGE["mode"] == "torch" else _Batch(arr)

def cpu(_id=0): return None

def _set_bridge(mode): _BRIDGE["mode"] = mode

def install():
    m = types.ModuleType("decord")
    m.VideoReader = VideoReader
    m.cpu = cpu
    br = types.ModuleType("decord.bridge")
    br.set_bridge = _set_bridge
    m.bridge = br
    sys.modules["decord"] = m
    sys.modules["decord.bridge"] = br
