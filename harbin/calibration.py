from __future__ import annotations

import math
import numpy as np


def brier_score(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6); return float(np.mean((p-y)**2))


def log_loss_score(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def expected_calibration_error(y,p,bins=10):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); total=len(y)
    if not total: return float("nan")
    edges=np.linspace(0,1,bins+1); ece=0.0
    for i in range(bins):
        mask=(p>=edges[i]) & (p<(edges[i+1]) if i<bins-1 else p<=edges[i+1])
        n=int(mask.sum())
        if n: ece += (n/total)*abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(ece)


def calibration_table(y,p,bins=10):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); rows=[]; edges=np.linspace(0,1,bins+1)
    for i in range(bins):
        mask=(p>=edges[i]) & (p<(edges[i+1]) if i<bins-1 else p<=edges[i+1])
        if mask.sum(): rows.append({"bin":i,"n":int(mask.sum()),"predicted":float(p[mask].mean()),"observed":float(y[mask].mean())})
    return rows
