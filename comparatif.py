```python
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def build_ssh(N_cells, v, w):
    dim = 2 * N_cells
    H = np.zeros((dim, dim))
    for i in range(N_cells):
        H[2*i, 2*i+1] = H[2*i+1, 2*i] = v
        if i < N_cells - 1:
            H[2*i+1, 2*i+2] = H[2*i+2, 2*i+1] = w
    return H

def winding_ssh(v, w, Nk=800):
    k = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    dx = v + w * np.cos(k)
    dy = w * np.sin(k)
    phi = np.arctan2(dy, dx)
    return int(np.round(np.sum(np.diff(np.unwrap(phi))) / (2*np.pi)))

def build_kitaev(N, t, delta, mu):
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H[i, i] = -mu
        H[N+i, N+i] = mu
        if i < N - 1:
            H[i, i+1] = H[i+1, i] = -t
            H[N+i, N+i+1] = H[N+i+1, N+i] = t
            H[i, N+i+1] = delta
            H[N+i+1, i] = np.conj(delta)
            H[i+1, N+i] = -delta
            H[N+i, i+1] = -np.conj(delta)
    return H

def winding_kitaev(t, delta, mu, Nk=800):
    k = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    dy = -2 * delta * np.sin(k)
    dz = -mu - 2 * t * np.cos(k)
    phi = np.arctan2(dy, dz)
    return int(np.round(np.sum(np.diff(np.unwrap(phi))) / (2*np.pi)))

if __name__ == "__main__":
    print("COMPARAISON SSH <-> KITAEV")
    N_ssh = 40
    H = build_ssh(N_ssh, v=1.0, w=0.5)
    print(f"SSH Triviale : W = {winding_ssh(1.0, 0.5)}")
    H = build_ssh(N_ssh, v=0.5, w=1.0)
    print(f"SSH Topologique : W = {winding_ssh(0.5, 1.0)}")
    
    N_kit = 50
    print(f"Kitaev Triviale : W = {winding_kitaev(1.0, 1.0, 3.0)}")
    print(f"Kitaev Topologique : W = {winding_kitaev(1.0, 1.0, 0.5)}")
```
