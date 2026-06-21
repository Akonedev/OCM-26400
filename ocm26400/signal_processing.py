"""Traitement du signal + contrôleur PID + spectre — réfute audit SIG.

EX-B289-296. Traitement du signal RÉEL (lié au noyau spectral FFT de l'archi) :
* PID controller : régulation Proportionnelle-Intégrale-Dérivée (servo, contrôle).
* Spectre FFT : amplitude/fréquences d'un signal (réutilise torch.fft — cohérent avec
  SpectralCoreBlock).
* Filtres : passe-bas (moyenne mobile), passe-haut, soustraction DC.
* Stats signal : RMS, énergie, zero-crossing rate.
Vérifiable : PID converge vers la consigne, FFT détecte les fréquences injectées.
"""
from __future__ import annotations
import math
from typing import List, Tuple
import torch


# ============ PID controller ============

class PIDController:
    """PID : u(t) = Kp·e + Ki·∫e dt + Kd·de/dt. e = consigne − mesure."""

    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01,
                 setpoint: float = 1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, measurement: float, dt: float = 0.1) -> float:
        """Un pas : mesure → commande u."""
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def simulate(self, process_fn, n_steps: int = 100, dt: float = 0.1, init: float = 0.0
                 ) -> List[float]:
        """Simule la boucle fermée. process_fn(state, u, dt) → new_state."""
        state = init
        history = [state]
        for _ in range(n_steps):
            u = self.step(state, dt)
            state = process_fn(state, u, dt)
            history.append(state)
        return history


def simulate_pid_to_setpoint(setpoint: float = 1.0, n_steps: int = 200) -> dict:
    """Démo : un processus intégrant (state += u*dt) régulé par PID → converge à la consigne."""
    pid = PIDController(kp=2.0, ki=1.0, kd=0.05, setpoint=setpoint)
    hist = pid.simulate(lambda s, u, dt: s + u * dt, n_steps=n_steps, dt=0.1)
    final = hist[-1]
    return {"setpoint": setpoint, "final_state": round(final, 4),
            "converged": abs(final - setpoint) < 0.1,
            "overshoot": round(max(hist) - setpoint, 4) if hist else 0.0}


# ============ Spectre FFT ============

def spectrum(signal: List[float]) -> Tuple[List[float], List[float]]:
    """FFT → (amplitudes, fréquences) du signal. Réutilise torch.fft (cohérent SpectralCoreBlock)."""
    x = torch.tensor(signal, dtype=torch.float32)
    X = torch.fft.rfft(x)
    amp = torch.abs(X).tolist()
    freqs = torch.fft.rfftfreq(len(signal)).tolist()
    return amp, freqs


def dominant_frequency(signal: List[float], sample_rate: float = 1.0) -> float:
    """Fréquence dominante du signal (pic spectral)."""
    amp, freqs = spectrum(signal)
    if not amp:
        return 0.0
    idx = max(range(len(amp)), key=lambda i: amp[i])
    return freqs[idx] * sample_rate


# ============ Filtres ============

def lowpass_moving_average(signal: List[float], window: int = 3) -> List[float]:
    """Passe-bas : moyenne mobile (lisse le bruit haute fréquence)."""
    out = []
    for i in range(len(signal)):
        lo = max(0, i - window // 2)
        hi = min(len(signal), i + window // 2 + 1)
        out.append(sum(signal[lo:hi]) / (hi - lo))
    return out


def highpass(signal: List[float]) -> List[float]:
    """Passe-haut : signal − passe-bas (garde la haute fréquence)."""
    return [s - l for s, l in zip(signal, lowpass_moving_average(signal, 5))]


def remove_dc(signal: List[float]) -> List[float]:
    """Soustrait la composante continue (moyenne)."""
    m = sum(signal) / len(signal) if signal else 0
    return [s - m for s in signal]


# ============ Stats ============

def rms(signal: List[float]) -> float:
    """Root Mean Square (énergie)."""
    if not signal:
        return 0.0
    return math.sqrt(sum(s * s for s in signal) / len(signal))


def zero_crossing_rate(signal: List[float]) -> int:
    """Nombre de changements de signe (caractéristique du signal, ex. voisement)."""
    return sum(1 for i in range(1, len(signal))
               if (signal[i] >= 0) != (signal[i - 1] >= 0))


if __name__ == "__main__":
    print("[signal] PID →", simulate_pid_to_setpoint(1.0))
    # signal 2 fréquences
    t = [i / 100 for i in range(200)]
    sig = [math.sin(2 * math.pi * 5 * ti) + 0.5 * math.sin(2 * math.pi * 20 * ti) for ti in t]
    fdom = dominant_frequency(sig, sample_rate=100)
    print(f"[signal] freq dominante (injecté 5Hz+20Hz) ≈ {fdom:.1f} Hz")
    print(f"[signal] RMS={rms(sig):.3f} | zero-crossings={zero_crossing_rate(sig)}")
