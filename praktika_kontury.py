import sys
from collections import deque

import numpy as np
import matplotlib
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

rng = np.random.default_rng(42)

class Plant:
    def __init__(self, T_obj=8.0, delay=3.0, dt=0.2):
        self.T_obj = T_obj
        self.y = 20.0
        self.K = 0.9
        self.Tin = 20.0

        n = int(round(delay / dt))
        self.buf = deque([50.0] * n, maxlen=n)

    def step(self, u, dt, noise=0.05, noise_val=None):
        buf = self.buf
        u_eff = buf.popleft()

        buf.append(0.0 if u < 0.0 else (100.0 if u > 100.0 else u))
        dy = (self.Tin - self.y + self.K * u_eff) / self.T_obj
        if noise_val is None:
            noise_val = rng.normal(0.0, noise) if noise else 0.0
        self.y += dy * dt + noise_val
        return self.y

def scenario_K(t):
    if t < 250: return 0.9
    if t < 320: return 0.9 + (2.4 - 0.9) * (t - 250) / 70
    return 2.4

def scenario_Tin(t):
    return 20.0 if t < 450 else 12.0

def scenario_sp(t):
    if t < 150: return 60.0
    if t < 400: return 75.0
    return 65.0

class PID:
    def __init__(self, Kp=2.0, Ki=0.12, Kd=4.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.I = 0.0
        self.d_prev = None

    def reset(self):
        self.I = 0.0
        self.d_prev = None

    def update(self, sp, y, dt):
        e = sp - y

        if self.d_prev is None:
            d_term = 0.0
        else:
            dydt = (y - self.d_prev) / dt
            d_term = -self.Kd * dydt
        self.d_prev = y
        self.I += self.Ki * e * dt
        u = self.Kp * e + self.I + d_term

        u_raw = 0.0 if u < 0.0 else (100.0 if u > 100.0 else u)

        if (u > 100 and e > 0) or (u < 0 and e < 0):
            self.I -= self.Ki * e * dt
        return u_raw

class AdaptivePID(PID):

    N_DELAY = 15

    def __init__(self, Kp=2.0, Ki=0.12, Kd=4.0, K_nom=0.9):
        super().__init__(Kp, Ki, Kd)
        self.K_nom = K_nom
        self.K_est = K_nom
        self.theta = np.array([0.9, 0.03, 0.0])
        self.P = np.eye(3) * 1000.0
        self.lam = 0.999
        self.y_hist = [20.0]
        self.u_hist = [50.0] * (self.N_DELAY + 2)

    def reset(self):
        super().reset()
        self.K_est = self.K_nom
        self.theta = np.array([0.9, 0.03, 0.0])
        self.P = np.eye(3) * 1000.0
        self.y_hist = [20.0]
        self.u_hist = [50.0] * (self.N_DELAY + 2)

    def update(self, sp, y, dt):
        u = super().update(sp, y, dt)
        self.u_hist.append(u)

        phi = np.array([self.y_hist[-1], self.u_hist[-(self.N_DELAY + 1)], 1.0])
        err = y - phi @ self.theta
        Pphi = self.P @ phi
        kgain = Pphi / (self.lam + phi @ Pphi)
        self.theta = self.theta + kgain * err
        self.P = (self.P - np.outer(kgain, Pphi)) / self.lam
        self.y_hist.append(y)

        a, b = self.theta[0], self.theta[1]
        if 0.0 < a < 0.999:
            K_meas = np.clip(b / (1.0 - a), 0.3, 2.5)
            self.K_est += 0.1 * (K_meas - self.K_est)

            ratio = np.clip(self.K_nom / self.K_est, 0.35, 2.0)
            self.Kp, self.Ki, self.Kd = 2.0*ratio, 0.12*ratio, 4.0*ratio
        return u

ACTIONS = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
N_BINS = 11
N_STATES = N_BINS * N_BINS
N_ACT = len(ACTIONS)

def discretize(e, de):
    e_i = int(min(max((e + 20.0) / 40.0, 0.0), 0.999) * N_BINS)
    de_i = int(min(max((de + 8.0) / 16.0, 0.0), 0.999) * N_BINS)
    return e_i * N_BINS + de_i

def rl_reward(e, u, du):
    return (-0.25 * abs(e) - 0.0005 * (u - 50.0) ** 2
            - 0.02 * abs(du) - 0.5 * max(0.0, abs(e) - 10))

class QAgent:
    def __init__(self):
        self.Q = np.zeros((N_STATES, N_ACT))
        self.eps, self.alpha, self.gamma = 1.0, 0.25, 0.95

    def act(self, s, greedy=False):
        if not greedy and rng.random() < self.eps:
            return rng.integers(N_ACT)
        return int(np.argmax(self.Q[s]))

    def learn(self, s, a, r, s2):
        self.Q[s, a] += self.alpha * (r + self.gamma * np.max(self.Q[s2]) - self.Q[s, a])

def train_agent(episodes=1500, steps=300, dt=0.2, log_every=100):
    agent = QAgent()
    rewards = []
    for ep in range(episodes):
        plant = Plant()
        plant.K = rng.uniform(0.6, 2.6)
        plant.Tin = rng.uniform(8, 25)
        sp = rng.uniform(50, 85)
        plant.y = sp - rng.uniform(0, 40)
        pid = PID(Kp=2.0, Ki=0.12, Kd=4.0)
        y_prev = plant.y
        e, de = sp - plant.y, 0.0
        s = discretize(e, de)
        agent.eps = max(0.05, 1.0 * (0.995 ** ep))
        total = 0.0
        for _ in range(steps):
            a = agent.act(s)
            u_pid = pid.update(sp, plant.y, dt)
            u = np.clip(u_pid + ACTIONS[a], 0, 100)
            y = plant.step(u, dt)
            e_new = sp - y
            de_new = (y - y_prev) / dt
            r = rl_reward(e_new, u, ACTIONS[a])
            s2 = discretize(e_new, de_new)
            agent.learn(s, a, r, s2)
            s, e, de, y_prev = s2, e_new, de_new, y
            total += r
        rewards.append(total / steps)
        if (ep + 1) % log_every == 0:
            print(f"  эпизод {ep+1:4d}/{episodes}, средняя награда {rewards[-1]:.2f}")
    return agent, np.array(rewards)

def simulate(controller, T_end=600, dt=0.2, rl_agent=None):
    n = int(T_end / dt)
    t = np.arange(n) * dt
    y = np.zeros(n); u_arr = np.zeros(n); sp_arr = np.zeros(n); K_arr = np.zeros(n)
    plant = Plant()
    pid = controller if controller is not None else PID(Kp=2.0, Ki=0.12, Kd=4.0)
    if hasattr(pid, "reset"):
        pid.reset()
    e, de, y_prev = scenario_sp(0) - plant.y, 0.0, plant.y
    for k in range(n):
        sp = scenario_sp(t[k])
        plant.K = scenario_K(t[k])
        plant.Tin = scenario_Tin(t[k])
        u_pid = pid.update(sp, plant.y, dt)
        if rl_agent is not None:
            s = discretize(e, de)
            u = np.clip(u_pid + ACTIONS[rl_agent.act(s, greedy=True)], 0, 100)
        else:
            u = u_pid
        y[k] = plant.step(u, dt)
        e_new = sp - y[k]
        de = (y[k] - y_prev) / dt
        e, y_prev = e_new, y[k]
        u_arr[k], sp_arr[k], K_arr[k] = u, sp, plant.K
    return t, y, u_arr, sp_arr, K_arr

def metrics(t, y, sp):
    dt = t[1] - t[0]
    iae = np.sum(np.abs(sp - y)) * dt

    k0 = int(150 / dt); k1 = int(400 / dt)
    overshoot = max(0, (np.max(y[k0:k1]) - 75.0))
    return iae, overshoot


def fig_pid_vs_apc(res_fixed, res_adapt):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t1, y1, u1, sp1, K1 = res_fixed
    t2, y2, u2, sp2, K2 = res_adapt

    ax = axes[0]
    ax.plot(t1, sp1, "k--", lw=1.2, label="Уставка (задание)")
    ax.plot(t1, y1, color="#c0392b", lw=1.6, label="Классический ПИД")
    ax.plot(t2, y2, color="#1a7a3a", lw=1.6, label="Адаптивный ПИД + APC")
    ax.axvline(250, color="gray", ls=":", lw=1)
    ax.text(255, 78, "смена сырья\n(K: 0.9→2.4)", fontsize=8, color="gray")
    ax.axvline(450, color="gray", ls=":", lw=1)
    ax.text(455, 78, "возмущение\n(Tвх: 20→12 °C)", fontsize=8, color="gray")
    ax.set_ylabel("Температура y, °C")
    ax.set_title("Рис. 1. Регулирование температуры: классический ПИД "
                 "против адаптивного контура (APC)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(t1, u1, color="#c0392b", lw=1.4, label="ПИД: воздействие u")
    ax.plot(t2, u2, color="#1a7a3a", lw=1.4, label="APC: воздействие u")
    ax.set_xlabel("Время, с"); ax.set_ylabel("Положение клапана u, %")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("fig2_pid_apc.png", bbox_inches="tight")
    plt.close(fig)
    print("  сохранено: fig2_pid_apc.png")

def fig_metrics(res_fixed, res_adapt):
    iae_f, over_f = metrics(*res_fixed[:3])
    iae_a, over_a = metrics(*res_adapt[:3])
    labels = ["Классический\nПИД", "Адаптивный\nПИД + APC"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    axes[0].bar(labels, [iae_f, iae_a], color=["#c0392b", "#1a7a3a"])
    axes[0].set_title("IAE (интеграл ошибки),\n°C·с — меньше лучше")
    axes[1].bar(labels, [over_f, over_a], color=["#c0392b", "#1a7a3a"])
    axes[1].set_title("Перерегулирование, °C —\nменьше лучше")
    for ax in axes[:2]:
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width()/2,
                        p.get_height()), ha="center", va="bottom", fontsize=10)

    move_f = np.sum(np.abs(np.diff(res_fixed[2])))
    move_a = np.sum(np.abs(np.diff(res_adapt[2])))
    axes[2].bar(labels, [move_f, move_a], color=["#c0392b", "#1a7a3a"])
    axes[2].set_title("Суммарное движение\nклапана, % (износ ИМ)")
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width()/2,
                     p.get_height()), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Рис. 2. Показатели качества регулирования", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig3_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  сохранено: fig3_metrics.png")
    return iae_f, iae_a, over_f, over_a

def fig_rl_training(rewards, agent):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.plot(rewards, color="#bdc3c7", lw=0.8, alpha=0.6, label="награда за эпизод")
    w = 25
    ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
    ax.plot(np.arange(w-1, len(rewards)), ma, color="#8e44ad", lw=2.2,
            label="скользящее среднее (25)")
    ax.set_xlabel("Эпизод обучения"); ax.set_ylabel("Средняя награда")
    ax.set_title("Кривая обучения Q-агента", fontsize=11, fontweight="bold")
    ax.legend()

    ax = axes[1]

    q_e = agent.Q.reshape(N_BINS, N_BINS, N_ACT).mean(axis=1)
    im = ax.imshow(q_e.T, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_yticks(range(N_ACT), [f"{a:+.0f} %" for a in ACTIONS])
    ax.set_xticks(range(N_BINS),
                  [f"{-15 + 30*i/(N_BINS-1):+.0f}" for i in range(N_BINS)])
    ax.set_xlabel("Ошибка регулирования e = SP - y, °C")
    ax.set_ylabel("Действие (приращение клапана)")
    ax.set_title("Выученная политика Q(s, a):\nчто делает агент при данной ошибке",
                 fontsize=11, fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Q-значение")
    fig.suptitle("Рис. 3. Обучение агента усиленным обучением (Q-learning)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig("fig4_rl_training.png", bbox_inches="tight")
    plt.close(fig)
    print("  сохранено: fig4_rl_training.png")

def fig_rl_vs_pid(res_fixed, res_rl):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t1, y1, u1, sp1, K1 = res_fixed
    t2, y2, u2, sp2, K2 = res_rl

    ax = axes[0]
    ax.plot(t1, sp1, "k--", lw=1.2, label="Уставка (задание)")
    ax.plot(t1, y1, color="#c0392b", lw=1.6, label="Классический ПИД")
    ax.plot(t2, y2, color="#8e44ad", lw=1.6, label="RL-агент (Q-learning)")
    ax.axvline(250, color="gray", ls=":", lw=1)
    ax.text(255, 78, "смена сырья", fontsize=8, color="gray")
    ax.axvline(450, color="gray", ls=":", lw=1)
    ax.text(455, 78, "возмущение", fontsize=8, color="gray")
    ax.set_ylabel("Температура y, °C")
    ax.set_title("Рис. 4. Интеллектуальный контур (усиленное обучение) "
                 "против классического ПИД", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(t1, u1, color="#c0392b", lw=1.4, label="ПИД: воздействие u")
    ax.plot(t2, u2, color="#8e44ad", lw=1.4, label="RL-агент: воздействие u")
    ax.set_xlabel("Время, с"); ax.set_ylabel("Положение клапана u, %")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("fig5_rl_vs_pid.png", bbox_inches="tight")
    plt.close(fig)
    print("  сохранено: fig5_rl_vs_pid.png")

def main():
    print("=" * 68)
    print("Практическая часть: интеллектуальные контуры управления (APC + AI)")
    print("=" * 68)

    print("\n[1/4] Симуляция классического ПИД и адаптивного контура (APC)...")
    res_fixed = simulate(PID(Kp=2.0, Ki=0.12, Kd=4.0))
    res_adapt = simulate(AdaptivePID(Kp=2.0, Ki=0.12, Kd=4.0, K_nom=0.9))

    print("\n[2/4] Обучение RL-агента (Q-learning), 1500 эпизодов...")
    agent, rewards = train_agent(episodes=1500)

    print("\n[3/4] Тестовый прогон обученного RL-агента...")
    res_rl = simulate(None, rl_agent=agent)

    print("\n[4/4] Построение графиков...")
    fig_pid_vs_apc(res_fixed, res_adapt)
    iae_f, iae_a, over_f, over_a = fig_metrics(res_fixed, res_adapt)
    fig_rl_training(rewards, agent)
    fig_rl_vs_pid(res_fixed, res_rl)

    print("\n" + "=" * 68)
    print("ИТОГИ СРАВНЕНИЯ")
    print("=" * 68)
    print(f"IAE классический ПИД : {iae_f:8.0f} °C·с")
    print(f"IAE адаптивный (APC) : {iae_a:8.0f} °C·с  "
          f"(улучшение на {100*(1-iae_a/iae_f):.0f} %)")
    print(f"Перерегулирование ПИД: {over_f:8.1f} °C")
    print(f"Перерегулирование APC: {over_a:8.1f} °C")
    iae_rl, over_rl = metrics(res_rl[0], res_rl[1], res_rl[3])
    print(f"IAE RL-агент (Q)      : {iae_rl:8.0f} °C·с")
    print(f"Перерегулирование RL  : {over_rl:8.1f} °C")
    print("\nГотово. Все рисунки сохранены в текущей папке (fig1..fig5, PNG).")

if __name__ == "__main__":
    main()
