import sys
import numpy as np
from collections import deque

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QSlider, QLabel, QPushButton, QComboBox, QProgressBar,
    QMessageBox,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from praktika_kontury import (
    Plant, PID, AdaptivePID, QAgent, ACTIONS, N_BINS, discretize, rl_reward,
)

DT = 0.2
WINDOW = 300.0
HIST = int(WINDOW / DT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

class TrainThread(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(object)

    EPISODES = 600
    STEPS = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        rng = np.random.default_rng(7)
        agent = QAgent()
        actions = ACTIONS
        steps = self.STEPS
        for ep in range(self.EPISODES):
            if self._stop:
                return
            plant = Plant()
            plant.K = rng.uniform(0.6, 2.6)
            plant.Tin = rng.uniform(8, 25)
            sp = rng.uniform(50, 85)
            plant.y = sp - rng.uniform(0, 40)
            pid = PID(Kp=2.0, Ki=0.12, Kd=4.0)

            plant_step, pid_update = plant.step, pid.update
            agent_act, agent_learn = agent.act, agent.learn
            noise = rng.normal(0.0, 0.05, steps)
            e, de = sp - plant.y, 0.0
            y_prev = plant.y
            s = discretize(e, de)
            agent.eps = max(0.05, 0.995 ** ep)
            for i in range(steps):
                a = agent_act(s)
                u = pid_update(sp, plant.y, DT) + actions[a]
                if u < 0.0:
                    u = 0.0
                elif u > 100.0:
                    u = 100.0
                y = plant_step(u, DT, noise_val=noise[i])
                e_new = sp - y
                de = (y - y_prev) / DT
                s2 = discretize(e_new, de)
                agent_learn(s, a, rl_reward(e_new, u, actions[a]), s2)
                s, e, y_prev = s2, e_new, y
            if (ep + 1) % 10 == 0:
                self.progress.emit(int(100 * (ep + 1) / self.EPISODES))
        agent.eps = 0.0
        self.done.emit(agent)

MODE_PID, MODE_APC, MODE_RL = 0, 1, 2
MODE_NAMES = ["Классический ПИД", "Адаптивный ПИД (APC)", "ПИД + RL (ИИ)"]

APP_QSS = """
QWidget {
    background-color: #12151a;
    color: #dfe6ee;
    font-size: 13px;
}
QGroupBox {
    background-color: #1a1f27;
    border: 1px solid #2e3642;
    border-radius: 10px;
    margin-top: 16px;
    padding: 10px 10px 12px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #8ab4ff;
}
QLabel { background: transparent; }
QLabel#valLbl {
    color: #f5c542;
    font-family: 'Consolas', monospace;
    font-weight: 700;
    font-size: 14px;
    min-width: 46px;
    qproperty-alignment: AlignRight;
}
QLabel#indVal {
    color: #7ee787;
    font-family: 'Consolas', monospace;
    font-weight: 700;
    font-size: 15px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #2e3642;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #3b82f6, stop:1 #60a5fa);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: #e8eef5;
    border: 1px solid #8ab4ff;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QPushButton {
    background-color: #242b35;
    border: 1px solid #39424f;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #2d3642; border-color: #4a5568; }
QPushButton:pressed { background-color: #1c222b; }
QPushButton#btnPrimary {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
}
QPushButton#btnPrimary:hover  { background-color: #3b82f6; }
QPushButton#btnPrimary:pressed { background-color: #1d4ed8; }
QPushButton#btnDanger {
    background-color: #3d1d22;
    border-color: #7f2d38;
    color: #ff9aa5;
}
QPushButton#btnDanger:hover  { background-color: #55252c; }
QPushButton:disabled { color: #6b7684; border-color: #2e3642; }
QComboBox {
    background-color: #242b35;
    border: 1px solid #39424f;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 600;
}
QComboBox:hover { border-color: #4a5568; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #8ab4ff;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1a1f27;
    border: 1px solid #39424f;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QProgressBar {
    background-color: #242b35;
    border: 1px solid #39424f;
    border-radius: 7px;
    text-align: center;
    color: #dfe6ee;
    font-size: 11px;
    max-height: 15px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #7c3aed, stop:1 #a78bfa);
    border-radius: 6px;
}
"""

PLT_BG, PLT_AXBG = "#12151a", "#1a1f27"
PLT_FG, PLT_GRID = "#dfe6ee", "#2e3642"
PLT_SP, PLT_Y = "#f5c542", "#ff5c5c"
PLT_U, PLT_TRIM = "#4da3ff", "#c084fc"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симулятор интеллектуального контура управления "
                            "(APC + AI)")
        self.resize(1220, 720)
        self.setStyleSheet(APP_QSS)
        self._e_state = None

        self.plant = Plant()
        self.agent = None
        self.iae = 0.0
        self.pulse = 0
        self.t = 0.0
        self.running = True
        self.trim = 0.0

        self.h_t = deque(maxlen=HIST)
        self.h_y = deque(maxlen=HIST)
        self.h_sp = deque(maxlen=HIST)
        self.h_u = deque(maxlen=HIST)
        self.h_trim = deque(maxlen=HIST)
        self._plot_skip = 3

        self._plot_cnt = 0

        self._build_ui()
        self._reset_model()
        self._start_training()

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 14, 12, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        root.addLayout(left, 0)

        gb_mode = QGroupBox("Контур управления")
        lm = QVBoxLayout(gb_mode)
        self.combo = QComboBox()
        self.combo.addItems(MODE_NAMES)
        self.combo.setCurrentIndex(MODE_PID)
        self.combo.currentIndexChanged.connect(self._mode_changed)
        lm.addWidget(self.combo)
        self.lbl_rl = QLabel("ИИ: обучение…")
        lm.addWidget(self.lbl_rl)
        self.bar_rl = QProgressBar()
        self.bar_rl.setRange(0, 100)
        lm.addWidget(self.bar_rl)
        left.addWidget(gb_mode)

        gb_set = QGroupBox("Управление процессом (действия оператора)")
        gs = QGridLayout(gb_set)

        gs.addWidget(QLabel("Уставка SP, °C"), 0, 0)
        self.sl_sp = QSlider(Qt.Orientation.Horizontal)
        self.sl_sp.setRange(40, 90); self.sl_sp.setValue(60)
        gs.addWidget(self.sl_sp, 0, 1)
        self.val_sp = QLabel("60"); self.val_sp.setObjectName("valLbl")
        gs.addWidget(self.val_sp, 0, 2)
        self.sl_sp.valueChanged.connect(
            lambda v: self.val_sp.setText(str(v)))

        gs.addWidget(QLabel("Сырьё: коэффициент K"), 1, 0)
        self.sl_k = QSlider(Qt.Orientation.Horizontal)
        self.sl_k.setRange(6, 26); self.sl_k.setValue(9)
        gs.addWidget(self.sl_k, 1, 1)
        self.val_k = QLabel("0.90"); self.val_k.setObjectName("valLbl")
        gs.addWidget(self.val_k, 1, 2)
        self.sl_k.valueChanged.connect(
            lambda v: self.val_k.setText(f"{v/10:.2f}"))

        gs.addWidget(QLabel("T входного потока, °C"), 2, 0)
        self.sl_tin = QSlider(Qt.Orientation.Horizontal)
        self.sl_tin.setRange(5, 30); self.sl_tin.setValue(20)
        gs.addWidget(self.sl_tin, 2, 1)
        self.val_tin = QLabel("20"); self.val_tin.setObjectName("valLbl")
        gs.addWidget(self.val_tin, 2, 2)
        self.sl_tin.valueChanged.connect(
            lambda v: self.val_tin.setText(str(v)))

        btn_pulse = QPushButton("⚡  Импульсное возмущение +10 °C")
        btn_pulse.setObjectName("btnDanger")
        btn_pulse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pulse.clicked.connect(self._pulse)
        gs.addWidget(btn_pulse, 3, 0, 1, 3)
        left.addWidget(gb_set)

        gb_run = QGroupBox("Модель")
        lr = QHBoxLayout(gb_run)
        self.btn_run = QPushButton("⏸ Пауза")
        self.btn_run.setObjectName("btnPrimary")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_reset = QPushButton("⟲ Сброс")
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self._reset_model)
        lr.addWidget(self.btn_run); lr.addWidget(self.btn_reset)
        left.addWidget(gb_run)

        gb_ind = QGroupBox("Показатели процесса")
        gi = QGridLayout(gb_ind)
        self.ind = {}
        for i, name in enumerate(["Температура y, °C", "Ошибка e, °C",
                                  "Клапан u, %", "IAE, °C·с",
                                  "Оценка K (APC)", "Добавка ИИ, %"]):
            gi.addWidget(QLabel(name), i, 0)
            lbl = QLabel("—")
            lbl.setObjectName("indVal")
            gi.addWidget(lbl, i, 1)
            self.ind[name] = lbl
        left.addWidget(gb_ind)
        left.addStretch(1)

        self.fig = Figure(figsize=(8, 6), facecolor=PLT_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        root.addWidget(self.canvas, 1)

        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_facecolor(PLT_AXBG)
        self.ax1.set_ylabel("Температура, °C", color=PLT_FG)
        self.ax1.set_title("Интеллектуальный контур управления: "
                           "процесс и регулятор в реальном времени",
                           color=PLT_FG, fontsize=11, fontweight="bold",
                           pad=10)
        (self.ln_sp,) = self.ax1.plot([], [], "--", color=PLT_SP, lw=1.3,
                                      label="Уставка")
        (self.ln_y,) = self.ax1.plot([], [], color=PLT_Y, lw=2.0,
                                     label="Температура y")
        self.ax1.legend(loc="upper right", fontsize=8)

        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.ax2.set_facecolor(PLT_AXBG)
        self.ax2.set_xlabel("Время, с", color=PLT_FG)
        self.ax2.set_ylabel("Клапан u, %", color=PLT_FG)
        (self.ln_u,) = self.ax2.plot([], [], color=PLT_U, lw=1.6,
                                     label="Клапан u")
        (self.ln_tr,) = self.ax2.plot([], [], color=PLT_TRIM, lw=1.3,
                                      label="Добавка ИИ")
        self.ax2.set_ylim(-40, 140)
        self.ax2.legend(loc="upper right", fontsize=8)

        for ax in (self.ax1, self.ax2):
            ax.tick_params(colors=PLT_FG, labelsize=9)
            for side in ax.spines.values():
                side.set_color(PLT_GRID)
            ax.grid(color=PLT_GRID, lw=0.6, alpha=0.9)
            leg = ax.get_legend()
            leg.get_frame().set_facecolor(PLT_AXBG)
            leg.get_frame().set_edgecolor(PLT_GRID)
            leg.get_frame().set_alpha(0.9)
            for txt in leg.get_texts():
                txt.set_color(PLT_FG)

        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.93,
                                 bottom=0.09, hspace=0.22)

    def _make_controller(self, mode):
        if mode == MODE_PID:
            return PID(Kp=2.0, Ki=0.12, Kd=4.0)
        if mode == MODE_APC:
            return AdaptivePID(Kp=2.0, Ki=0.12, Kd=4.0, K_nom=0.9)
        return PID(Kp=2.0, Ki=0.12, Kd=4.0)

    def _mode_changed(self, mode):
        self.controller = self._make_controller(mode)
        self.trim = 0.0

    def _reset_model(self):
        self.plant = Plant()
        self.t = 0.0
        self.iae = 0.0
        self.pulse = 0
        self.trim = 0.0
        for d in (self.h_t, self.h_y, self.h_sp, self.h_u, self.h_trim):
            d.clear()
        self.controller = self._make_controller(self.combo.currentIndex())
        self._plot_cnt = self._plot_skip - 1

    def _toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("▶ Пуск" if not self.running else "⏸ Пауза")
        self.btn_run.setObjectName("btnRun" if not self.running
                                  else "btnPrimary")
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)

    def _pulse(self):
        self.pulse = 25

    def _start_training(self):
        self.thread_train = TrainThread()
        self.thread_train.progress.connect(self._train_progress)
        self.thread_train.done.connect(self._train_done)
        self.thread_train.start()
        self.combo.model().item(MODE_RL).setEnabled(False)

    def _train_progress(self, p):
        self.bar_rl.setValue(p)
        self.lbl_rl.setText(f"ИИ: обучение… {p} %")

    def _train_done(self, agent):
        self.agent = agent
        self.bar_rl.setValue(100)
        self.lbl_rl.setText("ИИ: агент обучен ✔ (можно включать режим RL)")
        self.lbl_rl.setStyleSheet("color:#7ee787; font-weight:600;")
        self.combo.model().item(MODE_RL).setEnabled(True)

    def _tick(self):
        if not self.running:
            return
        sp = float(self.sl_sp.value())
        self.plant.K = self.sl_k.value() / 10.0
        self.plant.Tin = float(self.sl_tin.value())
        if self.pulse > 0:
            self.plant.Tin += 10.0
            self.pulse -= 1

        mode = self.combo.currentIndex()
        e = sp - self.plant.y
        de = 0.0 if not self.h_y else (self.plant.y - self.h_y[-1]) / DT

        u_pid = self.controller.update(sp, self.plant.y, DT)
        u = u_pid
        self.trim = 0.0
        if mode == MODE_RL and self.agent is not None:
            s = discretize(e, de)
            self.trim = ACTIONS[self.agent.act(s, greedy=True)]
            u = u_pid + self.trim
            if u < 0.0:
                u = 0.0
            elif u > 100.0:
                u = 100.0

        y = self.plant.step(u, DT)
        self.iae += abs(sp - y) * DT
        self.t += DT

        self.h_t.append(self.t)
        self.h_y.append(y)
        self.h_sp.append(sp)
        self.h_u.append(u)
        self.h_trim.append(self.trim)

        self._update_indicators(sp, y, e, u)
        self._plot_cnt += 1
        if self._plot_cnt >= self._plot_skip:
            self._plot_cnt = 0
            self._update_plots()

    def _update_plots(self):
        t = np.fromiter(self.h_t, float); y = np.fromiter(self.h_y, float)
        sp = np.fromiter(self.h_sp, float)
        u = np.fromiter(self.h_u, float)
        tr = np.fromiter(self.h_trim, float)
        self.ln_y.set_data(t, y)
        self.ln_sp.set_data(t, sp)
        self.ln_u.set_data(t, u)
        self.ln_tr.set_data(t, tr)
        self.ax1.set_xlim(max(0, self.t - WINDOW), max(WINDOW, self.t))
        if len(y):
            ymin = min(y.min(), sp.min()) - 5
            ymax = max(y.max(), sp.max()) + 5
        else:
            ymin, ymax = 0, 100
        self.ax1.set_ylim(ymin, ymax)

        self.canvas.draw_idle()

    def _update_indicators(self, sp, y, e, u):
        self.ind["Температура y, °C"].setText(f"{y:6.1f}")
        self.ind["Ошибка e, °C"].setText(f"{e:+6.1f}")
        self.ind["Клапан u, %"].setText(f"{u:6.1f}")
        self.ind["IAE, °C·с"].setText(f"{self.iae:8.0f}")
        kest = getattr(self.controller, "K_est", None)
        self.ind["Оценка K (APC)"].setText(
            f"{kest:4.2f}" if kest is not None else "—")
        self.ind["Добавка ИИ, %"].setText(f"{self.trim:+5.1f}")

        ae = abs(e)
        state = "ok" if ae <= 1.0 else ("warn" if ae <= 5.0 else "bad")
        if state != self._e_state:
            self._e_state = state
            self.ind["Ошибка e, °C"].setStyleSheet(
                {"ok":   "color:#7ee787; font-weight:700; font-size:15px;"
                         " font-family:'Consolas',monospace;",
                 "warn": "color:#f5c542; font-weight:700; font-size:15px;"
                         " font-family:'Consolas',monospace;",
                 "bad":  "color:#ff5c5c; font-weight:700; font-size:15px;"
                         " font-family:'Consolas',monospace;"}[state])

    def closeEvent(self, event):
        self.timer.stop()
        th = getattr(self, "thread_train", None)
        if th is not None and th.isRunning():
            th.stop()
            th.wait(3000)
        event.accept()

def selftest():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.combo.setCurrentIndex(MODE_APC)
    for _ in range(50):
        win._tick()
    win.combo.setCurrentIndex(MODE_APC)
    assert len(win.h_y) == 50, "история не заполняется"
    win.combo.setCurrentIndex(MODE_PID)
    for _ in range(20):
        win._tick()
    print(f"SELFTEST OK: y={win.h_y[-1]:.1f} °C, u={win.h_u[-1]:.1f} %, "
          f"IAE={win.iae:.0f}")
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
