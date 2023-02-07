from . import Optimizer
import numpy as np

EPSILON = 1e-8


class Adam(Optimizer):
    m_dW: np.ndarray
    v_dW: np.ndarray
    m_db: np.ndarray
    v_db: np.ndarray

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t: int = 1

    def step(
        self,
        W: np.ndarray,
        b: np.ndarray,
        dW: np.ndarray,
        db: np.ndarray,
        reg_W: np.ndarray,
    ) -> None:
        if self.t == 1:
            self._setup(W, b)

        self.m_dW = self._weighted_average(self.m_dW, dW, self.beta1)
        self.m_db = self._weighted_average(self.m_db, db, self.beta1)
        self.v_dW = self._weighted_average(self.v_dW, np.square(dW), self.beta2)
        self.v_db = self._weighted_average(self.v_db, np.square(db), self.beta2)

        m_dW_corr = self.m_dW / (1 - self.beta1**self.t)
        m_db_corr = self.m_db / (1 - self.beta1**self.t)
        v_dW_corr = self.v_dW / (1 - self.beta2**self.t)
        v_db_corr = self.v_db / (1 - self.beta2**self.t)

        W += -self.learning_rate * m_dW_corr / (np.sqrt(v_dW_corr) + EPSILON) - reg_W
        b += -self.learning_rate * m_db_corr / (np.sqrt(v_db_corr) + EPSILON)

        self.t += 1

    def _setup(self, W: np.ndarray, b: np.ndarray):
        self.m_dW = np.zeros(W.shape)
        self.v_dW = np.zeros(W.shape)
        self.m_db = np.zeros(b.shape)
        self.v_db = np.zeros(b.shape)

    @staticmethod
    def _weighted_average(A: np.ndarray, B: np.ndarray, beta: float) -> np.ndarray:
        return A * beta + B * (1 - beta)
