from enum import Enum, auto
from dataclasses import dataclass
import time
import threading


# ============================================================
# Robot States
# ============================================================

class RobotState(Enum):
    LISTEN = auto()
    THINK = auto()
    SPEAK = auto()
    BACKCHANNEL = auto()
    WAIT = auto()


# ============================================================
# Action returned to upper layer
# ============================================================

@dataclass
class RobotAction:

    state: RobotState

    speak: bool = False

    backchannel: bool = False

    interrupt_human: bool = False

    confidence: float = 0.0

    text: str | None = None


# ============================================================
# State Machine
# ============================================================

class RobotStateMachine:

    def __init__(self):

        self.state = RobotState.LISTEN

        self.last_backchannel = 0.0

        self.last_speech = 0.0

        self.last_decision = None

        # --------------------------
        # Parameters
        # --------------------------

        self.shift_threshold = 0.50

        self.backchannel_threshold = 0.60

        self.cooldown_backchannel = 2.5

        self.cooldown_speech = 3.0

        self.debounce_frames = 1

        self._same_counter = 0

        # Robot konuşurken veya konuşmaya hazırlanırken
        # yeni shift/backchannel kararlarını yok say.
        self.speech_locked = False

        self._lock = threading.Lock()

    # -------------------------------------------------------

    def _accept_decision(self, decision):

        if decision == self.last_decision:

            self._same_counter += 1

        else:

            self._same_counter = 1

            self.last_decision = decision

        return self._same_counter >= self.debounce_frames

    # -------------------------------------------------------
    def update(self,
               decision: str,
               confidence: float):

        """
        decision :

            shift

            hold

            backchannel
        """
        # ----------------------------------------------------
        # Robot şu anda konuşuyor veya konuşmaya hazırlanıyor.
        # MM-VAP'ten gelen yeni kararları görmezden gel.
        # ----------------------------------------------------
        with self._lock:
            if self.speech_locked:
                return RobotAction(
                    state=self.state,
                    confidence=confidence
                )

            now = time.time()

            accepted = self._accept_decision(decision)

            if not accepted:

                return RobotAction(
                    state=self.state,
                    confidence=confidence
                )

            # ====================================================
            # HOLD
            # ====================================================

            if decision == "hold":

                self.state = RobotState.LISTEN

                return RobotAction(
                    state=self.state,
                    confidence=confidence
                )

            # ====================================================
            # BACKCHANNEL
            # ====================================================

            if decision == "backchannel":

                if confidence < self.backchannel_threshold:

                    return RobotAction(
                        state=self.state,
                        confidence=confidence
                    )

                if now - self.last_backchannel < self.cooldown_backchannel:

                    return RobotAction(
                        state=self.state,
                        confidence=confidence
                    )

                self.last_backchannel = now

                self.state = RobotState.BACKCHANNEL

                return RobotAction(

                    state=self.state,

                    backchannel=True,

                    confidence=confidence
                )

            # ====================================================
            # SHIFT
            # ====================================================

            if decision == "shift":

                if confidence < self.shift_threshold:

                    return RobotAction(
                        state=self.state,
                        confidence=confidence
                    )

                if now - self.last_speech < self.cooldown_speech:

                    return RobotAction(
                        state=self.state,
                        confidence=confidence
                    )

                self.last_speech = now

                self.state = RobotState.THINK

                return RobotAction(

                    state=self.state,

                    speak=True,

                    confidence=confidence
                )

            return RobotAction(
                state=self.state,
                confidence=confidence
            )

    # -------------------------------------------------------

    def robot_started_speaking(self):
        with self._lock:

            self.state = RobotState.SPEAK

            self.speech_locked = True

    # -------------------------------------------------------

    def robot_finished_speaking(self):
        with self._lock:

            self.state = RobotState.LISTEN

            self.speech_locked = False

    # -------------------------------------------------------

    def reset(self):
        with self._lock:

            self.state = RobotState.LISTEN

            self.last_backchannel = 0.0

            self.last_speech = 0.0

            self.last_decision = None

            self._same_counter = 0

            self.speech_locked = False