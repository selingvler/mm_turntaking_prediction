import threading

import numpy as np
import sounddevice as sd
import time


class AudioMixer:
    """
    Human microphone + RobotTTS PCM

                ↓

         Stereo audio for MM-VAP

    Left  channel : Human
    Right channel : Robot
    """

    def __init__(
        self,
        robot_tts,
        sample_rate=16000,
        step_samples=8000,
    ):

        self.robot_tts = robot_tts

        self.sample_rate = sample_rate

        self.step_samples = step_samples

        #
        # Input stream
        #

        self.stream = None

        #
        # Stream durumu
        #

        self.running = False
        self.overflow_counter = 0

        #
        # Mikrofon erişimi thread-safe olsun.
        #

        self.stream_lock = threading.Lock()

    # =====================================================

    def start(self):
        """
        Mikrofonu aç.

        Program boyunca yalnızca
        bir kez çağrılması yeterlidir.
        """

        with self.stream_lock:

            if self.running:

                return

            self.stream = sd.InputStream(

                samplerate=self.sample_rate,

                channels=1,

                dtype="float32",

                blocksize=self.step_samples,

            )

            try:

                self.stream.start()

            except Exception as e:

                raise RuntimeError(
                    f"Mikrofon başlatılamadı: {e}"
                )

            self.running = True

    # =====================================================

    def stop(self):
        """
        Mikrofonu kapat.
        """

        with self.stream_lock:

            if not self.running:

                return

            self.stream.stop()

            self.stream.close()

            self.stream = None

            self.running = False
            # RobotTTS ayrı olarak shutdown() edilmelidir.

    # =====================================================

    def get_human_chunk(self):
        """
        Mikrofondan step_samples kadar ses oku.

        Döndürür
        --------
        np.ndarray
            shape = (step_samples,)
            dtype = float32
        """

        if not self.running:
            raise RuntimeError(
                "AudioMixer.start() çağrılmadan mikrofon okunamaz."
            )
        
        with self.stream_lock:

            data, overflowed = self.stream.read(self.step_samples)

            if overflowed:
                self.overflow_counter += 1

            #
            # (N,1) -> (N,)
            #

            human = data[:, 0]

        #
        # Emin olalım
        #

        return human.astype(np.float32)
    
    # =====================================================

    def get_robot_chunk(self):
        """
        RobotTTS'den aynı uzunlukta ses al.

        Döndürür
        --------
        np.ndarray
            shape = (step_samples,)
            dtype = float32
        """

        if self.robot_tts is None:

            return np.zeros(
                self.step_samples,
                dtype=np.float32,
            )

        try:

            chunk = self.robot_tts.read_chunk(
                self.step_samples
            )

        except Exception as e:

            print(f"RobotTTS error: {e}")

            chunk = np.zeros(
                self.step_samples,
                dtype=np.float32,
            )

        if chunk.dtype != np.float32:

            chunk = chunk.astype(
                np.float32
            )

        return chunk
    
    # =====================================================

    def mix(
        self,
        human_chunk,
        robot_chunk,
    ):
        """
        Human -> Left
        Robot -> Right

        Returns
        -------
        np.ndarray

            shape = (N,2)

        """

        if human_chunk.ndim != 1:

            raise ValueError(
                "Human chunk tek boyutlu olmalıdır."
            )

        if robot_chunk.ndim != 1:

            raise ValueError(
                "Robot chunk tek boyutlu olmalıdır."
            )

        if human_chunk.shape != robot_chunk.shape:

            raise ValueError(
                "Human ve robot chunk uzunlukları aynı olmalıdır."
            )

        human_chunk = human_chunk.astype(
            np.float32,
            copy=False,
        )

        robot_chunk = robot_chunk.astype(
            np.float32,
            copy=False,
        )

        stereo = np.stack(

            (

                human_chunk,

                robot_chunk,

            ),

            axis=1,

        )

        return np.ascontiguousarray(
            stereo,
            dtype=np.float32,
        )
    
    # =====================================================

    def get_next_frame(self):
        """
        Mikrofonu oku

              +

        Robot sesini oku

              +

        Stereo frame oluştur.

        Returns
        -------

        np.ndarray

            shape =

                (step_samples,2)

        """

        human_chunk = self.get_human_chunk()

        robot_chunk = self.get_robot_chunk()

        stereo = self.mix(

            human_chunk,

            robot_chunk,

        )

        return stereo
    
if __name__ == "__main__":

    from robot_tts import RobotTTS

    tts = RobotTTS()

    tts.start_audio_stream()

    tts.start_worker()

    mixer = AudioMixer(

        robot_tts=tts,

        sample_rate=16000,

        step_samples=8000,

    )

    mixer.start()

    tts.enqueue(

        "Merhaba ben MM VAP robotuyum."

    )

    try:

        while True:

            frame = mixer.get_next_frame()

            print(

                frame.shape,

                frame.dtype,

            )
            time.sleep(0.01)

    except KeyboardInterrupt:

        pass

    finally:

        mixer.stop()

        tts.shutdown()