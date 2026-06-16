import os
import tempfile
import threading
import subprocess
from pathlib import Path
from queue import Queue
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


class RobotTTS:
    """
    Robot speech engine.

    Görevleri
    ----------
    1. Metni sese dönüştürmek
    2. Hoparlörden çalmak
    3. Aynı PCM'i audio_mixer'a vermek
    4. Konuşma durumunu yönetmek
    """

    def __init__(
        self,
        sample_rate=16000,
        block_size=8000,
        voice="Yelda",
        state_machine=None
    ):

        self.sample_rate = sample_rate

        self.block_size = block_size

        self.voice = voice
        self.state_machine = state_machine
        self.worker_thread = None

        #
        # ---------- AUDIO BUFFER ----------
        #

        self.audio_buffer = np.zeros(
            0,
            dtype=np.float32
        )

        #
        # Hoparlörün okuyacağı index
        #

        self.speaker_cursor = 0

        #
        # MM-VAP mixer'ın okuyacağı index
        #

        self.mixer_cursor = 0

        #
        # Buffer thread-safe olmalı.
        #

        self.buffer_lock = threading.Lock()

        #
        # Robot şu an konuşuyor mu?
        #

        self.is_speaking = False

        #
        # Stop istendi mi?
        #

        self.stop_requested = False

        #
        # OutputStream daha sonra oluşturulacak.
        #

        self.output_stream = None

        #
        # Kuyruk
        #

        self.queue = Queue()

        #
        # Konuşma bitince çağrılacak callback.
        #

        self.finished_callback = None

        #
        # Konuşma tamamlandı sinyali
        #

        self.speech_finished_event = threading.Event()

        self.speech_finished_event.set()

        #
        # Audio callback konuşma bittiğinde
        # bu event'i set eder.
        #

        self.finished_event = threading.Event()

        #
        # Başlangıçta set olmasın.
        #

        self.finished_event.clear()

        #
        # Callback'i izleyen thread.
        #

        self.monitor_thread = None

        #
        # Geçici dosyalar
        #

        self.temp_directory = Path(
            tempfile.gettempdir()
        ) / "vap_robot"

        self.temp_directory.mkdir(
            exist_ok=True
        )

        self.aiff_file = self.temp_directory / "robot.aiff"

        self.wav_file = self.temp_directory / "robot.wav"

    # =====================================================
    # AIFF / WAV yardımcı fonksiyonları
    # =====================================================

    def _cleanup_temp_files(self):
        """
        Eski geçici dosyaları sil.
        """

        for f in (self.aiff_file, self.wav_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass


    # =====================================================

    def _generate_wav(self, text: str):
        """
        macOS 'say' komutu ile konuşmayı üret.

        robot.aiff
              ↓
        afconvert
              ↓
        robot.wav
        """

        self._cleanup_temp_files()

        #
        # 1) AIFF üret
        #

        subprocess.run(
            [
                "say",
                "-v",
                self.voice,
                "-o",
                str(self.aiff_file),
                text,
            ],
            check=True,
        )

        #
        # 2) AIFF -> WAV
        #
        # LEI16:
        # 16 bit PCM
        #

        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16",
                str(self.aiff_file),
                str(self.wav_file),
            ],
            check=True,
        )

        if not self.wav_file.exists():
            raise RuntimeError("WAV dosyası oluşturulamadı.")


    # =====================================================

    def _load_pcm(self):
        """
        robot.wav dosyasını belleğe yükle.

        Sonuç:

            self.audio_buffer
        """

        pcm, sr = sf.read(
            self.wav_file,
            dtype="float32",
        )

        #
        # Stereo ise mono yap.
        #

        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)

        #
        # Sample rate uyuşmuyorsa
        # lineer interpolation ile dönüştür.
        #

        if sr != self.sample_rate:

            duration = len(pcm) / sr

            old_time = np.linspace(
                0.0,
                duration,
                len(pcm),
                endpoint=False,
            )

            new_time = np.linspace(
                0.0,
                duration,
                int(duration * self.sample_rate),
                endpoint=False,
            )

            pcm = np.interp(
                new_time,
                old_time,
                pcm,
            ).astype(np.float32)

        #
        # Thread-safe buffer güncelle.
        #

        with self.buffer_lock:

            self.audio_buffer = pcm

            #
            # Her yeni konuşmada
            # iki cursor da başa döner.
            #

            self.speaker_cursor = 0

            self.mixer_cursor = 0

        return len(pcm)


    # =====================================================

    def prepare_text(self, text: str):
        """
        Bir metni konuşmaya hazır hale getirir.

        Henüz çalmaz.

        Sadece

            text
                ↓
            robot.wav
                ↓
            PCM buffer

        işlemini yapar.

        Döndürür:

            PCM örnek sayısı
        """

        self._generate_wav(text)

        sample_count = self._load_pcm()

        return sample_count
    
    # =====================================================
    # OutputStream callback
    # =====================================================

    def _audio_callback(
        self,
        outdata,
        frames,
        time_info,
        status,
    ):
        """
        sounddevice callback.

        Hoparlöre gönderilecek sesi
        audio_buffer'dan okur.
        """

        if status:
            print(status)

        with self.buffer_lock:

            #
            # Robot konuşmuyorsa
            #

            if not self.is_speaking:

                outdata.fill(0)

                return

            start = self.speaker_cursor

            end = start + frames

            #
            # Buffer sonuna geldik mi?
            #

            if start >= len(self.audio_buffer):

                outdata.fill(0)

                self.finished_event.set()

                return

            chunk = self.audio_buffer[start:end]

            self.speaker_cursor = end

        #
        # Eksik örnek varsa
        #

        if len(chunk) < frames:

            padding = np.zeros(
                frames - len(chunk),
                dtype=np.float32,
            )

            chunk = np.concatenate(
                (
                    chunk,
                    padding,
                )
            )
        if end >= len(self.audio_buffer):
            self.finished_event.set()

        outdata[:, 0] = chunk
    
    # =====================================================

    def _monitor_finished_playback(self):
        """
        Audio callback tarafından tetiklenen
        finished_event'i bekler.
        """

        while True:

            #
            # Callback event set edene kadar
            # CPU kullanmadan bekle.
            #

            self.finished_event.wait()

            #
            # Sonraki konuşma için event'i temizle.
            #

            self.finished_event.clear()

            #
            # Konuşma tamamlandı.
            #

            self._on_speech_finished()

    # =====================================================

    def start_audio_stream(self):
        """
        Hoparlör streamini başlat.

        Program boyunca yalnızca
        bir kez çağrılır.
        """

        if self.output_stream is not None:

            return

        self.output_stream = sd.OutputStream(

            samplerate=self.sample_rate,

            channels=1,

            dtype="float32",

            callback=self._audio_callback,

        )

        self.output_stream.start()

        #
        # Playback monitor thread
        #

        if self.monitor_thread is None:

            self.monitor_thread = threading.Thread(
                target=self._monitor_finished_playback,
                daemon=True,
            )

            self.monitor_thread.start()
    # =====================================================

    def enqueue(self, text):

        """
        Yeni konuşmayı kuyruğa ekle.
        """

        self.queue.put(text)
    # =====================================================

    def start_worker(self):
        """
        Kuyruğu dinleyen thread.
        """

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )

        self.worker_thread.start()
        # =====================================================

    def _worker_loop(self):

        while True:

            text = self.queue.get()

            if text is None:

                break

            #
            # WAV üret
            #

            self.prepare_text(text)

            self.speech_finished_event.clear()

            self.is_speaking = True

            if self.state_machine is not None:
                self.state_machine.robot_started_speaking()

            self.speech_finished_event.wait()

            self.queue.task_done()
    
    # =====================================================

    def _on_speech_finished(self):
        """
        Hoparlör son örneği de çaldığında çağrılır.
        """

        self.is_speaking = False

        if self.state_machine is not None:
            self.state_machine.robot_finished_speaking()

        self.speech_finished_event.set()

        if self.finished_callback is not None:
            self.finished_callback()

    # =====================================================

    def read_chunk(self, frames):
        """
        Robot sesinin sonraki kısmını döndür.

        audio_mixer.py bunu kullanacak.
        """

        with self.buffer_lock:

            start = self.mixer_cursor

            end = start + frames

            if start >= len(self.audio_buffer):

                return np.zeros(
                    frames,
                    dtype=np.float32,
                )

            chunk = self.audio_buffer[start:end]

            self.mixer_cursor = min(
                end,
                len(self.audio_buffer),
            )

        if len(chunk) < frames:

            padding = np.zeros(
                frames-len(chunk),
                dtype=np.float32,
            )

            chunk = np.concatenate(
                (
                    chunk,
                    padding,
                )
            )

        return chunk
    # =====================================================

    def stop(self):
        """
        Robot konuşmasını hemen durdur.
        """

        with self.buffer_lock:

            self.speaker_cursor = len(self.audio_buffer)

            self.mixer_cursor = len(self.audio_buffer)
        
        self.is_speaking = False

        self.finished_event.set()
    
    # =====================================================

    def shutdown(self):
        """
        Program kapanırken çağrılır.
        """

        self.stop()

        self.queue.put(None)

        if self.worker_thread is not None:
            self.worker_thread.join()
        #
        # Monitor thread bekliyorsa uyandır.
        #

        self.finished_event.set()

        if self.output_stream is not None:

            self.output_stream.stop()

            self.output_stream.close()

            self.output_stream = None

if __name__ == "__main__":

    tts = RobotTTS()

    tts.start_audio_stream()

    tts.start_worker()

    tts.enqueue(
        "Merhaba. Ben Türkçe MM VAP robotuyum."
    )

    tts.enqueue(
        "Seni dinliyorum."
    )

    tts.enqueue(
        "Devam edebilirsin."
    )

    try:

        tts.queue.join()

    except KeyboardInterrupt:

        pass

    finally:

        tts.shutdown()