import argparse
import yaml
import torch
import numpy as np

from pathlib import Path

from fuat_scripts.tahmin_fuat import (
    live_decision,
    fix_config_device
)

from turn_taking.analysis.validation.probabilities import (
    VAPDecoder,
)
from turn_taking.model.multimodal_model import (
    EarlyVAFusion,
)
from buse_workspace.live_robot.live_video_extractor import (
    LiveVideoExtractor,
)

from buse_workspace.live_robot.robot_state_machine import (
    RobotStateMachine,
)

from buse_workspace.live_robot.robot_tts import (
    RobotTTS,
)

from buse_workspace.live_robot.audio_mixer import (
    AudioMixer,
)

class LiveVAPRunner:

    def __init__(

        self,

        model_path,

        config_path,

        sample_rate=16000,

        window_size=4.0,

        step_size=0.5,

    ):

        #
        # Audio
        #

        self.sample_rate = sample_rate

        self.window_size = window_size

        self.step_size = step_size

        self.step_samples = int(

            sample_rate * step_size

        )

        self.window_samples = int(

            sample_rate * window_size

        )

        #
        # Device
        #

        if torch.backends.mps.is_available():

            self.device = torch.device("mps")

        else:

            self.device = torch.device("cpu")

        #
        # Model bilgileri
        #

        self.model_path = model_path

        self.config_path = config_path

        #
        # Decoder
        #

        self.decoder = VAPDecoder(

            bin_times=[

                0.2,

                0.4,

                0.6,

                0.8,

            ]

        )
        #
        # State Machine
        #

        self.state_machine = RobotStateMachine()

        #
        # Robot
        #

        self.robot = RobotTTS(

            sample_rate=sample_rate,

            block_size=self.step_samples,
            state_machine=self.state_machine
        )


        #
        # Audio Mixer
        #

        self.mixer = AudioMixer(

            robot_tts=self.robot,

            sample_rate=sample_rate,

            step_samples=self.step_samples,

        )
        
        self.video_extractor = LiveVideoExtractor(
            fps=16,
            target_len=600,
        )

        #
        # MM-VAP modeli
        #

        self.model = None

        #
        # Son 4 saniyelik stereo buffer
        #

        self.audio_buffer = np.zeros(

            (

                self.window_samples,

                2,

            ),

            dtype=np.float32,

        )

    # =====================================================

    def load_model(self):
        """
        Türkçe fine-tune edilmiş MM-VAP modelini yükle.
        """

        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        #
        # Config içinde kalan cuda referanslarını temizle.
        #

        fix_config_device(
            cfg,
            self.device,
        )

        model = EarlyVAFusion(
            cfg=cfg,
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.to(self.device)

        model.eval()

        self.model = model

        print(
            f"Model loaded on {self.device}"
        )
    
    # =====================================================

    def decode_predictions(
        self,
        vad_logits,
        vap_logits,
    ):
        """
        Model çıktılarını olasılıklara dönüştürür.

        Returns
        -------

        dict
        """

        vad = torch.sigmoid(
            vad_logits
        )

        vap = torch.softmax(
            vap_logits,
            dim=-1,
        )

        p_future = self.decoder.p_future(
            vap
        )

        p_bc = self.decoder.p_bc(
            vap
        )

        #
        # Son frame
        #

        vad_frame = vad[
            0,
            -1,
            :
        ].cpu()

        if p_future.ndim == 3:
            future_frame = p_future[0, -1, :].cpu()
        else:
            future_frame = p_future[-1, :].cpu()

        if p_bc.ndim == 3:
            bc_frame = p_bc[0, -1, :].cpu()
        else:
            bc_frame = p_bc[-1, :].cpu()

        return {

            "vad": vad_frame,

            "future": future_frame,

            "backchannel": bc_frame,

        }
    # =====================================================

    def infer(self):
        """
        Mikrofon + robot sesini al,
        4 saniyelik pencereyi güncelle,
        modeli çalıştır.

        Returns
        -------
        dict
        """

        #
        # Yeni stereo frame
        #

        try:

            new_frame = self.mixer.get_next_frame()

        except Exception as e:

            print(e)

            return None

        #
        # Ring buffer
        #

        self.audio_buffer = np.concatenate(

            (

                self.audio_buffer[self.step_samples:],

                new_frame,

            ),

            axis=0,

        )

        #
        # Torch tensor
        #

        audio = torch.from_numpy(

            self.audio_buffer

        ).unsqueeze(0)

        audio = audio.to(self.device)
        
        video = self.video_extractor.get_frames()
        video = video.to(self.device)


        batch = {

            "audio_chunk": audio,
            "frames":video,

        }

        with torch.inference_mode():

            vad_logits, vap_logits = self.model(

                batch

            )

        prediction = self.decode_predictions(

            vad_logits,

            vap_logits,

        )

        return prediction
    # =====================================================

    def execute_action(
        self,
        action,
    ):
        """
        RobotAction'ı uygula.
        """

        #
        # Konuşma
        #

        if action.speak:
            if self.robot.is_speaking:
                return

            text = "Tamam devam edebiliriz."
            self.robot.enqueue(text)

            return

        #
        # Backchannel
        #

        if action.backchannel:
            if self.robot.is_speaking:
                return

            self.robot.enqueue(

                "Hı hı."

            )

            return
    # =====================================================

    def run(self):

        #
        # Model
        #

        self.load_model()

        #
        # Robot
        #

        try:
            self.robot.start_audio_stream()

            self.robot.start_worker()

            self.mixer.start()

        except Exception:

            self.robot.shutdown()

            raise

        print("\n========== LIVE MM-VAP ==========")
        print("Ctrl+C ile çıkabilirsiniz.\n")

        try:

            while True:

                #
                # Model inference
                #

                prediction = self.infer()

                if prediction is None:
                    continue

                vad = prediction["vad"]

                future = prediction["future"]

                backchannel = prediction["backchannel"]

                #
                # MM-VAP kararı
                #
                print(
                    "DEBUG",
                    "vad=", vad.numpy(),
                    "future=", future.numpy(),
                    "backchannel=", backchannel.numpy(),
                )
                decision, confidence = live_decision(

                    vad,

                    future,

                    backchannel,

                )

                #
                # State machine
                #

                action = self.state_machine.update(

                    decision,

                    confidence,

                )

                #
                # Log
                #

                print(

                    f"{decision:12s}"

                    f" conf={confidence:.3f}"

                    f" state={action.state.name}"

                    f" speaking={self.robot.is_speaking}"

                )

                #
                # Robot
                #

                self.execute_action(

                    action,

                )

        except KeyboardInterrupt:

            print("\nStopping...\n")

        try:

            self.mixer.stop()
            self.video_extractor.release()

        finally:

            self.robot.shutdown()
                
        
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model",

        required=True,

        help="Fine-tuned model checkpoint",

    )

    parser.add_argument(

        "--config",

        required=True,

        help="Model yaml",

    )

    parser.add_argument(

        "--sample_rate",

        default=16000,

        type=int,

    )

    parser.add_argument(

        "--window",

        default=4.0,

        type=float,

    )

    parser.add_argument(

        "--step",

        default=0.5,

        type=float,

    )

    return parser.parse_args()


# =====================================================


def main():

    args = parse_args()

    runner = LiveVAPRunner(

        model_path=args.model,

        config_path=args.config,

        sample_rate=args.sample_rate,

        window_size=args.window,

        step_size=args.step,

    )

    runner.run()


# =====================================================


if __name__ == "__main__":

    main()