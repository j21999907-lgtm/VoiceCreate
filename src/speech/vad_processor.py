"""Voice activity detection for VoiceCreate.

Uses WebRTC VAD when available and falls back to an improved energy detector
with background-noise learning plus human-voice frequency checks.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("VoiceCreate")


import traceback
from typing import Dict, List, Optional

import numpy as np


try:
    import webrtcvad

    WEBRTCVAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    WEBRTCVAD_AVAILABLE = False
    logger.info("[VAD] webrtcvad not available; using improved energy detection")


class SpeechActivityDetector:
    """Detect speech in 16-bit mono PCM audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        if WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad(aggressiveness)
            self.detection_method = "webrtcvad"
        else:
            self.vad = None
            self.detection_method = "energy_based"

        # Lower defaults improve detection of short, quiet commands.
        self.energy_threshold = 120.0
        self.silence_threshold = 35.0
        self.voice_frequency_range = (85.0, 255.0)
        self.noise_floor: Optional[float] = None
        self.noise_history: List[float] = []
        self.noise_history_limit = 12

        logger.info(f"[VAD] initialized: method={self.detection_method}, sample_rate={sample_rate}")

    def detect_speech_activity(self, audio_data: bytes) -> bool:
        if not audio_data or len(audio_data) < self.frame_size * 2:
            logger.info(f"[VAD] audio too short: {len(audio_data) if audio_data else 0} bytes")
            return False

        try:
            if self.detection_method == "webrtcvad":
                return self._detect_with_webrtcvad(audio_data)
            return self._detect_with_energy(audio_data)
        except Exception as exc:
            logger.info(f"[VAD] detection failed: {exc}")
            traceback.print_exc()
            return False

    def set_energy_threshold(self, threshold: float, silence_threshold: Optional[float] = None) -> None:
        self.energy_threshold = max(0.0, float(threshold))
        if silence_threshold is not None:
            self.silence_threshold = max(0.0, float(silence_threshold))

    def learn_background_noise(self, audio_data: bytes) -> float:
        """Learn a noise floor from a quiet/background sample."""
        audio_array = self._audio_array(audio_data)
        if audio_array.size == 0:
            return self.noise_floor or 0.0

        energy = self._calculate_energy(audio_array)
        self.noise_history.append(energy)
        if len(self.noise_history) > self.noise_history_limit:
            self.noise_history.pop(0)
        self.noise_floor = float(np.median(self.noise_history))
        logger.debug(f"[VAD] learned noise_floor={self.noise_floor:.1f}")
        return self.noise_floor

    def is_silent(self, audio_data: bytes) -> bool:
        try:
            audio_array = self._audio_array(audio_data)
            if audio_array.size == 0:
                return True
            energy = self._calculate_energy(audio_array)
            threshold = max(self.silence_threshold, (self.noise_floor or 0.0) * 1.2)
            return energy < threshold
        except Exception:
            traceback.print_exc()
            return True

    def _audio_array(self, audio_data: bytes) -> np.ndarray:
        try:
            return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    def _calculate_energy(self, audio_array: np.ndarray) -> float:
        if audio_array.size == 0:
            return 0.0
        return float(np.mean(np.square(audio_array)))

    def _adaptive_energy_threshold(self) -> float:
        if self.noise_floor is None:
            return self.energy_threshold
        return max(self.energy_threshold, self.noise_floor * 2.2)

    def _dominant_frequency(self, audio_array: np.ndarray) -> float:
        if audio_array.size < max(32, self.sample_rate // 50):
            return 0.0

        samples = audio_array - float(np.mean(audio_array))
        if float(np.max(np.abs(samples))) < 1.0:
            return 0.0

        window = np.hanning(samples.size)
        spectrum = np.fft.rfft(samples * window)
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)
        magnitudes = np.abs(spectrum)

        mask = (frequencies >= 50.0) & (frequencies <= 1000.0)
        if not np.any(mask):
            return 0.0

        masked_magnitudes = magnitudes[mask]
        masked_frequencies = frequencies[mask]
        if masked_magnitudes.size == 0 or float(np.max(masked_magnitudes)) <= 0:
            return 0.0
        return float(masked_frequencies[int(np.argmax(masked_magnitudes))])

    def _has_voice_frequency(self, audio_array: np.ndarray) -> bool:
        dominant = self._dominant_frequency(audio_array)
        low, high = self.voice_frequency_range
        return low <= dominant <= high

    def _detect_with_webrtcvad(self, audio_data: bytes) -> bool:
        frames = self._split_audio_into_frames(audio_data)
        if not frames:
            return False

        speech_frames = 0
        for frame in frames:
            try:
                if self.vad and self.vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
            except Exception:
                continue

        total_frames = len(frames)
        speech_ratio = speech_frames / total_frames if total_frames else 0.0
        energy_has_speech = self._detect_with_energy(audio_data)
        has_speech = speech_ratio > 0.30 or (speech_ratio > 0.12 and energy_has_speech)

        logger.debug(f"[VAD] webrtc frames={speech_frames}/{total_frames} "
            f"ratio={speech_ratio:.1%}, energy_path={energy_has_speech}, speech={has_speech}")
        return has_speech

    def _detect_with_energy(self, audio_data: bytes) -> bool:
        audio_array = self._audio_array(audio_data)
        if audio_array.size == 0:
            return False

        energy = self._calculate_energy(audio_array)
        adaptive_threshold = self._adaptive_energy_threshold()
        dominant = self._dominant_frequency(audio_array)
        has_voice_frequency = self.voice_frequency_range[0] <= dominant <= self.voice_frequency_range[1]
        is_silent = self.is_silent(audio_data)
        energy_match = energy > adaptive_threshold

        # Very loud utterances may show dominant harmonics outside the basic
        # voice band, so keep a controlled escape hatch for clear speech.
        has_speech = energy_match and (has_voice_frequency or energy > adaptive_threshold * 3.0)
        db = 10 * np.log10(energy) if energy > 0 else -100.0

        logger.debug(f"[VAD] energy={energy:.0f}, threshold={adaptive_threshold:.0f}, "
            f"dB={db:.1f}, dominant={dominant:.1f}Hz, "
            f"voice_freq={has_voice_frequency}, silent={is_silent}, speech={has_speech}")
        return bool(has_speech and not is_silent)

    def _split_audio_into_frames(self, audio_data: bytes) -> List[bytes]:
        frames: List[bytes] = []
        frame_size_bytes = self.frame_size * 2
        for i in range(0, len(audio_data), frame_size_bytes):
            frame = audio_data[i:i + frame_size_bytes]
            if len(frame) == frame_size_bytes:
                frames.append(frame)
            elif len(frame) >= frame_size_bytes // 2:
                frames.append(frame + b"\x00" * (frame_size_bytes - len(frame)))
        return frames

    def analyze_audio_quality(self, audio_data: bytes) -> Dict[str, object]:
        try:
            audio_array = self._audio_array(audio_data)
            if audio_array.size == 0:
                return {"error": "empty audio"}

            energy = self._calculate_energy(audio_array)
            dominant = self._dominant_frequency(audio_array)
            zero_crossings = np.sum(np.diff(np.signbit(audio_array))) / audio_array.size
            return {
                "length_samples": int(audio_array.size),
                "length_seconds": float(audio_array.size / self.sample_rate),
                "mean": float(np.mean(audio_array)),
                "std": float(np.std(audio_array)),
                "max_amplitude": int(np.max(np.abs(audio_array))),
                "energy": float(energy),
                "dominant_frequency": float(dominant),
                "voice_frequency_match": bool(self.voice_frequency_range[0] <= dominant <= self.voice_frequency_range[1]),
                "noise_floor": self.noise_floor,
                "adaptive_energy_threshold": self._adaptive_energy_threshold(),
                "zero_crossing_rate": float(zero_crossings),
                "is_silent": self.is_silent(audio_data),
                "has_audio": energy > self._adaptive_energy_threshold() * 0.1,
            }
        except Exception as exc:
            logger.info(f"[VAD] audio quality analysis failed: {exc}")
            return {"error": str(exc)}


_global_vad_detector: Optional[SpeechActivityDetector] = None


def get_vad_detector(sample_rate: int = 16000) -> SpeechActivityDetector:
    global _global_vad_detector
    if _global_vad_detector is None or _global_vad_detector.sample_rate != sample_rate:
        _global_vad_detector = SpeechActivityDetector(sample_rate=sample_rate)
    return _global_vad_detector


def detect_speech_activity(
    audio_data: bytes,
    sample_rate: int = 16000,
    energy_threshold: Optional[float] = None,
) -> bool:
    detector = get_vad_detector(sample_rate)
    if energy_threshold is not None:
        detector.set_energy_threshold(energy_threshold)
    return detector.detect_speech_activity(audio_data)


def learn_background_noise(audio_data: bytes, sample_rate: int = 16000) -> float:
    return get_vad_detector(sample_rate).learn_background_noise(audio_data)


def is_silent(audio_data: bytes, sample_rate: int = 16000) -> bool:
    return get_vad_detector(sample_rate).is_silent(audio_data)


def test_vad_detection() -> bool:
    sample_rate = 16000
    vad = SpeechActivityDetector(sample_rate=sample_rate, aggressiveness=2)
    duration = 1.0
    samples = int(sample_rate * duration)
    silence_audio = np.zeros(samples, dtype=np.int16).tobytes()

    rng = np.random.default_rng(seed=7)
    noise_audio = rng.integers(-45, 45, samples, dtype=np.int16).tobytes()
    vad.learn_background_noise(noise_audio)

    t = np.linspace(0, duration, samples, endpoint=False)
    speech_wave = 0.25 * np.sin(2 * np.pi * 180 * t) * 32767
    speech_audio = speech_wave.astype(np.int16).tobytes()

    silence_result = vad.detect_speech_activity(silence_audio)
    speech_result = vad.detect_speech_activity(speech_audio)
    logger.info(f"[VAD TEST] silence={silence_result}, speech={speech_result}")
    return (not silence_result) and speech_result


if __name__ == "__main__":
    ok = test_vad_detection()
    logger.info("vad tests passed" if ok else "vad tests failed")
