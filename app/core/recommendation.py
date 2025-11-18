from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict

import requests

from .config import AppConfig


@dataclass
class RecommendationResult:
    prompt: str
    response: str


class GroqRecommender:
    """Wrapper around Groq API for water quality recommendations."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = api_key or os.getenv(AppConfig.GROQ_API_KEY_ENV)
        self.model_name = model_name or AppConfig.GROQ_MODEL_NAME
        if not self.api_key:
            raise RuntimeError(
                f"Set environment variable {AppConfig.GROQ_API_KEY_ENV} to enable recommendations."
            )

    @staticmethod
    def _build_prompt(payload: Dict[str, float], predicted_label: str) -> str:
        template = (
            "Anda adalah pakar kualitas air. Fokuskan analisis hanya pada parameter berikut:\n"
            f"- turbidity: {payload['turbidity']}\n"
            f"- tds: {payload['tds']}\n"
            f"- ph: {payload['ph']}\n\n"
            f"Model memprediksi kualitas air: {predicted_label}.\n"
            "Balas dalam Bahasa Indonesia dengan format PERSIS dua baris berikut (tanpa kata tambahan):\n"
            "Interpretasi: <jelaskan kondisi air, sebutkan angka turbidity (bandingkan dengan batas Permenkes 2023 yaitu 3 NTU), TDS (bandingkan dengan batas 300 mg/L), pH (ideal 6.5–8.5 menurut Permenkes 2023), dan jelaskan dampak bila nilai menyimpang; minimal 25 kata>\n"
            "Rekomendasi: <usulkan langkah treatment praktis dan spesifik yang sesuai dengan kondisi di atas. Gunakan metode yang berbeda jika perlu, misalnya filter, RO, resin, injeksi kimia, UV, dsb., dan jelaskan alasan teknisnya terhadap parameter bermasalah; minimal 20 kata>\n"
            "Setiap baris wajib diawali kata kunci di atas, tidak boleh ada baris lain, dan jangan menyebut parameter aliran. Hindari mengulang persis contoh yang diberikan."
        )
        return template.format(**payload, label=predicted_label)

    def generate(self, payload: Dict[str, float], predicted_label: str) -> RecommendationResult:
        prompt = self._build_prompt(payload, predicted_label)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert water quality consultant. Respond in Bahasa Indonesia.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        last_error: RuntimeError | None = None
        for attempt in range(3):
            try:
                response = requests.post(self.API_URL, json=body, headers=headers, timeout=60)
            except requests.RequestException as exc:
                last_error = RuntimeError(f"Tidak dapat menghubungi Groq API: {exc}")  # type: ignore[arg-type]
                time.sleep(1.5)
                continue

            if response.status_code >= 500:
                last_error = RuntimeError(
                    "Groq API sedang mengalami gangguan (kode 5xx). Silakan coba lagi beberapa saat lagi."
                )
                time.sleep(1.5)
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Groq API mengembalikan error {response.status_code}: {response.text}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(f"Groq API mengembalikan respons tidak valid: {exc}") from exc

            choices = data.get("choices", [])
            if not choices:
                text = "Tidak ada respon dari Groq."
            else:
                text = choices[0].get("message", {}).get("content") or "Tidak ada respon dari Groq."

            cleaned = text.strip()
            if cleaned.startswith("<!DOCTYPE"):
                last_error = RuntimeError(
                    "Groq API mengembalikan HTML (kemungkinan error 5xx). Coba ulang lagi nanti."
                )
                time.sleep(1.5)
                continue

            return RecommendationResult(prompt=prompt, response=cleaned)

        if last_error is not None:
            raise last_error

        raise RuntimeError("Groq API tidak memberikan respon.")


__all__ = ["GroqRecommender", "RecommendationResult"]


