import os
import io
import speech_recognition as sr
import pyttsx3
from typing import Optional
import tempfile
import wave

try:
	import azure.cognitiveservices.speech as speechsdk
	has_speechsdk = True
except Exception:
	has_speechsdk = False
	speechsdk = None  # type: ignore

class VoiceService:
	def __init__(self):
		self.recognizer = sr.Recognizer()
		self.microphone = sr.Microphone()
		# Azure Speech config (optional but preferred for Vietnamese TTS)
		self.azure_speech_key = os.getenv("AZURE_SPEECH_KEY", "")
		self.azure_speech_region = os.getenv("AZURE_SPEECH_REGION", "")
		self.use_azure_speech = has_speechsdk and bool(self.azure_speech_key and self.azure_speech_region)
		self.default_vi_voice = os.getenv("AZURE_SPEECH_VOICE", "vi-VN-HoaiMyNeural")
		
		self.tts_engine = None if self.use_azure_speech else pyttsx3.init()
		if not self.use_azure_speech:
			self.setup_tts()
	
	def setup_tts(self):
		"""Setup text-to-speech engine (pyttsx3 fallback)"""
		try:
			voices = self.tts_engine.getProperty('voices') if self.tts_engine else []
			# Try to find Vietnamese voice (rare on Windows without additional packs)
			for voice in voices:
				if 'vietnamese' in getattr(voice, 'name', '').lower() or 'vi' in getattr(voice, 'id', '').lower():
					self.tts_engine.setProperty('voice', voice.id)
					break
			# Set speech rate and volume
			if self.tts_engine:
				self.tts_engine.setProperty('rate', 150)
				self.tts_engine.setProperty('volume', 0.9)
		except Exception as e:
			print(f"Error setting up TTS: {e}")
	
	def speech_to_text(self, audio_data: bytes = None, language: str = "vi-VN") -> Optional[str]:
		"""Convert speech to text"""
		try:
			if audio_data:
				# Use provided audio data
				audio_source = sr.AudioData(audio_data, 16000, 2)
			else:
				# Use microphone
				with self.microphone as source:
					print("Đang nghe...")
					self.recognizer.adjust_for_ambient_noise(source)
					audio = self.recognizer.listen(source, timeout=5)
			# Recognize speech
			text = self.recognizer.recognize_google(audio, language=language)
			return text
		except sr.WaitTimeoutError:
			print("Không nghe thấy gì, vui lòng thử lại")
			return None
		except sr.UnknownValueError:
			print("Không thể nhận diện giọng nói")
			return None
		except sr.RequestError as e:
			print(f"Lỗi kết nối dịch vụ nhận diện giọng nói: {e}")
			return None
		except Exception as e:
			print(f"Lỗi chuyển đổi giọng nói thành văn bản: {e}")
			return None
	
	def _azure_tts_to_file(self, text: str, filename: str, language: str = "vi") -> bool:
		try:
			if not self.use_azure_speech:
				return False
			speech_config = speechsdk.SpeechConfig(subscription=self.azure_speech_key, region=self.azure_speech_region)
			# Select Vietnamese neural voice
			voice = self.default_vi_voice if language.startswith('vi') else os.getenv("AZURE_SPEECH_VOICE_FALLBACK", "en-US-AriaNeural")
			speech_config.speech_synthesis_voice_name = voice
			audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
			synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
			result = synthesizer.speak_text_async(text).get()
			return result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
		except Exception as e:
			print(f"Azure TTS error: {e}")
			return False
	
	def text_to_speech(self, text: str, language: str = "vi") -> bytes:
		"""Convert text to speech and return audio data"""
		try:
			# Create temporary file for audio output
			with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
				temp_path = temp_file.name
			
			ok = False
			if self.use_azure_speech:
				ok = self._azure_tts_to_file(text, temp_path, language)
			else:
				# pyttsx3 fallback
				self.tts_engine.save_to_file(text, temp_path)
				self.tts_engine.runAndWait()
				ok = True
			
			if not ok:
				# Cleanup and return empty on failure
				try:
					os.unlink(temp_path)
				except Exception:
					pass
				return b""
			
			# Read audio data
			with open(temp_path, 'rb') as f:
				audio_data = f.read()
			# Clean up
			os.unlink(temp_path)
			return audio_data
		except Exception as e:
			print(f"Lỗi chuyển đổi văn bản thành giọng nói: {e}")
			return b""
	
	def speak(self, text: str):
		"""Speak text directly (pyttsx3 only)"""
		try:
			if self.use_azure_speech:
				# For direct speaking to default speaker, create output without filename
				speech_config = speechsdk.SpeechConfig(subscription=self.azure_speech_key, region=self.azure_speech_region)
				speech_config.speech_synthesis_voice_name = self.default_vi_voice
				synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
				synthesizer.speak_text_async(text).get()
				return
			self.tts_engine.say(text)
			self.tts_engine.runAndWait()
		except Exception as e:
			print(f"Lỗi phát âm: {e}")
	
	def is_voice_enabled(self) -> bool:
		"""Check if voice features are available"""
		try:
			# Test microphone
			with self.microphone as source:
				self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
			# Test TTS
			if self.use_azure_speech:
				return True
			self.tts_engine.getProperty('voices')
			return True
		except Exception as e:
			print(f"Voice features not available: {e}")
			return False
	
	def get_available_languages(self) -> list:
		"""Get list of available languages for speech recognition"""
		return [
			("vi-VN", "Tiếng Việt"),
			("en-US", "English (US)"),
			("en-GB", "English (UK)"),
			("zh-CN", "Tiếng Trung (Trung Quốc)"),
			("ja-JP", "Tiếng Nhật (Nhật Bản)"),
			("ko-KR", "Tiếng Hàn (Hàn Quốc)")
		]
	
	def get_available_voices(self) -> list:
		"""Get list of available TTS voices"""
		try:
			if self.use_azure_speech:
				# Return the configured Azure voice only (querying full list requires extra API)
				return [{
					'id': self.default_vi_voice,
					'name': self.default_vi_voice,
					'languages': ['vi-VN']
				}]
			voices = self.tts_engine.getProperty('voices') if self.tts_engine else []
			voice_list = []
			for i, voice in enumerate(voices):
				voice_list.append({
					'id': voice.id,
					'name': voice.name,
					'languages': getattr(voice, 'languages', [])
				})
			return voice_list
		except Exception as e:
			print(f"Error getting voices: {e}")
			return []
