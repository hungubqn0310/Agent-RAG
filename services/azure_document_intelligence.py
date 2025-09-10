import os
from typing import Optional

try:
	from azure.ai.documentintelligence import DocumentIntelligenceClient
	from azure.core.credentials import AzureKeyCredential
	export_available = True
except Exception:
	export_available = False
	DocumentIntelligenceClient = None  # type: ignore
	AzureKeyCredential = None  # type: ignore


class AzureDocumentIntelligenceService:
	"""Lightweight wrapper around Azure Document Intelligence 'prebuilt-read' model."""
	def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None, model_id: str = "prebuilt-read") -> None:
		self.endpoint = endpoint or os.getenv("AZURE_DOCINT_ENDPOINT") or ""
		self.key = key or os.getenv("AZURE_DOCINT_KEY") or ""
		self.model_id = model_id
		self.available: bool = False
		self.error: str = ""

		if not export_available:
			self.error = "azure.ai.documentintelligence SDK is not installed."
			return

		if not (self.endpoint and self.key):
			self.error = "Missing AZURE_DOCINT_ENDPOINT or AZURE_DOCINT_KEY."
			return

		try:
			self.client = DocumentIntelligenceClient(self.endpoint, AzureKeyCredential(self.key))
			self.available = True
		except Exception as e:
			self.error = f"Failed to init DocumentIntelligenceClient: {e}"
			self.available = False

	def extract_text(self, file_path: str) -> str:
		"""Extract text using Azure Document Intelligence. Returns empty string on failure."""
		if not self.available:
			print("Azure DocInt not available; check endpoint/key and SDK installation.")
			return ""
		try:
			with open(file_path, "rb") as f:
				content = f.read()
			print(f"Azure DocInt: analyzing file {file_path} with model {self.model_id}")
			poller = self.client.begin_analyze_document(
				model_id=self.model_id,
				body=content,
				content_type="application/octet-stream",
			)
			result = poller.result()
			# Prefer full content if available, else aggregate page lines
			text = getattr(result, "content", "") or ""
			if text:
				print(f"Azure DocInt: extracted text length={len(text)}")
				return text
			pages = getattr(result, "pages", []) or []
			collected = []
			for page in pages:
				lines = getattr(page, "lines", []) or []
				for line in lines:
					line_content = getattr(line, "content", "") or ""
					if line_content:
						collected.append(line_content)
			joined = "\n".join(collected)
			print(f"Azure DocInt: aggregated page lines length={len(joined)}")
			return joined
		except Exception as e:
			print(f"Azure DocInt error: {e}")
			return "" 