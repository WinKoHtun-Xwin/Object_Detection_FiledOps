"""Recognition subsystem."""
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery
from app.recognition.live import LiveRecognizer
from app.recognition.worker import RecognitionWorker

dbmod.init_schema()
face_engine = FaceEngine()
gallery = Gallery()
recognition_worker = RecognitionWorker(engine=face_engine, gallery=gallery)
live_recognizer = LiveRecognizer(engine=face_engine, gallery=gallery)

__all__ = ["face_engine", "gallery", "recognition_worker", "live_recognizer"]
