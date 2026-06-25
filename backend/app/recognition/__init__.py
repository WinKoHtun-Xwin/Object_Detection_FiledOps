"""Recognition subsystem."""
from app.config import settings
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery
from app.recognition.live import LiveRecognizer
from app.recognition.worker import RecognitionWorker

dbmod.init_schema()
face_engine = FaceEngine()                                   # 640² — offline clip worker
live_face_engine = FaceEngine(det_size=settings.live_det_size)  # smaller — live crops
gallery = Gallery()
recognition_worker = RecognitionWorker(engine=face_engine, gallery=gallery)
live_recognizer = LiveRecognizer(engine=live_face_engine, gallery=gallery)

__all__ = [
    "face_engine",
    "live_face_engine",
    "gallery",
    "recognition_worker",
    "live_recognizer",
]
