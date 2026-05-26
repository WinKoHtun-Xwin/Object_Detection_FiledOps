"""Recognition subsystem."""
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

dbmod.init_schema()
face_engine = FaceEngine()
gallery = Gallery()

__all__ = ["face_engine", "gallery"]
