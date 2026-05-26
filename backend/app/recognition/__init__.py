"""Recognition subsystem."""
from app.recognition import db as dbmod
from app.recognition.gallery import Gallery

dbmod.init_schema()
gallery = Gallery()

__all__ = ["gallery"]
