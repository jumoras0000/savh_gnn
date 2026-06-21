# -*- coding: utf-8 -*-
"""Interface web Panacée : backend Starlette (REST + SSE) + frontend autonome.

Sous-modules :
  service : fonctions pures (découverte de runs, lecture live, verdict clinique).
  server  : application ASGI Starlette (API + fichiers statiques).
  run     : lanceur uvicorn.
"""
