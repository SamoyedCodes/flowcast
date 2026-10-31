"""Vercel entrypoint; keep the local dashboard and deployed API on the same handler."""
from dashboard.server import Handler


class handler(Handler):
    pass
