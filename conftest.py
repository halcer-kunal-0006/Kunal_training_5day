"""
conftest.py — project root on sys.path so `ticket_analytics` is importable
without installation, in any environment.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
