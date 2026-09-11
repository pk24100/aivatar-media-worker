"""Implementation modules behind the handler.py facade.

Code was split out of handler.py for readability. Modules here read shared
state and config via `import handler` at call time so existing
monkey-patching of handler.* attributes (tests, modal_worker) keeps working.
"""
