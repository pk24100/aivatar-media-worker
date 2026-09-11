#!/bin/bash
echo "Starting AiVatar worker in ${AIVATAR_RUNTIME_MODE:-modal} mode..."
python3 -u -c "import asyncio, handler; asyncio.run(handler.run_worker_app())"
