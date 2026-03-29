#!/bin/bash
echo "Starting AiVatar worker in ${AIVATAR_RUNTIME_MODE:-serverless} mode..."
python3 -u handler.py
