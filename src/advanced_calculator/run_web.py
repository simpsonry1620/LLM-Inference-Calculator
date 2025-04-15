#!/usr/bin/env python3
"""
Run script for the advanced calculator web interface.
"""
import argparse
import os
from src.advanced_calculator.web import create_app

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the LLM Infrastructure Scaling Calculator Web Interface"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the web app."""
    args = parse_args()
    
    app = create_app()
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main() 