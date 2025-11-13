#!/usr/bin/env python3
"""Quick script to verify the MCP server is running and accessible."""

import sys

import requests


def check_server():
    """Check if the MCP server is running."""
    url = "http://localhost:8001/sse"

    try:
        # Try to connect to the SSE endpoint
        response = requests.get(url, timeout=2, stream=True)
        print(f"✅ Server is running at {url}")
        print(f"   Status code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {url}")
        print("   Make sure the server is running: python mcp_server.py")
        return False
    except requests.exceptions.Timeout:
        print(f"⏱️  Connection to {url} timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False


if __name__ == "__main__":
    success = check_server()
    sys.exit(0 if success else 1)
