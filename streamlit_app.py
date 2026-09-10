import sys
import os

# Ensure app directory is in path and execute main app
sys.path.append(os.path.dirname(__file__))

from app.app import main

if __name__ == '__main__':
    main()
