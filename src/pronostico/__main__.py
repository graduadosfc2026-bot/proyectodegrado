"""Permite ejecutar el paquete con `python -m pronostico`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
