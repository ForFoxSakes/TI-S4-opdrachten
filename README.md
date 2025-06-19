# TI-S4 Opdrachten

Deze repository bevat alle opdrachten voor het semester TI-S4, netjes gescheiden in categorieën.

## Structuur

- `cpp/`: C++ opdrachten (via CMake)
- `alds/`: ALDS Jupyter notebooks
- `vision/`: Vision opdrachten met Python en OpenCV

## Instructies

### DevContainer gebruiken

1. Open deze map in VS Code
2. Kies "Reopen in Container" als dat gevraagd wordt
3. De omgeving bevat:
   - C++ tools (GCC, CMake)
   - Python + Jupyter
   - Vision libraries zoals OpenCV

### C++ opdracht builden

```bash
cd cpp/opdracht1
cmake .
make
./main
