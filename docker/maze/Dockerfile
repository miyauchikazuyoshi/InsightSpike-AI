FROM python:3.11-slim

# Avoid OpenBLAS/Accelerate multi-thread surprises
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal deps for maze experiments
# Add matplotlib for visualization (Agg backend used; no system GUI deps needed)
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "networkx==3.4.2" \
    "matplotlib==3.8.4"

# Default command prints versions; Makefile overrides with the run command
CMD python -c "import numpy, networkx; print('NumPy', numpy.__version__); print('NetworkX', networkx.__version__)"
