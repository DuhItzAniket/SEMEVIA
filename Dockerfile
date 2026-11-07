# SEMEVIA inference image. Training data is never copied in; only release artifacts listed in
# artifacts/release_manifest.json (verified by checksum before readiness).
FROM python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    OMP_NUM_THREADS=2 \
    SEMEVIA_TORCH_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-infer.txt .
# Ultralytics pulls in the GUI build of OpenCV; replace it with the headless build. Uninstalling
# opencv-python deletes the shared cv2 package, so the headless wheel is reinstalled afterwards.
RUN pip install -r requirements-infer.txt \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93 \
    && python -c "import cv2, uvicorn, torch, ultralytics, sklearn, onnxruntime; print('imports ok', cv2.__version__)"
COPY requirements-derive.txt .
RUN pip install -r requirements-derive.txt && python -c "import onnx, onnxruntime.quantization; print('onnx', onnx.__version__)"

COPY src ./src
COPY app ./app
COPY configs ./configs
COPY artifacts ./artifacts
COPY reports ./reports
# Fetch any release asset not present in the build context from its public URL and verify SHA256. Derived
# assets (the INT8 hybrid head) are rebuilt here, and check-derived always regenerates them into a temporary file
# to prove the Linux build is byte-identical to the pinned sha256.
RUN python -m src.release fetch && python -m src.release verify && python -m src.release check-derived

RUN useradd --create-home --uid 1000 semevia && chown -R semevia /srv
USER semevia

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
