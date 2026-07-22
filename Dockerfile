FROM python:3.12-slim

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

RUN vesuvius.accept_terms --yes

ENTRYPOINT ["python", "src/pipeline.py"]
CMD ["--segment_id", "20230827161847", "--y0", "2000", "--y1", "3500", \
     "--x0", "2500", "--x1", "4000", "--epochs", "40"]
