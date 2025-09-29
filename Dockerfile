FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY st_app.py . 
COPY cli_ans.py . 
COPY docs/ ./docs

EXPOSE 8501
CMD ["streamlit", "run", "st_app.py", "--server.port=8501", "--server.address=0.0.0.0"]