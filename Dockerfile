FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 

CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0"]