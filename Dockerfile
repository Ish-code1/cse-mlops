FROM python:3.14-slim
#start with a light weight python image

#set the working directory inside the container
WORKDIR /app

#copy dependency list before copying application code
COPY requirements.txt .
#avoid retaining pip's download cache in the image 
RUN pip install --no-cache-dir -r requirements.txt
#copy the application code and saved model into /app
COPY . .
#Document theport used by FASTAPI application 
EXPOSE 8000

#start the api server when the container launches
#0.0.0.0 allows the connection through containers 

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
