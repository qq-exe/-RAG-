# 企业部署轻量级Python3.1镜像只有100MB
FROM python:3.10-slim  

# [后续的COPY、RUN、CMD命令都会在这个目录执行]
WORKDIR /app

# [优化缓存]先复制依赖清单，再安装依赖
COPY requirements.txt .

# [安装依赖]--no-cache-dir避免pip在容器留下缓存文件，减小镜像体积
Run pip install --no-cache-dir -r requirements.txt

# [复制代码]把当前目录下所有代码复制到容器的/app目录下
COPY . . 

# [暴露端口]容器启动时执行的命令，注意生产环境不加--reload,那是开发模式用的
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]