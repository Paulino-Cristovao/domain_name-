FROM python:3.10-slim

# set workdir
WORKDIR /app

# Copy only requirements if you use a requirements.txt, otherwise install directly:
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Install all needed packages in one layer
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      transformers \
      datasets \
      peft \
      torch \
      tqdm \
      pandas \
      numpy \
      matplotlib \
      python-Levenshtein \
      gradio \
      openai \
      wandb

# Copy your notebooks and code
COPY . .

# Expose Jupyter if you want to launch it
EXPOSE 8888

# Default command (you can adjust as needed)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
