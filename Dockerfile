# ---------------------------
# Stage 1: Build environment
# ---------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (lighter wheels)
RUN pip3 install --no-cache-dir --upgrade pip

# Install only required Python packages
RUN pip3 install --no-cache-dir \
    tensorflow[and-cuda]==2.17.0 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    joblib==1.4.2

# ---------------------------
# Final Stage
# ---------------------------
FROM base

# Set working directory for scripts/models
WORKDIR /GSK3BMTPred

# Copy only required files
COPY PaDEL ./PaDEL
COPY padel_destype.xml gsk3bmtpred_predict.py gsk3bmt_model.h5 padel_scaler.pkl X_train.csv ./

# Default working directory for user files
WORKDIR /WorkPlace

# Entrypoint runs the predictor
ENTRYPOINT ["python3", "/GSK3BMTPred/gsk3bmtpred_predict.py"]
