# Base image
FROM pytorch/pytorch:latest

# Create and set the HOME directory
WORKDIR /workspace
ENV HOME=/workspace

RUN mkdir -p /workspace/diffversify/results/real_results/diffnaps /workspace/diffversify/results/synth_results/diffnaps/runs /workspace/diffversify/code /workspace/diffversify/data
# Add our job's file to this directory
COPY ./diffversify/code /workspace/diffversify/code
COPY ./diffversify/data /workspace/diffversify/data


# Give the OVHcloud user (42420:42420) access to this directory
RUN chown -R 42420:42420 /workspace

# Install required packages and libraries
RUN apt-get update && apt-get install -y  vim git curl ;
# RUN pip install -r requirements.txt
RUN python -m pip install pandas scipy torchmetrics scikit-learn matplotlib seaborn tqdm tabulate openpyxl tensorboard jupyterlab

# Run your job (Optional. You can specify your file when launching the AI Training job)
# CMD ["python", "/workspace/main.py"]


######################## CMD guideline :

# To build : 
#  $docker build . -t diff:test

# To test locally :
# $docker run --rm -it --gpus all --user=42420:42420 diff:test
# then run exp as before

# jupyter-lab --ip 0.0.0.0 --allow-root --no-browser