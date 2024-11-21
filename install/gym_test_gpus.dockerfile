

FROM tensorflow/tensorflow:latest-gpu-jupyter

# # Copy all relevant files into the image.
COPY ./ .

RUN pip install --upgrade pip
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

## update conda and install mamba
RUN conda update -n base conda

RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba

RUN mamba install -c conda-forge scikit-image ipython tqdm pandas natsort matplotlib transformers -y && \
    pip install -U doodleverse_utils && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

ENV PROJ_LIB='/opt/conda/share/proj'

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Indicate that Jupyter Lab inside the container will be listening on port 8888.
EXPOSE 8888

# COPY ./ .
# CMD jupyter lab SDS_coastsat_classifier.ipynb
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
# CMD ["jupyter", "lab", "SDS_coastsat_classifier.ipynb","--ip=0.0.0.0", "--no-browser", "--allow-root"]
CMD["python","test_gpus.py"]