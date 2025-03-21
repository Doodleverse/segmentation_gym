# FILE: Dockerfile
FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /gym

# Copy the src code, the model from scratch_test and the seg_images in folders script
COPY ./test_gpus.py /gym/test_gpus.py
COPY ./seg_images_in_folder_no_tkinter.py /gym/seg_images_in_folder_no_tkinter.py
COPY ./train_model_script_no_tkinter.py /gym/train_model_script_no_tkinter.py
COPY ./make_dataset_no_tkinter.py /gym/make_dataset_no_tkinter.py
COPY ./batch_train_models_no_tkinter.py /gym/batch_train_models_no_tkinter.py

COPY ./src /gym/src

# Copy the scripts and pixi lock file so that the setup will run
COPY pixi.lock /gym/pixi.lock
COPY pyproject.toml /gym/pyproject.toml

ENV CONDA_OVERRIDE_CUDA=11.8

RUN /usr/local/bin/pixi install --manifest-path pyproject.toml --locked

# Entrypoint shell script ensures that any commands we run start with `pixi shell`,
# which in turn ensures that we have the environment activated
# when running any commands.
COPY entrypoint.sh /gym/entrypoint.sh
RUN chmod 700 /gym/entrypoint.sh
ENTRYPOINT [ "/gym/entrypoint.sh" ]
