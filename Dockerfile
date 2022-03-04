# Based on:
# https://stackoverflow.com/questions/68673221/warning-running-pip-as-the-root-user
# https://www.thegeekdiary.com/run-docker-as-a-non-root-user/

FROM huggingface/transformers-tensorflow-gpu



# Get username from commandline
ARG username

RUN pip install --upgrade pip

# Set username
RUN adduser --disabled-password $username
USER $username
WORKDIR /home/$username
COPY --chown=$username:$username requirements.txt requirements.txt

RUN pip install --user -r requirements.txt

ENV PATH="/home/$username/.local/bin:${PATH}"

COPY --chown=$username:$username . .

ENV NUMBA_CACHE_DIR=/tmp/


# Build: sudo docker build --build-arg username=$USER -t test01 .
# Run: sudo docker run --rm test01 id