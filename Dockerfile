FROM python:3.12.8

# Environment Variables
    # Python
ENV PYTHONUNBUFFERED=1 \
    # Poetry
    POETRY_VIRTUALENVS_CREATE=false \ 
    POETRY_HOME="/opt/poetry" \
    # Path
    PATH="/opt/poetry/bin:/opt/poetry/venv:$PATH"

# Change to the app directory
WORKDIR /app

# Stuff needed for OpenGL
RUN apt update \
    && apt install -y --no-install-recommends libsm6 libxext6 \
        ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3
COPY pyproject.toml poetry.lock README.md ./
RUN poetry check --lock && poetry install --no-root

# Copy the rest of the code
ADD additional_files /app/additional_files
ADD draco_models /app/draco_models

# Run the simulation as an entry point
CMD ["poetry", "run", "python", "-m", "draco_models.main", "--config", "additional_files/training_config.yaml"]