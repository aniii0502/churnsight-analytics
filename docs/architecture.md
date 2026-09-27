# System Architecture

## Separation of Concerns
The application follows a strict modular design, cleanly separating business logic from the presentation layer to ensure independent testability.

* **`app/etl.py`**: The data pipeline. Enforces strict schema contracts, adapts known public datasets, parses dates, computes leap-year-safe tenures, and structures the matrices required for downstream modeling.
* **`app/survival_models.py`**: The statistical engine. Ingests cleaned data to fit the `lifelines` Kaplan-Meier estimator and applies `scikit-learn`'s K-Means clustering algorithm.
* **`app/main.py`**: The presentation layer. Contains zero business logic. It handles Streamlit state, caches ETL operations via `@st.cache_data`, and maps calculated arrays to user interface components.

## Containerization and Infrastructure
The platform is containerized for zero-config deployment across environments.
* **Dockerfile**: Utilizes a lightweight `python:3.11-slim` base image. It aggressively caches the `requirements.txt` installation layer to speed up subsequent builds, exposes port 8501, and binds the Streamlit server to `0.0.0.0`.
* **Docker Compose**: Wires the container execution. It maps the internal port to the host and mounts the `./data` and `./app` directories as volumes, enabling live-reloading of datasets and code without requiring container rebuilds.

## CI/CD Pipeline
The repository utilizes GitHub Actions for continuous integration. On every push to the main branch, the pipeline provisions a runner, installs dependencies, and executes the `pytest` suite against the ETL and modeling layers to prevent regressions.