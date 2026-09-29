# SOLUTION NAME: rover_best_v1
FROM arena-base

USER root
WORKDIR /submission
COPY . /submission/
RUN python -m pip install --no-build-isolation --no-deps --force-reinstall /submission
USER 10000:10000
CMD ["python", "/submission/train.py"]
