FROM arena-base

LABEL org.opencontainers.image.version="0.16.0"

USER root
WORKDIR /submission
COPY . /submission/
RUN MARS_ROVER_REQUIRE_PLATFORM=1 python -m unittest discover -s /submission/tests -v

USER 10000:10000
CMD ["python", "/submission/train.py"]
