FROM python:3.10
RUN git clone --depth 1 https://github.com/sandialabs/sdynpy.git && \
    cd sdynpy/ && \
    pip install .[all]
