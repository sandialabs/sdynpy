FROM python:3
RUN git clone --depth 1 https://github.com/sandialabs/sdynpy.git
RUN cd sdynpy/ && ls -la
RUN pip install .[all]
