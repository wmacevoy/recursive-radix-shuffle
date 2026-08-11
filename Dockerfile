FROM python:3.12-slim

# LaTeX toolchain for exact_shuffle.tex + latexmk to drive multi-pass builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
