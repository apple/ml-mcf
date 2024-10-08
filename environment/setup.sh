apt-get update
apt-get install unzip
apt-get install nano
apt-get -o Dpkg::Options::="--force-confnew" install -y \
    python-tk \
    libffi-dev \
    libhdf5-dev \
    htop \
    ffmpeg \
    xz-utils \
    libatlas3-base \
    tmux \
    locales \
    build-essential git \
    unzip \
    && \
    apt-get autoremove && \
    apt-get clean

apt-get install xvfb -y
apt-get install libgl1-mesa-glx -y
apt-get python-qt4

pip install -r environment/requirements.txt
# tmux commands
tmux show -g > ~/.tmux.conf