# =====================
# Dockerfile para ROS1 Melodic + Gazebo + dependencias del repo
# =====================

FROM osrf/ros:melodic-desktop-full

# =====================
# Dependencias básicas de ROS, Gazebo y herramientas
# =====================
RUN apt-get update && apt-get install -y \
    git \
    wget \
    software-properties-common \
    ros-melodic-gazebo-ros \
    ros-melodic-gazebo-ros-control \
    ros-melodic-ros-control \
    ros-melodic-ros-controllers \
    ros-melodic-rqt \
    ros-melodic-rqt-common-plugins \
    ros-melodic-robot-localization \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# =====================
# Dependencias de Sistema y Gráficos
# =====================
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    lsb-release \
    gnupg2 \
    # En 18.04 el paquete core es este:
    libopenscenegraph-dev \
    # Dependencia de renderizado para DART
    libopenscenegraph100v5 \
    && rm -rf /var/lib/apt/lists/*

# =====================
# DART (Componentes específicos para Melodic)
# =====================
RUN add-apt-repository ppa:dartsim/ppa -y && \
    apt-get update && \
    apt-get install -y \
        libdart6-dev \
        libdart6-utils-urdf-dev \
        libdart6-gui-osg-dev \
    && rm -rf /var/lib/apt/lists/*
    
    
# =====================
# Catkin Tools 
# =====================
RUN apt-get update && apt-get install -y curl lsb-release gnupg2 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list \
    && apt-get update \
    && apt-get install -y python-catkin-tools

# =====================
# Workspace
# =====================
WORKDIR /catkin_ws
RUN mkdir -p src

# =====================
# Build inicial (workspace vacío, solo para crear devel)
# =====================
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin build -j 2"

# =====================
# Comando por defecto
# =====================
CMD ["bash"]