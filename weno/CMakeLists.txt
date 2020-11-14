#libray include directory
INCLUDE_DIRECTORIES(include)

#weno libray
ADD_LIBRARY(weno
  source/stencil.cc
)

DEAL_II_SETUP_TARGET(weno)

