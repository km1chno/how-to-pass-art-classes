CPPFLAGS := -L/usr/X11R6/lib -lm -lpthread -lX11 -ljpeg -std=c++17 -O3

all:
	g++ -o imageprep.out -I ../../lib main.cpp datasetManager.cpp imagesHelper.cpp $(CPPFLAGS)