TARGET := main
SRC_DIR := ./src
OBJ_DIR := ./obj
LIB_DIR := ./lib
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
LDFLAGS := 
CPPFLAGS := -I $(LIB_DIR) -L/usr/X11R6/lib -lm -lpthread -lX11 -ljpeg -std=c++17 -O3
CXXFLAGS :=

$(TARGET): $(OBJ_FILES)
	g++ $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p obj
	g++ $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm $(OBJ_DIR)/*.o
	rm $(TARGET)