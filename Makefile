NVCC = nvcc
CXX = g++
NVCC_FLAGS = -arch=sm_50 -I.
CXX_FLAGS = -I.
OBJS = matrix_ops.o kalman_gpu.o main.o

all: kalman_demo

matrix_ops.o: matrix_ops.cpp matrix_ops.h
	$(CXX) $(CXX_FLAGS) -c matrix_ops.cpp -o matrix_ops.o

kalman_gpu.o: kalman_gpu.cu kalman_gpu.cuh
	$(NVCC) $(NVCC_FLAGS) -c kalman_gpu.cu -o kalman_gpu.o

main.o: main.cpp kalman_cpu.h matrix_ops.h
	$(NVCC) $(NVCC_FLAGS) -c main.cpp -o main.o

kalman_demo: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o kalman_demo -lcublas -lcusolver

clean:
	rm -f *.o kalman_demo
