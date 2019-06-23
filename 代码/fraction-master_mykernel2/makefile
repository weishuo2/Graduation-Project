INCLUDES=/usr/local/cuda/samples/common/inc/
CC=g++ -std=c++11
NVCC=nvcc -std=c++11 -I ${INCLUDES}

exe:input_data.o utility.o coloring.o main.o
	${NVCC} main.o coloring.o input_data.o utility.o -lcusparse -o exe #用到了这个库

input_data.o:input_data.cpp input_data.h
	${CC} -c input_data.cpp

utility.o:utility.cpp utility.h
	${CC} -c utility.cpp

coloring.o:coloring.cu
	${NVCC} -c coloring.cu

main.o:main.cu coloring.h
	${NVCC} -c main.cu

clean:
	rm *.o
