CC=g++
CFLAGS=-Wall -g -std=c++0x

.SUFFIXES: .o .cpp .h

SRC_DIRS = ./ ./benchmarks/ ./concurrency_control/ ./storage/ ./system/
INCLUDE = -I. -I./benchmarks -I./concurrency_control -I./storage -I./system

CFLAGS += $(INCLUDE) -D NOGRAPHITE=1 -Werror -Wno-error=overloaded-virtual -O3
LDFLAGS = -Wall -L. -L./libs -pthread -g -lrt -std=c++0x -O3 -no-pie -ljemalloc
LDFLAGS += $(CFLAGS)

CPPS = $(filter-out ./benchmarks/astra_vita.cpp,$(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp)))
OBJS = $(CPPS:.cpp=.o)
DEPS = $(CPPS:.cpp=.d)

all:rundb astra_vita

rundb : $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

astra_vita: benchmarks/astra_vita.cpp
	$(CC) $(CFLAGS) -I../.. -std=c++17 -pthread -o $@ $<

-include $(OBJS:%.o=%.d)

%.d: %.cpp
	$(CC) -MM -MT $*.o -MF $@ $(CFLAGS) $<

%.o: %.cpp
	$(CC) -c $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f rundb astra_vita $(OBJS) $(DEPS)
