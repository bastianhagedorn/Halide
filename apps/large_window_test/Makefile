include ../support/Makefile.inc

auto: clean_auto test_auto

large_window_test: large_window_test.cpp
	$(CXX) $(CXXFLAGS) large_window_test.cpp $(LIB_HALIDE) -o large_window_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

large_win_ref.o: large_window_test
	./large_window_test 0

large_win_auto.o: large_window_test
	./large_window_test -1

large_win_naive.o: large_window_test
	HL_AUTO_NAIVE=1 ./large_window_test -1

large_win_sweep.o: large_window_test
	HL_AUTO_SWEEP=1 ./large_window_test -1

test_%: test.cpp large_win_%.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl $(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f test large_window_test large_win.h large_win*.o ref_perf.txt test_*
