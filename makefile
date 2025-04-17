# CC = gcc
# CFLAGS = -Wall -O2

# EXE = nn.exe
# SRC = nn.c

# all: $(EXE) run

# $(EXE): $(SRC)
# 	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

# run: $(EXE)
# 	./$(EXE)

# clean:
# 	rm -f $(EXE)


CC = gcc
CFLAGS = -Wall -O2 -pg
LDFLAGS = -lm -pg

EXE = nn.exe
SRC = nn.c
PROF_TXT = gprof_canny.txt
DOT_FILE = callgraph.dot
PNG_FILE = callgraph.png

all: $(EXE) profile png

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) $(LDFLAGS)

run: $(EXE)
	./$(EXE)

profile: run
	gprof $(EXE) gmon.out > $(PROF_TXT)

png: $(PROF_TXT)
	@wget -nc https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py
	chmod +x gprof2dot.py
	python3 gprof2dot.py -s -n 2 -e 2 < $(PROF_TXT) > $(DOT_FILE)
	dot -Tpng $(DOT_FILE) -o $(PNG_FILE)
	@echo "Generated $(PNG_FILE)"

clean:
	rm -f $(EXE) gmon.out $(PROF_TXT) $(DOT_FILE) $(PNG_FILE)
