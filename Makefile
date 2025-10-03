PYTHON = python3
SCRIPT = part1.py
EXEC = coherence   

PROTOCOL ?= MESI
BENCHMARK ?= bodytrack
CACHE_SIZE ?= 1024
ASSOC ?= 1
BLOCK_SIZE ?= 16

all: $(EXEC)

$(EXEC): $(SCRIPT)
	echo "#!/usr/bin/env $(PYTHON)" > $(EXEC)
	cat $(SCRIPT) >> $(EXEC)
	chmod +x $(EXEC)

run: $(EXEC)
	./$(EXEC) $(PROTOCOL) $(BENCHMARK) $(CACHE_SIZE) $(ASSOC) $(BLOCK_SIZE)

clean:
	rm -f $(EXEC)