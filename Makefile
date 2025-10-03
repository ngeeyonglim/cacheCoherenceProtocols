PYTHON = python3
SCRIPT = part1.py
EXEC = coherence   # the "executable" name you want

# Default arguments (can be overridden by user at make command line)
PROTOCOL ?= MESI
BENCHMARK ?= bodytrack
CACHE_SIZE ?= 1024
ASSOC ?= 1
BLOCK_SIZE ?= 16

all: $(EXEC)

# Build wrapper executable
$(EXEC): $(SCRIPT)
	echo "#!/usr/bin/env $(PYTHON)" > $(EXEC)
	cat $(SCRIPT) >> $(EXEC)
	chmod +x $(EXEC)

# Run with user-provided arguments
run: $(EXEC)
	./$(EXEC) $(PROTOCOL) $(BENCHMARK) $(CACHE_SIZE) $(ASSOC) $(BLOCK_SIZE)

clean:
	rm -f $(EXEC)