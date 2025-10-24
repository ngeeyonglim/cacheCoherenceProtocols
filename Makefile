PYTHON = python3
SCRIPT = part2.py
EXEC = coherence

# Default arguments (can be overridden by user at make command line)
PROTOCOL ?= MESI
BENCHMARK ?= bodytrack
CACHE_SIZE ?= 4096
ASSOC ?= 2
BLOCK_SIZE ?= 32

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