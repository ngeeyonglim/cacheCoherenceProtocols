#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <cstdint>

using namespace std;

/**
 * Cache Line structure
 */
struct CacheLine
{
    bool valid = false;
    bool dirty = false;
    uint32_t tag = 0;
    int lruCounter = 0; // for LRU replacement
};

/**
 * Cache Class
 */
class Cache
{
private:
    int cacheSize;
    int associativity;
    int blockSize;
    int numSets;

    vector<vector<CacheLine>> sets; // 2D: [set][way]

    // Statistics
    long hits = 0;
    long misses = 0;

public:
    Cache(int size, int assoc, int block)
    {
        cacheSize = size;
        associativity = assoc;
        blockSize = block;

        numSets = cacheSize / (blockSize * associativity);
        sets.resize(numSets, vector<CacheLine>(associativity));
    }

    bool access(uint32_t address, bool isWrite, long &cycles)
    {
        uint32_t blockAddr = address / blockSize;
        uint32_t setIndex = blockAddr % numSets;
        uint32_t tag = blockAddr / numSets;

        auto &set = sets[setIndex];

        // Search for hit
        for (int i = 0; i < associativity; i++)
        {
            if (set[i].valid && set[i].tag == tag)
            {
                // Hit
                hits++;
                cycles += 1; // cache hit = 1 cycle
                // update LRU
                set[i].lruCounter = 0;
                for (int j = 0; j < associativity; j++)
                {
                    if (i != j)
                        set[j].lruCounter++;
                }
                if (isWrite)
                    set[i].dirty = true;
                return true;
            }
        }

        // Miss
        misses++;
        cycles += 100; // miss penalty = fetch from DRAM (100 cycles)

        // Find victim (LRU)
        int victim = 0;
        for (int i = 1; i < associativity; i++)
        {
            if (set[i].lruCounter > set[victim].lruCounter)
            {
                victim = i;
            }
        }

        // Write-back if dirty
        if (set[victim].valid && set[victim].dirty)
        {
            cycles += 100; // writeback cost
        }

        // Replace
        set[victim].valid = true;
        set[victim].dirty = isWrite;
        set[victim].tag = tag;
        set[victim].lruCounter = 0;
        for (int j = 0; j < associativity; j++)
        {
            if (j != victim)
                set[j].lruCounter++;
        }

        return false;
    }

    void printStats()
    {
        cout << "Cache Hits: " << hits << endl;
        cout << "Cache Misses: " << misses << endl;
    }
};

/**
 * Simulator
 */
class Simulator
{
private:
    Cache cache;
    long totalCycles = 0;
    long computeCycles = 0;
    long idleCycles = 0;
    long loads = 0;
    long stores = 0;

public:
    Simulator(int cacheSize, int assoc, int blockSize)
        : cache(cacheSize, assoc, blockSize) {}

    void runTrace(const string &filename)
    {
        ifstream infile(filename);
        if (!infile.is_open())
        {
            cerr << "Error opening trace file: " << filename << endl;
            return;
        }

        string line;
        while (getline(infile, line))
        {
            if (line.empty())
                continue;
            stringstream ss(line);
            int label;
            string valueStr;
            ss >> label >> valueStr;

            if (label == 2)
            {
                // computation
                long cycles = stol(valueStr, nullptr, 16);
                computeCycles += cycles;
                totalCycles += cycles;
            }
            else
            {
                uint32_t addr = stoul(valueStr, nullptr, 16);
                bool isWrite = (label == 1);

                long cyclesBefore = totalCycles;
                cache.access(addr, isWrite, totalCycles);
                idleCycles += (totalCycles - cyclesBefore);

                if (isWrite)
                    stores++;
                else
                    loads++;
            }
        }
        infile.close();
    }

    void printResults()
    {
        cout << "===== Simulation Results =====" << endl;
        cout << "Total Cycles: " << totalCycles << endl;
        cout << "Compute Cycles: " << computeCycles << endl;
        cout << "Idle Cycles: " << idleCycles << endl;
        cout << "Loads: " << loads << "  Stores: " << stores << endl;
        cache.printStats();
    }
};

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cerr << "Usage: coherence protocol input_file cache_size associativity block_size" << endl;
        return 1;
    }

    string protocol = argv[1]; // unused in Part 1
    string inputFile = argv[2];
    int cacheSize = stoi(argv[3]);
    int associativity = stoi(argv[4]);
    int blockSize = stoi(argv[5]);

    Simulator sim(cacheSize, associativity, blockSize);
    sim.runTrace(inputFile + "_0.data"); // Part 1: only use core 0
    sim.printResults();

    return 0;
}