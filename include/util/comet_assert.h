#pragma once
#include <iostream>
#define COMET_ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)

#define COMET_ERROR(message) \
    do { \
      std::cerr << message << std::endl; \
      std::terminate(); \
    } while (false)


const int verbose_level=0;
const bool RTS_dependent = false; //relative time step dependent
const bool perfect_pipeline = false; //true;

// const bool calc_noc_energy=0;