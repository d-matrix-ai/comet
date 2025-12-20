#pragma once


// Energy per operation (in picojoules)
const float DIV_ENERGY  = 0.8f;
const float EXP_ENERGY  = 3.86f;
const float ADD_ENERGY  = 0.11f;
const float MULT_ENERGY = 0.64f;
const float MAX_ENERGY  = 0.0025f;
const float SQRT_ENERGY = 2.84f;

// Latency per operation (in cycles)
const float DIV_CYCLES  = 1.0f;
const float EXP_CYCLES  = 3.0f;
const float ADD_CYCLES  = 1.0f;
const float MULT_CYCLES = 1.0f;
const float MAX_CYCLES  = 1.0f;
const float SQRT_CYCLES = 1.0f;