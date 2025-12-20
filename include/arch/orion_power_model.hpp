#pragma once
#include <iostream>
#include <cmath>

namespace arch{
    double power_summary_router(double channel_width, int input_switch, int output_switch, uint32_t hop, double trc, double tva, double tsa, double tst, double tl, double tenq, double Q, uint32_t mesh_edge);
}