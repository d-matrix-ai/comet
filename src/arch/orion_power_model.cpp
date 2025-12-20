// #include <iostream>
// #include <cmath>
#include "arch/orion_power_model.hpp"

using namespace std;

namespace arch{
    // Constants
    const int H_INVD2 = 8;
    const int W_INVD2 = 3;
    const int H_DFQD1 = 8;
    const int W_DFQD1 = 16;
    const int H_ND2D1 = 8;
    const int W_ND2D1 = 3;
    const int H_SRAM = 8;
    const int W_SRAM = 6;

    const double Vdd = 0.9;
    const double R = 606.321;
    const double IoffP = 0.00000102;
    const double IoffN = 0.00000102;
    const double IoffSRAM = 0.00000032;

    const double Cg_pwr = 0.000000000000000534;
    const double Cd_pwr = 0.000000000000000267;
    const double Cgdl = 0.0000000000000001068;
    const double Cg = 0.000000000000000534;
    const double Cd = 0.000000000000000267;

    const double LAMBDA = 0.016;
    const double MetalPitch = 0.000080;
    const double Rw = 0.0435644;

    // Derived constants
    const double Ci_delay = 3 * (Cg + Cgdl);
    const double Co_delay = 3 * Cd;
    const double Ci = (1.0 + 2.0) * Cg_pwr;
    const double Co = (1.0 + 2.0) * Cd_pwr;
    const double FO4 = R * (3.0 * Cd + 12 * Cg + 12 * Cgdl);
    const double tCLK = 20 * FO4;
    const double fCLK = 1.0 / tCLK;

    const double ChannelPitch = 2.0 * MetalPitch;
    const double CrossbarPitch = 2.0 * MetalPitch;

    // Configurable parameters
    const int numVC = 3;
    const int buf_size = 10;
    const int depthVC = 10;
    const int output_buffer_size = 1;


    double calculate_latency(int hop, double trc, double tva, double tsa, double tst, double tl, double tenq, double Q, double channel_width) {
        return hop * (trc + tva + tsa + tst + tl) + (tenq) * (Q / channel_width);
    }

    //Power Module Calculate Channel
    double Power_Module_powerRepeatedWire(double L, double K, double M, double N, double Cw) {
        double segments = M * N;
        double Ca = K * (Ci + Co) + Cw * (L / segments);
        double Pa = 0.5 * Ca * Vdd * Vdd * fCLK;
        return Pa * M * N;
    }

    double Power_Module_powerWireClk(double M, double W, double Cw) {
        double columns = H_DFQD1 * MetalPitch / ChannelPitch;
        double clockLength = W * ChannelPitch;
        double Cclk = (1 + 5.0 / 16.0 * (1 + Co_delay / Ci_delay)) * (clockLength * Cw * columns + W * Ci_delay);
        return M * Cclk * (Vdd * Vdd) * fCLK;
    }

    double Power_Module_powerRepeatedWireLeak(double K, double M, double N) {
        double Pl = K * 0.5 * (IoffN + 2.0 * IoffP) * Vdd;
        return Pl * M * N;
    }

    double Power_Module_powerWireDFF(double M, double W, double alpha = 1.0) {
        double Cdin = 2 * 0.8 * (Ci + Co) + 2 * (2.0 / 3.0 * 0.8 * Co);
        double Cclk = 2 * 0.8 * (Ci + Co) + 2 * (2.0 / 3.0 * 0.8 * Cg_pwr);
        double Cint = (alpha * 0.5) * Cdin + alpha * Cclk;
        return Cint * M * W * (Vdd * Vdd) * fCLK;
    }

    double Power_Module_areaChannel(double K, double N, double M, double channel_width) {
        double Adff = M * W_DFQD1 * H_DFQD1;
        double Ainv = M * N * (W_INVD2 + 3 * K) * H_INVD2;
        return channel_width * (Adff + Ainv) * MetalPitch * MetalPitch;
    }
    //////////////////////////


    //Power Module Calculate Buffer
    double Power_Module_powerWordLine(double memoryWidth, double memoryDepth, double channel_width, double Cw) {
        //wordline capacitance
        double Ccell = 2 * (4.0 * LAMBDA) * Cg_pwr + 6 * MetalPitch * Cw;
        double Cwl = memoryWidth * Ccell;
        //wordline circuits
        double Warray = 8 * MetalPitch + memoryDepth;
        double x = 1.0 + (5.0 / 16.0) * (1 + Co / Ci);
        double Cpredecode = x * (Cw * Warray * Ci);
        double Cdecode = x * Cwl;
        //bitline circuits
        double Harray = 6 * memoryWidth * MetalPitch;
        double y = (1 + 0.25) * (1 + Co / Ci);
        double Cprecharge = y * (Cw * Harray + 3 * channel_width * Ci);
        double Cwren = y * (Cw * Harray + 2 * channel_width * Ci);

        double Cbd = Cprecharge + Cwren;
        double Cwd = 2 * Cpredecode + Cdecode;

        return (Cbd + Cwd) * Vdd * Vdd * fCLK;
    }    
    double Power_Module_powerMemoryBitRead(double memoryDepth, double Cw) {
        double Ccell = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw;
        double Cbl = memoryDepth * Ccell;
        double Vswing = Vdd;
        return Cbl * Vdd * Vswing * fCLK;
    }

    double Power_Module_powerMemoryBitWrite(double memoryDepth, double Cw) {
        double Ccell = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw;
        double Cbl = memoryDepth * Ccell;
        double Ccc = 2 * (Co + Ci);

        return (0.5 * Ccc * (Vdd * Vdd)) + (Cbl * Vdd * Vdd * fCLK);
    }

    // Power Module calculate Switch
    double Power_Module_areaCrossbar(double Inputs, double Outputs, double channel_width) {
            return (Inputs * channel_width * CrossbarPitch) * (Outputs * channel_width * CrossbarPitch);
        }

        double Power_Module_areaOutputModule(double Outputs, double channel_width) {
            double Adff = Outputs * W_DFQD1 * H_DFQD1;
            return channel_width * Adff * MetalPitch * MetalPitch;
        }

        double Power_Module_powerCrossbarLeak(double width, double inputs, double outputs, double Cw) {
            double Wxbar = width * outputs * CrossbarPitch;
            double Hxbar = width * inputs * CrossbarPitch;

            double CwIn = Wxbar * Cw;
            double CwOut = Hxbar * Cw;

            double Cxi = (1.0 / 16.0) * CwOut;
            double Cti = (1.0 / 16.0) * CwIn;

            return 0.5 * (IoffN + 2 * IoffP) * width * (inputs * outputs * Cxi + inputs * Cti + outputs * Cti) / Ci;
        }

        double Power_Module_powerCrossbar(double width, double inputs, double outputs, double Cw) {
            double Wxbar = width * outputs * CrossbarPitch;
            double Hxbar = width * inputs * CrossbarPitch;

            double CwIn = Wxbar * Cw;
            double CwOut = Hxbar * Cw;

            double Cxi = (1.0 / 16.0) * CwOut;
            double Cxo = 4.0 * Cxi * (Co_delay / Ci_delay);

            double Cti = (1.0 / 16.0) * CwIn;
            double Cto = 4.0 * Cti * (Co_delay / Ci_delay);

            double CinputDriver = 5.0 / 16.0 * (1 + Co_delay / Ci_delay) * (0.5 * Cw * Wxbar + Cti);

            double Cin = CinputDriver + CwIn + Cti + (outputs * Cxi);
            double Cout = CwOut + Cto + (inputs * Cxo);

            return 0.5 * (Cin + Cout) * (Vdd * Vdd * fCLK); //Watt
        }

        double Power_Module_powerCrossbarCtrl(double width, double inputs, double outputs, double Cw) {
            double Wxbar = width * outputs * CrossbarPitch;
            double Hxbar = width * inputs * CrossbarPitch;

            double CwIn = Wxbar * Cw;
            double Cti = (5.0 / 16.0) * CwIn;

            double Cctrl = width * Cti + (Wxbar + Hxbar) * Cw;
            double Cdrive = (5.0 / 16.0) * (1 + Co_delay / Ci_delay) * Cctrl;

            return (Cdrive + Cctrl) * (Vdd * Vdd) * fCLK;
        }

        double Power_Module_powerOutputCtrl(double width, double Cw) {
            double Woutmod = width * ChannelPitch;
            double Cen = Ci;

            double Cenable = (1 + 5.0 / 16.0) * (1.0 + Co / Ci) * (Woutmod * Cw + width * Cen);

            return Cenable * (Vdd * Vdd) * fCLK;
        }    


    double power_summary_router(double channel_width, int input_switch, int output_switch, uint32_t hop, double trc, double tva, double tsa, double tst, double tl, double tenq, double Q, uint32_t mesh_edge) {

        double K =8.1, M=2, N=1, wire_length=2.0, Cw=2.0 * 0.000000000000267339 + 2.0 * 0.000000000000267339;
        //Power Module Calculate Channel
        auto channel_wire_power = Power_Module_powerRepeatedWire(wire_length, K, M, N, Cw);

        auto channel_clk_power = Power_Module_powerWireClk(M, wire_length, Cw);

        auto channel_DFF_power = Power_Module_powerWireDFF(M, wire_length);
        // auto channel_leak_power = Power_Module_powerRepeatedWireLeak(K, M, N);

        //Power Module Calculate Buffer
        auto depth = numVC * depthVC;
        auto Pwl = Power_Module_powerWordLine(channel_width, depth, channel_width, Cw);
        auto Prd = Power_Module_powerMemoryBitRead(depth, Cw)*channel_width;
        auto Pwr = Power_Module_powerMemoryBitWrite(depth, Cw)*channel_width;
        auto inputReadPower = Pwl + Prd;
        auto inputWritePower = Pwl + Pwr;

        //Power Module calcSwitch
        // double switchPowerLeak = Power_Module_powerCrossbarLeak(channel_width, input_switch, output_switch, Cw);
        double switchPower = Power_Module_powerCrossbar(channel_width, input_switch, output_switch, Cw)*channel_width;
        double switchPowerCtrl = Power_Module_powerCrossbarCtrl(channel_width, input_switch, input_switch, Cw);
        double outputPower = Power_Module_powerWireDFF(1, channel_width, 1.0);
        double outputPowerClk = Power_Module_powerWireClk( 1, channel_width,Cw );
        double outputCtrlPower = Power_Module_powerOutputCtrl(channel_width ,Cw);


        //input_switch=5 calculations
        channel_wire_power = channel_wire_power*(hop*tl+tenq*Q/channel_width)*mesh_edge*(mesh_edge*2-2);

        channel_clk_power = channel_clk_power*(2*mesh_edge*mesh_edge);

        channel_DFF_power = channel_DFF_power*(hop*tl+tenq*Q/channel_width)*mesh_edge*(mesh_edge*2-2);

        inputReadPower = inputReadPower*(tenq*Q/channel_width)*mesh_edge*mesh_edge;
        inputWritePower = inputWritePower*(tenq*Q/channel_width)*mesh_edge*mesh_edge;
        switchPower = switchPower*hop*(trc+tva+tsa)*mesh_edge*mesh_edge;
        switchPowerCtrl = switchPowerCtrl*(hop*(trc+tva+tsa)+tenq*Q/channel_width)*mesh_edge*mesh_edge;
        outputPower = outputPower*(hop*tst+tenq*Q/channel_width)*mesh_edge*mesh_edge;
        outputPowerClk = outputPowerClk*(hop*tst)*(2*mesh_edge*mesh_edge);
        outputCtrlPower = outputPowerClk*(hop*tst+tenq*Q/channel_width)*mesh_edge*mesh_edge;

        auto total_energy = channel_wire_power + channel_clk_power + channel_DFF_power + inputReadPower + inputWritePower + switchPower + switchPowerCtrl + outputPower + outputPowerClk + outputCtrlPower;

        return total_energy;




        // double Latency_cycle = calculate_latency(hop, trc, tva, tsa, tst, tl, tenq, Q, channel_width);




        // double total_area_router = switchArea + outputArea;
        // double dynamic_power = switchPower + switchCtrlPower + outputCtrlPower;
        // double total_power = dynamic_power + switchPowerLeak;

        // cout << "Latency Cycle: " << Latency_cycle << endl;
        // cout << "Total Area of Router: " << total_area_router << endl;

        // return total_power;
    }



}