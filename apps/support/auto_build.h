#pragma once
#include <Halide.h>
#include <string>
#include <vector>

void auto_build(Halide::Pipeline p,
                const std::string &name,
                const std::vector<Halide::Argument> &args,
                const Halide::Target &target,
                bool auto_schedule)
{
    Halide::Outputs o;
    std::string suffix = "_ref";
    if (auto_schedule) {
        const char *naive = getenv("HL_AUTO_NAIVE");
        const char *sweep = getenv("HL_AUTO_SWEEP");
        const char *rand = getenv("HL_AUTO_RAND");
        const char *gpu = getenv("HL_AUTO_GPU");
        if (naive && atoi(naive)) {
            suffix = "_naive";
        } else if (sweep && atoi(sweep)){
            suffix = "_sweep";
        } else if (rand && atoi(rand)) {
            suffix = "_rand";
        } else {
            suffix = "_auto";
        }
        if (gpu && atoi(gpu))
            suffix += "_gpu";
    }
    o = o.c_header(name+".h").object(name+suffix+".o");
    p.compile_to(o, args, name, target, auto_schedule);
}

void auto_build(Halide::Func f,
                const std::string &name,
                const std::vector<Halide::Argument> &args,
                const Halide::Target &target,
                bool auto_schedule)
{
    auto_build(Halide::Pipeline(f), name, args, target, auto_schedule);
}
