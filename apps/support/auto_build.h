#pragma once
#include <Halide.h>
#include <string>
#include <vector>

void auto_build(Halide::Func f,
                const std::string &name,
                const std::vector<Halide::Argument> &args,
                const Halide::Target &target,
                bool auto_schedule)
{
    Halide::Outputs o;
    std::string suffix = "_ref";
    if (auto_schedule) {
        suffix = "_auto";
    }
    o = o.header(name+".h").object(name+suffix+".o");
    f.compile_to(o, args, name, target, auto_schedule);
}
