#include "Halide.h"

class CPlusPlusNameManglingDefineExternGenerator :
    public Halide::Generator<CPlusPlusNameManglingDefineExternGenerator> {
public:
    // Use all the parameter types to make sure mangling works for each of them.
    ImageParam input{UInt(8), 1, "input"};
    Param<int32_t *> ptr{"ptr", 0};
    Param<int32_t const *> const_ptr{"const_ptr", 0};

    Func build() {
        assert(get_target().has_feature(Target::CPlusPlusMangling));
        Var x("x");

        Func g("g");
        g(x) = input(x) + 42;

        Func f("f");

        std::vector<ExternFuncArgument> args;
        args.push_back(g);
        args.push_back(cast<int8_t>(1));
        args.push_back(cast<uint8_t>(2));
        args.push_back(cast<int16_t>(3));
        args.push_back(cast<uint16_t>(4));
        args.push_back(cast<int32_t>(5));
        args.push_back(cast<uint32_t>(6));
        args.push_back(cast<int64_t>(7));
        args.push_back(cast<uint64_t>(8));
        args.push_back(cast<bool>(9 == 9));
        args.push_back(cast<float>(10.0f));
        args.push_back(Expr(11.0));
        args.push_back(ptr);
        args.push_back(const_ptr);
        f.define_extern("HalideTest::cxx_mangling",
                        args, Float(64), 1, true);

        g.compute_root();

        return f;
    }
};

Halide::RegisterGenerator<CPlusPlusNameManglingDefineExternGenerator>
    register_my_gen{"cxx_mangling_define_extern"};
