#include "Halide.h"

namespace {

class Deinterleave : public Halide::Generator<Deinterleave> {
public:
    ImageParam uvInterleaved{ UInt(8), 2, "uvInterleaved" };

    Func build() {
        Var x, y;

        Func result("result");
        result(x, y) = { uvInterleaved(2 * x, y), uvInterleaved(2 * x + 1, y) };

        // CPU schedule:
        //   Parallelize over scan lines, 4 scanlines per task.
        //   Independently, vectorize over x.
        result
            .parallel(y, 4)
            .vectorize(x, natural_vector_size(UInt(8)));

        // Cope with rotated inputs
        uvInterleaved.set_stride(0, Expr());
        result.specialize(uvInterleaved.stride(0) == 1);
        result.specialize(uvInterleaved.stride(0) == -1);

        return result;
    }
};

Halide::RegisterGenerator<Deinterleave> register_deinterleave{ "deinterleave" };

}  // namespace
