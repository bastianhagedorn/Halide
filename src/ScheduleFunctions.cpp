#include "ScheduleFunctions.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Substitute.h"
#include "ExprUsesVar.h"
#include "Var.h"
#include "Qualify.h"
#include "IRMutator.h"
#include "Target.h"
#include "Inline.h"

#include "FindCalls.h"
#include "OneToOne.h"
#include "ParallelRVar.h"
#include "Derivative.h"
#include "CodeGen_GPU_Dev.h"

#include <algorithm>

namespace Halide {
namespace Internal {

using std::string;
using std::set;
using std::map;
using std::vector;
using std::pair;
using std::make_pair;

namespace {
// A structure representing a containing LetStmt or For loop. Used in
// build_provide_loop_nest below.
struct Container {
    int dim_idx; // index in the dims list. -1 for let statements.
    string name;
    Expr value;
};
}

// Build a loop nest about a provide node using a schedule
Stmt build_provide_loop_nest(Function f,
                             string prefix,
                             const vector<Expr> &site,
                             const vector<Expr> &values,
                             const Schedule &s,
                             bool is_update) {

    // We'll build it from inside out, starting from a store node,
    // then wrapping it in for loops.

    // Make the (multi-dimensional multi-valued) store node.
    Stmt stmt = Provide::make(f.name(), values, site);

    // The dimensions for which we have a known static size.
    map<string, Expr> known_size_dims;
    // First hunt through the bounds for them.
    for (const Bound &i : s.bounds()) {
        known_size_dims[i.var] = i.extent;
    }
    // Then use any reduction domain.
    const ReductionDomain &rdom = s.reduction_domain();
    if (rdom.defined()) {
        for (const ReductionVariable &i : rdom.domain()) {
            known_size_dims[i.var] = i.extent;
        }
    }

    vector<Split> splits = s.splits();

    // Rebalance the split tree to make the outermost split first.
    for (size_t i = 0; i < splits.size(); i++) {
        for (size_t j = i+1; j < splits.size(); j++) {

            Split &first = splits[i];
            Split &second = splits[j];
            if (first.outer == second.old_var) {
                internal_assert(!second.is_rename())
                    << "Rename of derived variable found in splits list. This should never happen.";

                if (first.is_rename()) {
                    // Given a rename:
                    // X -> Y
                    // And a split:
                    // Y -> f * Z + W
                    // Coalesce into:
                    // X -> f * Z + W
                    second.old_var = first.old_var;
                    // Drop first entirely
                    for (size_t k = i; k < splits.size()-1; k++) {
                        splits[k] = splits[k+1];
                    }
                    splits.pop_back();
                    // Start processing this split from scratch,
                    // because we just clobbered it.
                    j = i+1;
                } else {
                    // Given two splits:
                    // X  ->  a * Xo  + Xi
                    // (splits stuff other than Xo, including Xi)
                    // Xo ->  b * Xoo + Xoi

                    // Re-write to:
                    // X  -> ab * Xoo + s0
                    // s0 ->  a * Xoi + Xi
                    // (splits on stuff other than Xo, including Xi)

                    // The name Xo went away, because it was legal for it to
                    // be X before, but not after.

                    first.exact |= second.exact;
                    second.exact = first.exact;
                    second.old_var = unique_name('s');
                    first.outer   = second.outer;
                    second.outer  = second.inner;
                    second.inner  = first.inner;
                    first.inner   = second.old_var;
                    Expr f = simplify(first.factor * second.factor);
                    second.factor = first.factor;
                    first.factor  = f;
                    // Push the second split back to just after the first
                    for (size_t k = j; k > i+1; k--) {
                        std::swap(splits[k], splits[k-1]);
                    }
                }
            }
        }
    }

    Dim innermost_non_trivial_loop;
    for (const Dim &d : s.dims()) {
        if (d.for_type != ForType::Vectorized &&
            d.for_type != ForType::Unrolled) {
            innermost_non_trivial_loop = d;
            break;
        }
    }

    // Define the function args in terms of the loop variables using the splits
    map<string, pair<string, Expr>> base_values;
    for (const Split &split : splits) {
        Expr outer = Variable::make(Int(32), prefix + split.outer);
        if (split.is_split()) {
            Expr inner = Variable::make(Int(32), prefix + split.inner);
            Expr old_max = Variable::make(Int(32), prefix + split.old_var + ".loop_max");
            Expr old_min = Variable::make(Int(32), prefix + split.old_var + ".loop_min");

            known_size_dims[split.inner] = split.factor;

            Expr base = outer * split.factor + old_min;

            map<string, Expr>::iterator iter = known_size_dims.find(split.old_var);
            if ((iter != known_size_dims.end()) &&
                is_zero(simplify(iter->second % split.factor))) {

                // We have proved that the split factor divides the
                // old extent. No need to adjust the base.
                known_size_dims[split.outer] = iter->second / split.factor;
            } else if (split.exact) {
                // It's an exact split but we failed to prove that the
                // extent divides the factor. This is a problem.
                user_error << "When splitting " << split.old_var << " into "
                           << split.outer << " and " << split.inner << ", "
                           << "could not prove the split factor (" << split.factor << ") "
                           << "divides the extent of " << split.old_var
                           << " (" << iter->second << "). This is required when "
                           << "the split originates from an RVar.\n";
            } else if (!is_update  && !split.partial) {
                // Adjust the base downwards to not compute off the
                // end of the realization.

                // Only mark the base as likely (triggering a loop
                // partition) if the outer var is the innermost
                // non-trivial loop and it's a serial loop. This
                // usually is due to an unroll or vectorize call.
                if (split.outer == innermost_non_trivial_loop.var &&
                    innermost_non_trivial_loop.for_type == ForType::Serial) {
                    base = likely(base);
                }

                base = Min::make(base, old_max + (1 - split.factor));
            }

            string base_name = prefix + split.inner + ".base";
            Expr base_var = Variable::make(Int(32), base_name);
            // Substitute in the new expression for the split variable ...
            stmt = substitute(prefix + split.old_var, base_var + inner, stmt);
            // ... but also define it as a let for the benefit of bounds inference.
            stmt = LetStmt::make(prefix + split.old_var, base_var + inner, stmt);
            stmt = LetStmt::make(base_name, base, stmt);

        } else if (split.is_fuse()) {
            // Define the inner and outer in terms of the fused var
            Expr fused = Variable::make(Int(32), prefix + split.old_var);
            Expr inner_min = Variable::make(Int(32), prefix + split.inner + ".loop_min");
            Expr outer_min = Variable::make(Int(32), prefix + split.outer + ".loop_min");
            Expr inner_extent = Variable::make(Int(32), prefix + split.inner + ".loop_extent");

            // If the inner extent is zero, the loop will never be
            // entered, but the bounds expressions lifted out might
            // contain divides or mods by zero. In the cases where
            // simplification of inner and outer matter, inner_extent
            // is a constant, so the max will simplify away.
            Expr factor = max(inner_extent, 1);
            Expr inner = fused % factor + inner_min;
            Expr outer = fused / factor + outer_min;

            stmt = substitute(prefix + split.inner, inner, stmt);
            stmt = substitute(prefix + split.outer, outer, stmt);
            stmt = LetStmt::make(prefix + split.inner, inner, stmt);
            stmt = LetStmt::make(prefix + split.outer, outer, stmt);

            // Maintain the known size of the fused dim if
            // possible. This is important for possible later splits.
            map<string, Expr>::iterator inner_dim = known_size_dims.find(split.inner);
            map<string, Expr>::iterator outer_dim = known_size_dims.find(split.outer);
            if (inner_dim != known_size_dims.end() &&
                outer_dim != known_size_dims.end()) {
                known_size_dims[split.old_var] = inner_dim->second*outer_dim->second;
            }

        } else {
            stmt = substitute(prefix + split.old_var, outer, stmt);
            stmt = LetStmt::make(prefix + split.old_var, outer, stmt);
        }
    }

    // All containing lets and fors. Outermost first.
    vector<Container> nest;

    // Put the desired loop nest into the containers vector.
    for (int i = (int)s.dims().size() - 1; i >= 0; i--) {
        const Dim &dim = s.dims()[i];
        Container c = {i, prefix + dim.var, Expr()};
        nest.push_back(c);
    }

    // Strip off the lets into the containers vector.
    while (const LetStmt *let = stmt.as<LetStmt>()) {
        Container c = {-1, let->name, let->value};
        nest.push_back(c);
        stmt = let->body;
    }

    // Resort the containers vector so that lets are as far outwards
    // as possible. Use reverse insertion sort. Start at the first letstmt.
    for (int i = (int)s.dims().size(); i < (int)nest.size(); i++) {
        // Only push up LetStmts.
        internal_assert(nest[i].value.defined());

        for (int j = i-1; j >= 0; j--) {
            // Try to push it up by one.
            internal_assert(nest[j+1].value.defined());
            if (!expr_uses_var(nest[j+1].value, nest[j].name)) {
                std::swap(nest[j+1], nest[j]);
            } else {
                break;
            }
        }
    }

    // Rewrap the statement in the containing lets and fors.
    for (int i = (int)nest.size() - 1; i >= 0; i--) {
        if (nest[i].value.defined()) {
            stmt = LetStmt::make(nest[i].name, nest[i].value, stmt);
        } else {
            const Dim &dim = s.dims()[nest[i].dim_idx];
            Expr min = Variable::make(Int(32), nest[i].name + ".loop_min");
            Expr extent = Variable::make(Int(32), nest[i].name + ".loop_extent");
            stmt = For::make(nest[i].name, min, extent, dim.for_type, dim.device_api, stmt);
        }
    }

    // Define the bounds on the split dimensions using the bounds
    // on the function args
    for (size_t i = splits.size(); i > 0; i--) {
        const Split &split = splits[i-1];
        Expr old_var_extent = Variable::make(Int(32), prefix + split.old_var + ".loop_extent");
        Expr old_var_max = Variable::make(Int(32), prefix + split.old_var + ".loop_max");
        Expr old_var_min = Variable::make(Int(32), prefix + split.old_var + ".loop_min");
        if (split.is_split()) {
            Expr inner_extent;
            if (split.partial)
                inner_extent = Min::make(likely(split.factor), old_var_max + 1);
            else
                inner_extent = split.factor;
            Expr outer_extent = (old_var_max - old_var_min + split.factor)/split.factor;

            stmt = LetStmt::make(prefix + split.inner + ".loop_min", 0, stmt);
            stmt = LetStmt::make(prefix + split.inner + ".loop_max", inner_extent-1, stmt);
            stmt = LetStmt::make(prefix + split.inner + ".loop_extent", inner_extent, stmt);
            stmt = LetStmt::make(prefix + split.outer + ".loop_min", 0, stmt);
            stmt = LetStmt::make(prefix + split.outer + ".loop_max", outer_extent-1, stmt);
            stmt = LetStmt::make(prefix + split.outer + ".loop_extent", outer_extent, stmt);
        } else if (split.is_fuse()) {
            // Define bounds on the fused var using the bounds on the inner and outer
            Expr inner_extent = Variable::make(Int(32), prefix + split.inner + ".loop_extent");
            Expr outer_extent = Variable::make(Int(32), prefix + split.outer + ".loop_extent");
            Expr fused_extent = inner_extent * outer_extent;
            stmt = LetStmt::make(prefix + split.old_var + ".loop_min", 0, stmt);
            stmt = LetStmt::make(prefix + split.old_var + ".loop_max", fused_extent - 1, stmt);
            stmt = LetStmt::make(prefix + split.old_var + ".loop_extent", fused_extent, stmt);
        } else {
            // rename
            stmt = LetStmt::make(prefix + split.outer + ".loop_min", old_var_min, stmt);
            stmt = LetStmt::make(prefix + split.outer + ".loop_max", old_var_max, stmt);
            stmt = LetStmt::make(prefix + split.outer + ".loop_extent", old_var_extent, stmt);
        }
    }

    // Define the bounds on the outermost dummy dimension.
    {
        string o = prefix + Var::outermost().name();
        stmt = LetStmt::make(o + ".loop_min", 0, stmt);
        stmt = LetStmt::make(o + ".loop_max", 1, stmt);
        stmt = LetStmt::make(o + ".loop_extent", 1, stmt);
    }

    // Define the loop mins and extents in terms of the mins and maxs produced by bounds inference
    for (const std::string &i : f.args()) {
        string var = prefix + i;
        Expr max = Variable::make(Int(32), var + ".max");
        Expr min = Variable::make(Int(32), var + ".min"); // Inject instance name here? (compute instance names during lowering)
        stmt = LetStmt::make(var + ".loop_extent",
                             (max + 1) - min,
                             stmt);
        stmt = LetStmt::make(var + ".loop_min", min, stmt);
        stmt = LetStmt::make(var + ".loop_max", max, stmt);
    }

    // Make any specialized copies
    for (size_t i = s.specializations().size(); i > 0; i--) {
        Expr c = s.specializations()[i-1].condition;
        Schedule sched = s.specializations()[i-1].schedule;
        const EQ *eq = c.as<EQ>();
        const Variable *var = eq ? eq->a.as<Variable>() : c.as<Variable>();

        Stmt then_case =
            build_provide_loop_nest(f, prefix, site, values, sched, is_update);

        if (var && eq) {
            then_case = simplify_exprs(substitute(var->name, eq->b, then_case));
            Stmt else_case = stmt;
            if (eq->b.type().is_bool()) {
                else_case = simplify_exprs(substitute(var->name, !eq->b, else_case));
            }
            stmt = IfThenElse::make(c, then_case, else_case);
        } else if (var) {
            then_case = simplify_exprs(substitute(var->name, const_true(), then_case));
            Stmt else_case = simplify_exprs(substitute(var->name, const_false(), stmt));
            stmt = IfThenElse::make(c, then_case, else_case);
        } else {
            stmt = IfThenElse::make(c, then_case, stmt);
        }
    }

    return stmt;
}

// Turn a function into a loop nest that computes it. It will
// refer to external vars of the form function_name.arg_name.min
// and function_name.arg_name.extent to define the bounds over
// which it should be realized. It will compute at least those
// bounds (depending on splits, it may compute more). This loop
// won't do any allocation.
Stmt build_produce(Function f) {

    if (f.has_extern_definition()) {
        // Call the external function

        // Build an argument list
        vector<Expr> extern_call_args;
        const vector<ExternFuncArgument> &args = f.extern_arguments();

        const string &extern_name = f.extern_function_name();

        vector<pair<string, Expr>> lets;

        // Iterate through all of the input args to the extern
        // function building a suitable argument list for the
        // extern function call.
        for (const ExternFuncArgument &arg : args) {
            if (arg.is_expr()) {
                extern_call_args.push_back(arg.expr);
            } else if (arg.is_func()) {
                Function input(arg.func);
                for (int k = 0; k < input.outputs(); k++) {
                    string buf_name = input.name();
                    if (input.outputs() > 1) {
                        buf_name += "." + std::to_string(k);
                    }
                    buf_name += ".buffer";
                    Expr buffer = Variable::make(Handle(), buf_name);
                    extern_call_args.push_back(buffer);
                }
            } else if (arg.is_buffer()) {
                Buffer b = arg.buffer;
                Parameter p(b.type(), true, b.dimensions(), b.name());
                p.set_buffer(b);
                Expr buf = Variable::make(Handle(), b.name() + ".buffer", p);
                extern_call_args.push_back(buf);
            } else if (arg.is_image_param()) {
                Parameter p = arg.image_param;
                Expr buf = Variable::make(Handle(), p.name() + ".buffer", p);
                extern_call_args.push_back(buf);
            } else {
                internal_error << "Bad ExternFuncArgument type\n";
            }
        }

        // Grab the buffer_ts representing the output. If the store
        // level matches the compute level, then we can use the ones
        // already injected by allocation bounds inference. If it's
        // the output to the pipeline then it will similarly be in the
        // symbol table.
        if (f.schedule().store_level() == f.schedule().compute_level()) {
            for (int j = 0; j < f.outputs(); j++) {
                string buf_name = f.name();
                if (f.outputs() > 1) {
                    buf_name += "." + std::to_string(j);
                }
                buf_name += ".buffer";
                Expr buffer = Variable::make(Handle(), buf_name);
                extern_call_args.push_back(buffer);
            }
        } else {
            // Store level doesn't match compute level. Make an output
            // buffer just for this subregion.
            string stride_name = f.name();
            if (f.outputs() > 1) {
                stride_name += ".0";
            }
            string stage_name = f.name() + ".s0.";
            for (int j = 0; j < f.outputs(); j++) {

                vector<Expr> buffer_args(2);

                vector<Expr> top_left;
                for (int k = 0; k < f.dimensions(); k++) {
                    string var = stage_name + f.args()[k];
                    top_left.push_back(Variable::make(Int(32), var + ".min"));
                }
                Expr host_ptr = Call::make(f, top_left, j);
                host_ptr = Call::make(Handle(), Call::address_of, {host_ptr}, Call::Intrinsic);

                buffer_args[0] = host_ptr;
                buffer_args[1] = make_zero(f.output_types()[j]);
                for (int k = 0; k < f.dimensions(); k++) {
                    string var = stage_name + f.args()[k];
                    Expr min = Variable::make(Int(32), var + ".min");
                    Expr max = Variable::make(Int(32), var + ".max");
                    Expr stride = Variable::make(Int(32), stride_name + ".stride." + std::to_string(k));
                    buffer_args.push_back(min);
                    buffer_args.push_back(max - min + 1);
                    buffer_args.push_back(stride);
                }

                Expr output_buffer_t = Call::make(Handle(), Call::create_buffer_t,
                                                  buffer_args, Call::Intrinsic);

                string buf_name = f.name() + "." + std::to_string(j) + ".tmp_buffer";
                extern_call_args.push_back(Variable::make(Handle(), buf_name));
                lets.push_back(make_pair(buf_name, output_buffer_t));
            }
        }

        // Make the extern call
        Expr e = Call::make(Int(32), extern_name,
                            extern_call_args, Call::Extern);
        string result_name = unique_name('t');
        Expr result = Variable::make(Int(32), result_name);
        // Check if it succeeded
        Expr error = Call::make(Int(32), "halide_error_extern_stage_failed",
                                {extern_name, result}, Call::Extern);
        Stmt check = AssertStmt::make(EQ::make(result, 0), error);
        check = LetStmt::make(result_name, e, check);

        for (size_t i = 0; i < lets.size(); i++) {
            check = LetStmt::make(lets[i].first, lets[i].second, check);
        }

        return check;
    } else {

        string prefix = f.name() + ".s0.";

        // Compute the site to store to as the function args
        vector<Expr> site;

        vector<Expr> values(f.values().size());
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = qualify(prefix, f.values()[i]);
        }

        for (size_t i = 0; i < f.args().size(); i++) {
            site.push_back(Variable::make(Int(32), prefix + f.args()[i]));
        }

        return build_provide_loop_nest(f, prefix, site, values, f.schedule(), false);
    }
}

// Build the loop nests that update a function (assuming it's a reduction).
vector<Stmt> build_update(Function f) {

    vector<Stmt> updates;

    for (size_t i = 0; i < f.updates().size(); i++) {
        UpdateDefinition r = f.updates()[i];

        string prefix = f.name() + ".s" + std::to_string(i+1) + ".";

        vector<Expr> site(r.args.size());
        vector<Expr> values(r.values.size());
        for (size_t i = 0; i < values.size(); i++) {
            Expr v = r.values[i];
            v = qualify(prefix, v);
            values[i] = v;
        }

        for (size_t i = 0; i < r.args.size(); i++) {
            Expr s = r.args[i];
            s = qualify(prefix, s);
            site[i] = s;
            debug(2) << "Update site " << i << " = " << s << "\n";
        }

        Stmt loop = build_provide_loop_nest(f, prefix, site, values, r.schedule, true);

        // Now define the bounds on the reduction domain
        if (r.domain.defined()) {
            const vector<ReductionVariable> &dom = r.domain.domain();
            for (size_t i = 0; i < dom.size(); i++) {
                string p = prefix + dom[i].var;
                Expr rmin = Variable::make(Int(32), p + ".min");
                Expr rmax = Variable::make(Int(32), p + ".max");
                loop = LetStmt::make(p + ".loop_min", rmin, loop);
                loop = LetStmt::make(p + ".loop_max", rmax, loop);
                loop = LetStmt::make(p + ".loop_extent", rmax - rmin + 1, loop);
            }
        }

        updates.push_back(loop);
    }

    return updates;
}

pair<Stmt, Stmt> build_production(Function func) {
    Stmt produce = build_produce(func);
    vector<Stmt> updates = build_update(func);

    // Build it from the last stage backwards.
    Stmt merged_updates;
    for (size_t s = updates.size(); s > 0; s--) {
        merged_updates = Block::make(updates[s-1], merged_updates);
    }
    return make_pair(produce, merged_updates);
}

// A schedule may include explicit bounds on some dimension. This
// injects assertions that check that those bounds are sufficiently
// large to cover the inferred bounds required.
Stmt inject_explicit_bounds(Stmt body, Function func) {
    const Schedule &s = func.schedule();
    for (size_t stage = 0; stage <= func.updates().size(); stage++) {
        for (size_t i = 0; i < s.bounds().size(); i++) {
            Bound b = s.bounds()[i];
            Expr max_val = (b.extent + b.min) - 1;
            Expr min_val = b.min;
            string prefix = func.name() + ".s" + std::to_string(stage) + "." + b.var;
            string min_name = prefix + ".min_unbounded";
            string max_name = prefix + ".max_unbounded";
            Expr min_var = Variable::make(Int(32), min_name);
            Expr max_var = Variable::make(Int(32), max_name);
            Expr check = (min_val <= min_var) && (max_val >= max_var);
            Expr error_msg = Call::make(Int(32), "halide_error_explicit_bounds_too_small",
                                        {b.var, func.name(), min_val, max_val, min_var, max_var},
                                        Call::Extern);

            // bounds inference has already respected these values for us
            //body = LetStmt::make(prefix + ".min", min_val, body);
            //body = LetStmt::make(prefix + ".max", max_val, body);

            body = Block::make(AssertStmt::make(check, error_msg), body);
        }
    }

    return body;
}

class IsUsedInStmt : public IRVisitor {
    string func;

    using IRVisitor::visit;

    void visit(const Call *op) {
        IRVisitor::visit(op);
        if (op->name == func) result = true;
    }

    // A reference to the function's buffers counts as a use
    void visit(const Variable *op) {
        if (op->type == Handle() &&
            starts_with(op->name, func + ".") &&
            ends_with(op->name, ".buffer")) {
            result = true;
        }
    }

public:
    bool result;
    IsUsedInStmt(Function f) : func(f.name()), result(false) {
    }

};

bool function_is_used_in_stmt(Function f, Stmt s) {
    IsUsedInStmt is_called(f);
    s.accept(&is_called);
    return is_called.result;
}

// Inject the allocation and realization of a function into an
// existing loop nest using its schedule
class InjectRealization : public IRMutator {
public:
    const Function &func;
    bool is_output, found_store_level, found_compute_level, inject_asserts;

    InjectRealization(const Function &f, bool o, bool asserts) :
        func(f), is_output(o),
        found_store_level(false), found_compute_level(false),
        inject_asserts(asserts) {}

private:

    string producing;

    Stmt build_pipeline(Stmt s) {
        pair<Stmt, Stmt> realization = build_production(func);

        return ProducerConsumer::make(func.name(), realization.first, realization.second, s);
    }

    Stmt build_realize(Stmt s) {
        if (!is_output) {
            Region bounds;
            string name = func.name();
            for (int i = 0; i < func.dimensions(); i++) {
                string arg = func.args()[i];
                Expr min = Variable::make(Int(32), name + "." + arg + ".min_realized");
                Expr extent = Variable::make(Int(32), name + "." + arg + ".extent_realized");
                bounds.push_back(Range(min, extent));
            }

            s = Realize::make(name, func.output_types(), bounds, const_true(), s);
        }

        // This is also the point at which we inject explicit bounds
        // for this realization.
        if (inject_asserts) {
            return inject_explicit_bounds(s, func);
        } else {
            return s;
        }
    }

    using IRMutator::visit;

    void visit(const ProducerConsumer *op) {
        string old = producing;
        producing = op->name;
        Stmt produce = mutate(op->produce);
        Stmt update;
        if (op->update.defined()) {
            update = mutate(op->update);
        }
        producing = old;
        Stmt consume = mutate(op->consume);

        if (produce.same_as(op->produce) &&
            update.same_as(op->update) &&
            consume.same_as(op->consume)) {
            stmt = op;
        } else {
            stmt = ProducerConsumer::make(op->name, produce, update, consume);
        }
    }

    void visit(const For *for_loop) {
        debug(3) << "InjectRealization of " << func.name() << " entering for loop over " << for_loop->name << "\n";
        const LoopLevel &compute_level = func.schedule().compute_level();
        const LoopLevel &store_level = func.schedule().store_level();

        Stmt body = for_loop->body;

        // Dig through any let statements
        vector<pair<string, Expr>> lets;
        while (const LetStmt *l = body.as<LetStmt>()) {
            lets.push_back(make_pair(l->name, l->value));
            body = l->body;
        }

        // Can't schedule extern things inside a vector for loop
        if (func.has_extern_definition() &&
            func.schedule().compute_level().is_inline() &&
            for_loop->for_type == ForType::Vectorized &&
            function_is_used_in_stmt(func, for_loop)) {

            // If we're trying to inline an extern function, schedule it here and bail out
            debug(2) << "Injecting realization of " << func.name() << " around node " << Stmt(for_loop) << "\n";
            stmt = build_realize(build_pipeline(for_loop));
            found_store_level = found_compute_level = true;
            return;
        }

        body = mutate(body);

        if (compute_level.match(for_loop->name)) {
            debug(3) << "Found compute level\n";
            if (function_is_used_in_stmt(func, body) || is_output) {
                body = build_pipeline(body);
            }
            found_compute_level = true;
        }

        if (store_level.match(for_loop->name)) {
            debug(3) << "Found store level\n";
            internal_assert(found_compute_level)
                << "The compute loop level was not found within the store loop level!\n";

            if (function_is_used_in_stmt(func, body) || is_output) {
                body = build_realize(body);
            }

            found_store_level = true;
        }

        // Reinstate the let statements
        for (size_t i = lets.size(); i > 0; i--) {
            body = LetStmt::make(lets[i - 1].first, lets[i - 1].second, body);
        }

        if (body.same_as(for_loop->body)) {
            stmt = for_loop;
        } else {
            stmt = For::make(for_loop->name,
                             for_loop->min,
                             for_loop->extent,
                             for_loop->for_type,
                             for_loop->device_api,
                             body);
        }
    }

    // If we're an inline update or extern, we may need to inject a realization here
    virtual void visit(const Provide *op) {
        if (op->name != func.name() &&
            !func.is_pure() &&
            func.schedule().compute_level().is_inline() &&
            function_is_used_in_stmt(func, op)) {

            // Prefix all calls to func in op
            stmt = build_realize(build_pipeline(op));
            found_store_level = found_compute_level = true;
        } else {
            stmt = op;
        }
    }
};


class ComputeLegalSchedules : public IRVisitor {
public:
    struct Site {
        bool is_parallel;
        LoopLevel loop_level;
    };
    vector<Site> sites_allowed;

    ComputeLegalSchedules(Function f) : func(f), found(false) {}

private:
    using IRVisitor::visit;

    vector<Site> sites;
    Function func;
    bool found;

    void visit(const For *f) {
        f->min.accept(this);
        f->extent.accept(this);
        size_t first_dot = f->name.find('.');
        size_t last_dot = f->name.rfind('.');
        internal_assert(first_dot != string::npos && last_dot != string::npos);
        string func = f->name.substr(0, first_dot);
        string var = f->name.substr(last_dot + 1);
        Site s = {f->for_type == ForType::Parallel ||
                  f->for_type == ForType::Vectorized,
                  LoopLevel(func, var)};
        sites.push_back(s);
        f->body.accept(this);
        sites.pop_back();
    }

    void register_use() {
        if (!found) {
            found = true;
            sites_allowed = sites;
        } else {
            vector<Site> common_sites;

            // Take the common sites between sites and sites_allowed
            for (const Site &s1 : sites) {
                for (const Site &s2 : sites_allowed) {
                    if (s1.loop_level.match(s2.loop_level)) {
                        common_sites.push_back(s1);
                        break;
                    }
                }
            }

            sites_allowed.swap(common_sites);
        }
    }

    void visit(const Call *c) {
        IRVisitor::visit(c);

        if (c->name == func.name()) {
            register_use();
        }
    }

    void visit(const Variable *v) {
        if (v->type == Handle() &&
            starts_with(v->name, func.name() + ".") &&
            ends_with(v->name, ".buffer")) {
            register_use();
        }
    }
};

string schedule_to_source(Function f,
                          LoopLevel store_at,
                          LoopLevel compute_at) {
    std::ostringstream ss;
    ss << f.name();
    if (compute_at.is_inline()) {
        ss << ".compute_inline()";
    } else {
        string store_var_name = store_at.var;
        string compute_var_name = compute_at.var;
        if (store_var_name == Var::outermost().name()) {
            store_var_name = "Var::outermost()";
        }
        if (compute_var_name == Var::outermost().name()) {
            compute_var_name = "Var::outermost()";
        }
        if (!store_at.match(compute_at)) {
            if (store_at.is_root()) {
                ss << ".store_root()";
            } else {
                ss << ".store_at(" << store_at.func << ", " << store_var_name << ")";
            }
        }
        if (compute_at.is_root()) {
            ss << ".compute_root()";
        } else {
            ss << ".compute_at(" << compute_at.func << ", " << compute_var_name << ")";
        }
    }
    ss << ";";
    return ss.str();
}

class StmtUsesFunc : public IRVisitor {
    using IRVisitor::visit;
    string func;
    void visit(const Call *op) {
        if (op->name == func) {
            result = true;
        }
        IRVisitor::visit(op);
    }
public:
    bool result = false;
    StmtUsesFunc(string f) : func(f) {}
};

class PrintUsesOfFunc : public IRVisitor {
    using IRVisitor::visit;

    int indent = 1;
    string func, caller;
    bool last_print_was_ellipsis = false;
    std::ostream &stream;

    void do_indent() {
        for (int i = 0; i < indent; i++) {
            stream << "  ";
        }
    }

    void visit(const For *op) {
        if (ends_with(op->name, Var::outermost().name()) ||
            ends_with(op->name, LoopLevel::root().var)) {
            IRVisitor::visit(op);
        } else {

            int old_indent = indent;

            StmtUsesFunc uses(func);
            op->body.accept(&uses);
            if (!uses.result) {
                if (!last_print_was_ellipsis) {
                    do_indent();
                    stream << "...\n";
                    last_print_was_ellipsis = true;
                }
            } else {
                do_indent();
                stream << "for " << op->name << ":\n";
                last_print_was_ellipsis = false;
                indent++;
            }

            IRVisitor::visit(op);
            indent = old_indent;
        }
    }

    void visit(const ProducerConsumer *op) {
        string old_caller = caller;
        caller = op->name;
        op->produce.accept(this);
        if (op->update.defined()) {
            op->update.accept(this);
        }
        caller = old_caller;
        op->consume.accept(this);
    }

    void visit(const Call *op) {
        if (op->name == func) {
            do_indent();
            stream << caller << " uses " << func << "\n";
            last_print_was_ellipsis = false;
        } else {
            IRVisitor::visit(op);
        }
    }

public:
    PrintUsesOfFunc(string f, std::ostream &s) : func(f), stream(s) {}
};

void validate_schedule(Function f, Stmt s, bool is_output) {

    // If f is extern, check that none of its inputs are scheduled inline.
    if (f.has_extern_definition()) {
        for (const ExternFuncArgument &arg : f.extern_arguments()) {
            if (arg.is_func()) {
                Function g(arg.func);
                if (g.schedule().compute_level().is_inline()) {
                    user_error
                        << "Func " << g.name() << " cannot be scheduled to be computed inline, "
                        << "because it is used in the externally-computed function " << f.name() << "\n";
                }
            }
        }
    }

    // Emit a warning if only some of the steps have been scheduled.
    bool any_scheduled = f.schedule().touched();
    for (const UpdateDefinition &r : f.updates()) {
        any_scheduled = any_scheduled || r.schedule.touched();
    }
    if (any_scheduled) {
        for (size_t i = 0; i < f.updates().size(); i++) {
            const UpdateDefinition &r = f.updates()[i];
            if (!r.schedule.touched()) {
                std::cerr << "Warning: Update step " << i
                          << " of function " << f.name()
                          << " has not been scheduled, even though some other"
                          << " steps have been. You may have forgotten to"
                          << " schedule it. If this was intentional, call "
                          << f.name() << ".update(" << i << ") to suppress"
                          << " this warning.\n";
            }
        }
    }

    LoopLevel store_at = f.schedule().store_level();
    LoopLevel compute_at = f.schedule().compute_level();

    // Outputs must be compute_root and store_root. They're really
    // store_in_user_code, but store_root is close enough.
    if (is_output) {
        if (store_at.is_root() && compute_at.is_root()) {
            return;
        } else {
            user_error << "Func " << f.name() << " is the output, so must"
                       << " be scheduled compute_root (which is the default).\n";
        }
    }

    // Inlining is always allowed
    if (store_at.is_inline() && compute_at.is_inline()) {
        return;
    }

    // Otherwise inspect the uses to see what's ok.
    ComputeLegalSchedules legal(f);
    s.accept(&legal);

    bool store_at_ok = false, compute_at_ok = false;
    const vector<ComputeLegalSchedules::Site> &sites = legal.sites_allowed;
    size_t store_idx = 0, compute_idx = 0;
    for (size_t i = 0; i < sites.size(); i++) {
        if (sites[i].loop_level.match(store_at)) {
            store_at_ok = true;
            store_idx = i;
        }
        if (sites[i].loop_level.match(compute_at)) {
            compute_at_ok = store_at_ok;
            compute_idx = i;
        }
    }

    // Check there isn't a parallel loop between the compute_at and the store_at
    std::ostringstream err;

    if (store_at_ok && compute_at_ok) {
        for (size_t i = store_idx + 1; i <= compute_idx; i++) {
            if (sites[i].is_parallel) {
                err << "Func \"" << f.name()
                    << "\" is stored outside the parallel loop over "
                    << sites[i].loop_level.func << "." << sites[i].loop_level.var
                    << " but computed within it. This is a potential race condition.\n";
                store_at_ok = compute_at_ok = false;
            }
        }
    }

    if (!store_at_ok || !compute_at_ok) {
        err << "Func \"" << f.name() << "\" is computed at the following invalid location:\n"
            << "  " << schedule_to_source(f, store_at, compute_at) << "\n"
            << "Legal locations for this function are:\n";
        for (size_t i = 0; i < sites.size(); i++) {
            err << "  " << schedule_to_source(f, sites[i].loop_level, sites[i].loop_level) << "\n";
        }
        err << "\"" << f.name() << "\" is used in the following places:\n";
        PrintUsesOfFunc printer(f.name(), err);
        s.accept(&printer);

        user_error << err.str();
    }
}

class RemoveLoopsOverOutermost : public IRMutator {
    using IRMutator::visit;

    void visit(const For *op) {
        if (ends_with(op->name, ".__outermost")) {
            stmt = mutate(op->body);
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const Variable *op) {
        if (ends_with(op->name, ".__outermost.loop_extent")) {
            expr = 1;
        } else if (ends_with(op->name, ".__outermost.loop_min")) {
            expr = 0;
        } else if (ends_with(op->name, ".__outermost.loop_max")) {
            expr = 1;
        } else {
            expr = op;
        }
    }

    void visit(const LetStmt *op) {
        if (ends_with(op->name, ".__outermost.loop_extent") ||
            ends_with(op->name, ".__outermost.loop_min") ||
            ends_with(op->name, ".__outermost.loop_max")) {
            stmt = mutate(op->body);
        } else {
            IRMutator::visit(op);
        }
    }
};


class PropagateLoopDeviceAPI : public IRMutator {
    using IRMutator::visit;

    DeviceAPI for_device;

    void visit(const For *op) {
        DeviceAPI save_device = for_device;
        for_device = (op->device_api == DeviceAPI::Parent) ? for_device : op->device_api;

        Expr min = mutate(op->min);
        Expr extent = mutate(op->extent);
        Stmt body = mutate(op->body);

        if (min.same_as(op->min) &&
            extent.same_as(op->extent) &&
            body.same_as(op->body) &&
            for_device == op->device_api) {
            stmt = op;
        } else {
            stmt = For::make(op->name, min, extent, op->for_type, for_device, body);
        }

        for_device = save_device;
    }

public:
    PropagateLoopDeviceAPI() : for_device(DeviceAPI::Host) {
    }
};

Stmt schedule_functions(const vector<Function> &outputs,
                        const vector<string> &order,
                        const map<string, Function> &env,
                        bool &any_memoized,
                        bool inject_asserts) {

    string root_var = LoopLevel::root().func + "." + LoopLevel::root().var;
    Stmt s = For::make(root_var, 0, 1, ForType::Serial, DeviceAPI::Host, Evaluate::make(0));

    any_memoized = false;

    for (size_t i = order.size(); i > 0; i--) {
        Function f = env.find(order[i-1])->second;

        bool is_output = false;
        for (Function o : outputs) {
            is_output |= o.same_as(f);
        }

        validate_schedule(f, s, is_output);

        if (f.has_pure_definition() &&
            !f.has_update_definition() &&
            f.schedule().compute_level().is_inline()) {
            debug(1) << "Inlining " << order[i-1] << '\n';
            s = inline_function(s, f);
        } else {
            debug(1) << "Injecting realization of " << order[i-1] << '\n';
            InjectRealization injector(f, is_output, inject_asserts);
            s = injector.mutate(s);
            internal_assert(injector.found_store_level && injector.found_compute_level);
        }
        any_memoized = any_memoized || f.schedule().memoized();
        debug(2) << s << '\n';
    }

    // We can remove the loop over root now
    const For *root_loop = s.as<For>();
    internal_assert(root_loop);
    s = root_loop->body;

    // We can also remove all the loops over __outermost now.
    s = RemoveLoopsOverOutermost().mutate(s);

    // And finally we can propagate loop device types.
    s = PropagateLoopDeviceAPI().mutate(s);

    return s;

}

/* Find all the internal halide calls in an expr */
class FindCallArgs : public IRVisitor {
    public:
        map<string, std::vector<const Call*> > calls;
        vector<vector<Expr>> load_args;

        using IRVisitor::visit;

        void visit(const Call *call) {
            // See if images need to be included
            if (call->call_type == Call::Halide) {
                calls[call->func.name()].push_back(call);
                load_args.push_back(call->args);
            }
            for (size_t i = 0; (i < call->args.size()); i++)
                call->args[i].accept(this);
        }
};

/* Visitor for computing the cost of a single value of a function*/
class ExprCostEarly : public IRVisitor {
    public:
        int ops;
        int loads;

        ExprCostEarly() {
            ops = 0; loads = 0;
        }

        using IRVisitor::visit;

        void visit(const IntImm *) {}
        void visit(const UIntImm *) {}
        void visit(const FloatImm *) {}
        void visit(const StringImm *) {}
        void visit(const Cast *) { ops+=1; }
        void visit(const Variable *) {}

        template<typename T>
            void visit_binary_operator(const T *op) {
                op->a.accept(this);
                op->b.accept(this);
                ops += 1;
            }

        void visit(const Add *op) {visit_binary_operator(op);}
        void visit(const Sub *op) {visit_binary_operator(op);}
        void visit(const Mul *op) {visit_binary_operator(op);}
        void visit(const Div *op) {visit_binary_operator(op);}
        void visit(const Mod *op) {visit_binary_operator(op);}
        void visit(const Min *op) {visit_binary_operator(op);}
        void visit(const Max *op) {visit_binary_operator(op);}
        void visit(const EQ *op) {visit_binary_operator(op);}
        void visit(const NE *op) {visit_binary_operator(op);}
        void visit(const LT *op) {visit_binary_operator(op);}
        void visit(const LE *op) {visit_binary_operator(op);}
        void visit(const GT *op) {visit_binary_operator(op);}
        void visit(const GE *op) {visit_binary_operator(op);}
        void visit(const And *op) {visit_binary_operator(op);}
        void visit(const Or *op) {visit_binary_operator(op);}

        void visit(const Not *op) {
            op->a.accept(this);
            ops+=1;
        }

        void visit(const Select *op) {
            op->condition.accept(this);
            op->true_value.accept(this);
            op->false_value.accept(this);
            ops+=1;
        }

        void visit(const Call * call) {
            if (call->call_type == Call::Halide) {
                loads+=1;
            } else if (call->call_type == Call::Intrinsic) {
                ops+=1;
            } else if (call->call_type == Call::Image) {
                loads+=1;
            }
            for (size_t i = 0; (i < call->args.size()); i++)
                call->args[i].accept(this);
        }

        void visit(const Let * let) {
            let->value.accept(this);
            let->body.accept(this);
        }

        void visit(const Load *) { assert(0); }
        void visit(const Ramp *) { assert(0); }
        void visit(const Broadcast *) { assert(0); }
        void visit(const LetStmt *) { assert(0); }
        void visit(const AssertStmt *) {}
        void visit(const ProducerConsumer *) { assert(0); }
        void visit(const For *) { assert(0); }
        void visit(const Store *) { assert(0); }
        void visit(const Provide *) { assert(0); }
        void visit(const Allocate *) { assert(0); }
        void visit(const Free *) { assert(0); }
        void visit(const Realize *) { assert(0); }
        void visit(const Block *) { assert(0); }
        void visit(const IfThenElse *) { assert(0); }
        void visit(const Evaluate *) { assert(0); }
};

bool is_simple_const(Expr e) {
    if (e.as<IntImm>()) return true;
    if (e.as<UIntImm>()) return true;
    if (e.as<FloatImm>()) return true;
    if (const Broadcast *b = e.as<Broadcast>()) {
        return is_simple_const(b->value);
    }
    return false;
}

void simplify_box(Box& b) {
    for (unsigned int i = 0; i < b.size(); i++) {
        b[i].min = simplify(b[i].min);
        b[i].max = simplify(b[i].max);
    }
}

/* Compute the regions of functions required to compute a region of the function
   'f' given symbolic sizes of the tile in each dimension. */
map<string, Box> regions_required(Function f,
                                  const vector< pair<Expr, Expr> > &sym_bounds,
                                  map<string, Function> &env,
                                  const FuncValueBounds &func_val_bounds){
    // Define the bounds for each variable of the function
    std::vector<Interval> bounds;
    int num_args = f.args().size();

    // The region of function 'f' for which the analysis is done ranges from
    // zero to tile_size in each dimension. The underlying assumption is that
    // the dependence patterns are more or less uniform over the range of the
    // function. This assumption may not hold for more sophisticated functions.
    // However, note that this assumption will not affect the program
    // correctness but might result in poor performance decisions. Polyhedral
    // analysis should be able to capture the exact dependence regions
    // compactly. Capturing the exact dependences may lead to large
    // approximations which are not desirable. Going forward as we encounter
    // more exotic patterns we will need to revisit this simple analysis.
    for (int arg = 0; arg < num_args; arg++)
        bounds.push_back(Interval(sym_bounds[arg].first, sym_bounds[arg].second));

    map<string, Box> regions;
    // Add the function and its region to the queue
    std::deque< pair<Function, std::vector<Interval> > > f_queue;
    f_queue.push_back(make_pair(f, bounds));
    // Recursively compute the regions required
    while(!f_queue.empty()) {
        Function curr_f = f_queue.front().first;
        vector<Interval> curr_bounds = f_queue.front().second;
        f_queue.pop_front();
        for (auto& val: curr_f.values()) {
            map<string, Box> curr_regions;
            Scope<Interval> curr_scope;
            int interval_index = 0;
            for (auto& arg: curr_f.args()) {
                // Check simplification cost
                Interval simple_bounds = Interval(simplify(curr_bounds[interval_index].min),
                                                  simplify(curr_bounds[interval_index].max));
                curr_scope.push(arg, simple_bounds);
                interval_index++;
            }
            curr_regions = boxes_required(val, curr_scope, func_val_bounds);
            // Each function will only appear once in curr_regions
            for (auto& reg: curr_regions) {
                // Merge region with an existing region for the function in
                // the global map
                if (regions.find(reg.first) == regions.end())
                    regions[reg.first] = reg.second;
                else
                    merge_boxes(regions[reg.first], reg.second);
                f_queue.push_back(make_pair(env[reg.first], reg.second.bounds));
            }
        }
    }
    // Simplify
    for (auto &f : regions) {
        simplify_box(f.second);
    }
    return regions;
}

/* Compute the redundant regions computed while computing a tile of the function
   'f' given sizes of the tile in each dimension. */
map<string, Box> redundant_regions(Function f, int dir,
                                   const vector<pair<Expr, Expr>> &sym_bounds,
                                   map<string, Function> &env,
                                   const FuncValueBounds &func_val_bounds){
    map<string, Box> regions = regions_required(f, sym_bounds, env,
                                                func_val_bounds);
    vector<pair<Expr, Expr>> shifted_bounds;
    int num_args = f.args().size();
    for (int arg = 0; arg < num_args; arg++) {
        if (dir == arg) {
            Expr len = sym_bounds[arg].second - sym_bounds[arg].first + 1;
            pair<Expr, Expr> bounds = make_pair(sym_bounds[arg].first + len,
                                              sym_bounds[arg].second + len);
            shifted_bounds.push_back(bounds);
        }
        else
            shifted_bounds.push_back(sym_bounds[arg]);
    }

    map<string, Box> regions_shifted = regions_required(f, shifted_bounds,
                                                        env, func_val_bounds);

    map<string, Box> overalps;
    for (auto& reg: regions) {
        if (regions_shifted.find(reg.first) == regions.end()) {
            // Interesting case to be dealt with
            assert(0);
        } else {
            Box b = reg.second;
            Box b_shifted = regions_shifted[reg.first];
            // The boxes should be of the same size
            assert(b.size() == b_shifted.size());
            // The box used makes things complicated but ignoring it for now
            Box b_intersect;
            for (unsigned int i = 0 ; i < b.size(); i++)
                b_intersect.push_back(interval_intersect(b[i], b_shifted[i]));
            // A function should appear once in the regions and therefore cannot
            // already be present in the overlaps map
            assert(overalps.find(reg.first) == overalps.end());
            overalps[reg.first] = b_intersect;
        }
    }
    // Simplify
    for (auto &f : overalps)
        simplify_box(f.second);

    return overalps;
}

map<string, Box> sym_to_concrete_bounds(vector< pair<Var, Var> > &sym,
                                        vector< pair<int, int> > &bounds,
                                        vector<bool> &eval,
                                        map<string, Box> &sym_regions) {

    map<string, Expr> replacements;
    for (unsigned int i = 0; i < sym.size(); i++) {
        if (eval[i]) {
            replacements[sym[i].first.name()] = bounds[i].first;
            replacements[sym[i].second.name()] = bounds[i].second;
        }
    }
    map<string, Box> concrete_regions;
    for (const auto &r: sym_regions) {
        Box concrete_box;
        for (unsigned int i = 0; i < r.second.size(); i++) {
            Expr lower = simplify(substitute(replacements, r.second[i].min));
            Expr upper = simplify(substitute(replacements, r.second[i].max));
            Interval concrete_bounds = Interval(lower, upper);
            concrete_box.push_back(concrete_bounds);
        }
        concrete_regions[r.first] = concrete_box;
    }
    return concrete_regions;
}

struct DependenceAnalysis {

    map<string, Function> &env;
    const FuncValueBounds &func_val_bounds;
    map<string, map<string, Box> > func_dep_regions;
    map<string, vector< map<string, Box> > > func_overlaps;
    map<string, vector< pair<Var, Var> > > func_sym;


    DependenceAnalysis(map<string, Function> &_env,
                     const FuncValueBounds &_func_val_bounds):
                     env(_env), func_val_bounds(_func_val_bounds) {
        for (auto& kv : env) {
            // For each argument create a variables which will server as the lower
            // and upper bounds of the interval corresponding to the argument
            const vector<string>  &args = kv.second.args();
            vector< pair<Expr, Expr> > sym_bounds;
            for (unsigned int arg = 0; arg < args.size(); arg++) {
                Var lower = Var(args[arg] + "_l");
                Var upper = Var(args[arg] + "_u");
                pair<Var, Var> sym = make_pair(lower, upper);
                pair<Expr, Expr> bounds = make_pair(Expr(lower), Expr(upper));
                func_sym[kv.first].push_back(sym);
                sym_bounds.push_back(bounds);
            }

            map<string, Box> regions = regions_required(kv.second, sym_bounds,
                                                        env, func_val_bounds);
            assert(func_dep_regions.find(kv.first) == func_dep_regions.end());
            func_dep_regions[kv.first] = regions;

            /*
               std::cout << "Function regions required for " << kv.first << ":" << std::endl;
               disp_regions(regions);
               std::cout << std::endl; */

            assert(func_overlaps.find(kv.first) == func_overlaps.end());
            for (unsigned int arg = 0; arg < args.size(); arg++) {
                map<string, Box> overlaps = redundant_regions(kv.second, arg,
                        sym_bounds, env,
                        func_val_bounds);
                func_overlaps[kv.first].push_back(overlaps);

                /*
                   std::cout << "Function region overlaps for var " <<
                   kv.second.args()[arg]  << " " << kv.first << ":" << std::endl;
                   disp_regions(overlaps);
                   std::cout << std::endl; */
            }
        }
    }

    map<string, Box> concrete_dep_regions(string name, vector<bool> &eval,
                                          vector<pair<int, int> > &bounds) {
        return sym_to_concrete_bounds(func_sym[name], bounds, eval,
                                      func_dep_regions[name]);
    }
};



int box_area(Box &b) {
    int box_area = 1;
    for(unsigned int i = 0; i < b.size(); i++) {
        // Maybe should check for unsigned integers and floats too
        if ((b[i].min.as<IntImm>()) && (b[i].max.as<IntImm>())) {
            const IntImm * bmin = b[i].min.as<IntImm>();
            const IntImm * bmax = b[i].max.as<IntImm>();
            // Count only if the overlap makes sense
            if (bmin->value <= bmax->value)
                box_area = box_area * (bmax->value - bmin->value + 1);
            else {
                box_area = 0;
                break;
            }
        } else {
            // std::cout << "Box area computation failed" << std::endl;
            // std::cout << "min:" << b[i].min << " max:" << b[i].max << std::endl;
            box_area = -1;
            break;
        }
    }
    return box_area;
}

int region_size(string func, Box &region, map<string, Function> &env) {

    Function &f = env[func];
    int area = box_area(region);
    if (area < 0)
        // Area could not be determined
        return -1;
    int size = 0;
    const vector<Type> &types = f.output_types();
    for(unsigned int i = 0; i < types.size(); i++)
        size += types[i].bytes();
    return area * size;
}

int region_size(map<string, Box> &funcs, map<string, Function> &env) {

    int total_size = 0;
    for(auto &f: funcs) {
        int size = region_size(f.first, f.second, env);
        if (size < 0)
            return -1;
        else
            total_size += size;
    }
    return total_size;
}

int region_cost(string func, Box &region,
                map<string, vector<pair<int, int> > > &func_cost) {
    int area = box_area(region);
    if (area < 0)
        // Area could not be determined
        return -1;
    auto &costs = func_cost[func];
    // Going over each of the outputs of the function
    int op_cost = 0;
    for (unsigned int t = 0; t < costs.size(); t++)
        op_cost += costs[t].first;
    int cost = area * (op_cost + 1);
    return cost;
}

int region_cost(map<string, Box> &regions,
                map<string, vector<pair<int, int> > > &func_cost) {
    int total_cost = 0;
    for(auto &f: regions) {
        int cost = region_cost(f.first, f.second, func_cost);
        if (cost < 0)
            return -1;
        else
            total_cost += cost;
    }
    return total_cost;
}

int overlap_cost(string cons, Function prod,
                 map<string, vector<map<string, Box> > > &func_overlaps,
                 map<string, vector<pair<int, int> > > &func_cost, int dim=-1) {
    auto &overlaps = func_overlaps[cons];
    int total_area = 0;
    assert((int)overlaps.size() > dim);
    for (unsigned int d = 0; d < overlaps.size(); d++) {
        // Overlap area
        if (overlaps[d].find(prod.name()) != overlaps[d].end()
                && (dim==-1 || dim == (int)d) ) {
            int area = box_area(overlaps[d][prod.name()]);
            if (area >= 0)
                total_area += area;
            else
                // Area could not be determined
                return -1;
        }
    }
    auto &costs = func_cost[prod.name()];
    // Going over each of the outputs of the function
    int op_cost = 0;
    for (unsigned int t = 0; t < costs.size(); t++)
        op_cost += costs[t].first;
    int overlap_cost = total_area * (op_cost + 1);
    return overlap_cost;
}

int overlap_cost(string cons, vector<Function> &prods,
                 map<string, vector<map<string, Box> > > &func_overlaps,
                 map<string, vector<pair<int, int> > > &func_cost, int dim=-1) {

    int total_cost = 0;
    for(auto& p: prods) {
        if (p.name()!=cons) {
            int cost = overlap_cost(cons, p, func_overlaps, func_cost, dim);
            if (cost < 0)
                // Cost could not be estimated
                return -1;
            else
                total_cost+=cost;
        }
    }
    return total_cost;
}

void add_children(map<string, set<string> > &children,
                  map<string, Function> &calls,
                  map<string, string> &inlines, string func) {
    for (auto &c: calls) {
        if (inlines.find(c.first) == inlines.end())
            children[c.first].insert(func);
        else {
            map<string, Function> recur_calls = find_direct_calls(c.second);
            add_children(children, recur_calls, inlines, func);
        }
    }
}



void disp_children(map<string, set<string> > &children) {
    for (auto &f: children) {
        std::cout << f.first <<  ":" << std::endl;
        for (auto &c: f.second)
            std::cout << c << ",";
        std::cout << std::endl;
    }
}

void disp_box(Box &b) {
    for (unsigned int dim = 0; dim < b.size(); dim++)
        std::cout << "(" << b[dim].min << "," << b[dim].max << ")";
}

struct Partitioner {

    struct Option {
        // Option is the cost when the prod_group is merged with the child_group
        // and computed at the granularity of the tile given by tile_sizes
        string prod_group;
        string child_group;
        // Tile sizes of along dimensions of the output of the child group
        // A tile size of -1 indicates no tiling along the dimension
        vector<int> tile_sizes;
        // A score indicating the benefit of the option
        float benefit;
    };

    map<string, Box> &pipeline_bounds;
    map<string, vector<pair<int, int> > > &func_cost;
    map<string, string> &inlines;
    DependenceAnalysis &analy;

    map<string, vector<Function> > groups;
    map<string, set<string> > children;

    Partitioner(map<string, Box> &_pipeline_bounds,
                map<string, string> &_inlines, DependenceAnalysis &_analy,
                map<string, vector<pair<int, int> > > &_func_cost):
                pipeline_bounds(_pipeline_bounds), inlines(_inlines),
                analy(_analy), func_cost(_func_cost) {
        // Place each function in its own group
        for (auto &kv: analy.env)
            groups[kv.first].push_back(kv.second);

        // Find consumers of each function relate groups with their children
        for (auto &kv: analy.env) {
            map<string, Function> calls = find_direct_calls(kv.second);
            for (auto &c: calls)
                children[c.first].insert(kv.first);
        }

        // disp_children(children);
        // Add inlined functions to their child group
        for (auto &in: inlines) {
            string dest = in.second;
            if (groups.find(dest) == groups.end()) {
                for (auto &g: groups)
                    for (auto &m: g.second)
                        if (m.name() == dest)
                            dest = g.first;
            }
            merge_groups(in.first, dest);
        }
    }

    void merge_groups(string cand_group, string child_group)
    {
        assert(groups.find(child_group) != groups.end());
        vector<Function> cand_funcs = groups[cand_group];

        groups.erase(cand_group);
        groups[child_group].insert(groups[child_group].end(),
                cand_funcs.begin(), cand_funcs.end());
        // Fix the children mapping
        children.erase(cand_group);
        for (auto &f: children) {
            set<string> &children = f.second;
            if (children.find(cand_group) != children.end()) {
                children.erase(cand_group);
                children.insert(child_group);
            }
        }
    }

    void disp_grouping() {
        for (auto& g: groups) {
            std::cout << "Group " <<  g.first  << " :"<< std::endl;
            for (auto& m: g.second)
                std::cout << m.name() << std::endl;
        }
    }

    int choose_candidate(const vector< pair<string, string> > &cand_pairs);
    map<string, vector<Function> > overlap_tile();

};

map<string, vector<Function> > Partitioner::overlap_tile() {

    // Partition the pipeline by iteratively merging groups until a fixpoint
    bool fixpoint = false;
    while(!fixpoint) {
        string cand_group, child_group;
        fixpoint = true;
        vector< pair<string, string> > cand_pairs;
        // Find all the groups which have a single child
        for (auto &g: groups) {
            if (children.find(g.first) != children.end()) {
                // TODO be careful about inputs and outputs to the pipeline
                int num_children = children[g.first].size();
                if (num_children == 1) {
                    auto cand = make_pair(g.first, *children[g.first].begin());
                    cand_pairs.push_back(cand);
                }
            }
        }

        // Pick a pair of groups to merge. This is a tricky choice.
        int cand_index = choose_candidate(cand_pairs, groups, pipeline_bounds,
                                          func_cost, analy);

        if (cand_index != -1) {
            cand_group = cand_pairs[cand_index].first;
            child_group = cand_pairs[cand_index].second;
            fixpoint = false;
        }

        // Do the necessary book keeping required to perform the merge if
        // there is a merge candidate
        if (!fixpoint) {
            merge_groups(groups, children, cand_group, child_group);
            //std::cout << "Megre candidate" << std::endl;
            //std::cout << cand_group << std::endl;
        }
    }

    return groups;
}

int Partitioner::choose_candidate(
                    const vector< pair<string, string> > &cand_pairs) {

    // The choose candidate operates by considering many posssible fusion
    // structures between each pair of candidates. The options considered are
    // computing a all functions in both the groups at some granularity of the
    // output function in the child group.
    //
    // Among these options the only ones considered are the ones that satisfy
    // the machine constraints. This means the following things:
    //
    // 1) Do all the intermediate buffers fit in the fast level of memory. One
    // needs to account for eary frees and the high watermark of intermediate
    // storage. There might be performance gains by doing the buffer
    // allocation statically as opposed to dynamic allocation. It might be
    // useful to investigate this both on CPU and GPU architectures.
    //
    // 2) Is the amount of redundant computation introduced in the process
    // give the best redundant compute vs. locality trade-off. One way to
    // handle this is to start with the option that introduces the least amount
    // of redundant computation and check if that satisfies the other criteria.
    // Then consider the next option until it gets to a point where it is
    // beneficial to load from slow memory than to redundantly compute.
    //
    // 3) Does the fused group have enough parallelism both for multiple cores.
    // This can get tricky as it has load balancing aspect to it too. For
    // example, if the group can be split into 10 tiles and there are 4 cores the
    // latency of the entire pipeline is 2 tiles. So either the number of tiles
    // have to a multiple of the cores or large in number to avoid the load
    // imbalance. Till this point I have not noticed the collapse being
    // particularly useful it might be an issue with Halide task scheduling. I
    // need experiments confirming this obsevation.
    //
    // 4) Does the fusion limit vectorization. Reordering function dimensions
    // and modifying data layout have significant interactions with
    // vectorization. As a first pass the goal is to not miss any obvious
    // vectorization and does not not create new oportunities.  Generating a
    // schedule which makes good use of vector units is a challenging problem
    // in itself.  It might be worthwile to perform a prepass on the pipeline
    // to first decide what is going to be vectorized and prevent further
    // phases from interfering with that decision.
    //
    // The options that are currently conisdered are computing at different
    // granularities at each level of the output function. The tile sizes at
    // each level are determined by the sizes of the intermediate data and the
    // size of the fast memory. We then construct a list of valid options atmost
    // one per candidate pair. For choosing among the options there needs to be
    // benefit associated with each of the options. The benefit we associate
    // with each of the choices is the potential number of accesses to slow
    // memory that are eliminated weighted by the inverse of the arithmetic
    // intensity of the child group in the pair.

    // Hard code machine parameters for now
    /*
    int fast_mem_size = 100;
    float overlap_thresh = 0.1;
    int parallelism = 8;
    */
    // TODO should consider vector length too

    vector<Option> options;
    vector<int> size_variants = {256, 128, 64, 32, 16};
    for (auto &p: cand_pairs) {

        std::cout << "(" << p.first << "," << p.second << ")" << std::endl;
        if (pipeline_bounds.find(p.second) != pipeline_bounds.end()) {
            std::cout << "Child size:" << std::endl;
            Box &b = pipeline_bounds[p.second];
            std::cout << p.second;
            disp_box(b);
            std::cout << std::endl;
        }
        std::cout << "Child function characteristics:" << std::endl;
        std::cout << "Loads:" << func_cost[p.second][0].second <<
                     " Ops:" << func_cost[p.second][0].first  << std::endl;
        std::cout << "Expr:" << analy.env[p.second].values()[0] << std::endl;
        std::cout << std::endl;

        // For each of the pairs create an option and evaluate

        // Get the output function of the child group
        Function output = analy.env[p.second];
        const vector<string> &args = output.args();
        // From the outer to the inner most argument
        for (int i = (int)args.size() - 1; i >= 0; i --) {
            for (auto s: size_variants) {
                Option opt;
                opt.prod_group = p.first;
                opt.child_group = p.second;
                opt.benefit = -1;
                for (int j = 0; j < i; j++)
                    opt.tile_sizes.push_back(-1);
                for (unsigned int j = i; j < args.size(); j++)
                    opt.tile_sizes.push_back(s);
                // evaluate_option(opt);
            }

        }
    }
    return -1;
}

void disp_function_value_bounds(const FuncValueBounds &func_val_bounds) {

	for (auto& kv: func_val_bounds) {
        std::cout << kv.first.first << "," << kv.first.second << ":"
                  << "(" << kv.second.min  << ","  << kv.second.max << ")"
                  << std::endl;
    }
}

void disp_schedule_and_storage_mapping(map<string, Function> &env) {
    // Names of all the functions in the environment and their schedules
    for (auto& kv : env) {
        std::cout << schedule_to_source(kv.second,
                                        kv.second.schedule().compute_level(),
                                        kv.second.schedule().store_level())
                  << std::endl;
    }
    std::cout << std::endl;
}

void disp_regions(map<string, Box> &regions) {
    for (auto& reg: regions) {
        std::cout << reg.first;
        disp_box(reg.second);
        std::cout << std::endl;
    }
}

void disp_inlines(map<string, string> &inlines) {
    for (auto& in: inlines)
        std::cout << in.first << "->" << in.second << std::endl;
}

map<string, string> simple_inline(map<string, vector<const Call*>> &all_calls,
                                  map<string, vector<string> > &consumers,
                                  map<string, Function> &env) {
    map<string, string> inlines;
    for (auto& fcalls: all_calls) {
        // Check if all arguments to the function call over all the calls are
        // one-to-one. If this holds and the number of calls == 1 it is a good
        // candidate for inlining.
        bool all_one_to_one = true;
        int num_calls = 0;
        for (auto& call: fcalls.second){
            num_calls++;
            for(auto& arg: call->args){
                // Skip casts to an integer there seems to be a bug lurking
                // in is_one_to_one
                bool one_to_one = (!arg.as<Cast>()) && is_one_to_one(arg);
                all_one_to_one = all_one_to_one && (one_to_one
                                                    || is_simple_const(arg));
            }
        }
        if (consumers[fcalls.first].size() == 1 &&
                all_one_to_one && num_calls == 1) {
            inlines[fcalls.first] = consumers[fcalls.first][0];
            env[fcalls.first].schedule().store_level().var = "";
            env[fcalls.first].schedule().compute_level().var = "";
        }
    }
    return inlines;
}

// Helpers for schedule surgery

// Parallel
void parallelize_dim(Function &func, vector<int> levels) {

    vector<Dim> &dims = func.schedule().dims();
    // TODO Provide an option for collapsing all the parallel
    // loops
    for (auto dim: levels)
        dims[dim].for_type = ForType::Parallel;
}

void move_dim_to_outermost(Function& func, int dim) {

    vector<Dim> &dims = func.schedule().dims();
    dims.insert(dims.end() - 1, dims[dim]);
    dims.erase(dims.begin() + dim);
}

void move_dim_to_var(Function& func, int dim, string var) {

    vector<Dim> &dims = func.schedule().dims();
    int cand_dim = -1;
    for (unsigned int i = 0;  i < dims.size(); i++)
        if (dims[i].var == var)
            cand_dim = i;
    assert(cand_dim != -1);
    dims.insert(dims.begin() + cand_dim, dims[dim]);
    dims.erase(dims.begin() + dim);
}

void swap_dim(Function &func, int dim1, int dim2) {

    vector<Dim> &dims = func.schedule().dims();

    string name1 = dims[dim1].var;
    ForType type1 = dims[dim1].for_type;
    bool pure1 = dims[dim1].pure;

    dims[dim1].var = dims[dim2].var;
    dims[dim1].for_type = dims[dim2].for_type;
    dims[dim1].pure = dims[dim2].pure;

    dims[dim2].var = name1;
    dims[dim2].for_type = type1;
    dims[dim2].pure = pure1;
}

// Splitting
void split_dim(Function &func, int dim, int split_size, bool partial) {

    vector<Dim> &dims = func.schedule().dims();
    // Vectorization is not easy to insert in a Function object
    // have to revisit if this is the cleanest way to do it
    string old = dims[dim].var;
    string inner_name, outer_name, old_name;

    old_name = dims[dim].var;
    inner_name = old_name + "." + "in";
    outer_name = old_name + "." + "out";
    dims.insert(dims.begin() + dim, dims[dim]);
    dims[dim].var = inner_name;
    dims[dim+1].var = outer_name;
    dims[dim+1].pure = dims[dim].pure;
    dims[dim+1].for_type = dims[dim].for_type;

    // Add the split to the splits list
    Split split = {old_name, outer_name, inner_name, split_size,
                   false, partial, Split::SplitVar};
    func.schedule().splits().push_back(split);
}

void fuse_dim(Function &func, int dim1, int dim2) {
    // Add the fuse to the splits list
    string inner_name, outer_name, fused_name;
    vector<Dim> &dims = func.schedule().dims();

    outer_name = dims[dim1].var;
    bool outer_pure = dims[dim1].pure;
    dims.erase(dims.begin() + dim1);

    inner_name = dims[dim2].var;
    fused_name = inner_name + "." + outer_name;
    dims[dim2].var = fused_name;
    dims[dim2].pure &= outer_pure;

    Split split = {fused_name, outer_name, inner_name, Expr(),
                   true, false, Split::FuseVars};
    func.schedule().splits().push_back(split);
}

// Vectorization
void vectorize_dim(Function &func, int dim, int vec_width) {

    vector<Dim> &dims = func.schedule().dims();
    split_dim(func, dim, vec_width, false);
    dims[dim].for_type = ForType::Vectorized;
}

bool check_dim_size(Function &func, int dim, int min_size,
                    map<string, Box> &pipeline_bounds) {
    if (pipeline_bounds.find(func.name()) == pipeline_bounds.end()) {
        // Optimistic
        return true;
    }
    else {
        Box &b = pipeline_bounds[func.name()];
        vector<Dim> &dims = func.schedule().dims();
        const vector<string> vars = func.args();
        for (unsigned int i = 0; i < vars.size(); i++)
            if (dims[dim].var == vars[i]) {
                if ((b[i].min.as<IntImm>()) && (b[i].max.as<IntImm>())) {
                    const IntImm * bmin = b[i].min.as<IntImm>();
                    const IntImm * bmax = b[i].max.as<IntImm>();
                    int size = bmax->value - bmin->value + 1;
                    return size >= min_size;
                } else  {
                    return true;
                }
            }
    }
    return true;
}

void simple_vectorize(Function &func, int inner_dim, int vec_width) {
    // Collect all the load args
    FindCallArgs find;
    func.accept(&find);
    // For all the loads find the stride of the innermost loop
    bool constant_stride = true;
    for(auto& larg: find.load_args) {
        Expr diff = simplify(finite_difference(larg[inner_dim],
                             func.args()[inner_dim]));
        constant_stride = constant_stride && is_simple_const(diff);
    }
    if (constant_stride)
        vectorize_dim(func, inner_dim, vec_width);
}

bool check_bounds_on_outputs(const vector<Function> &outputs) {
    bool bounds_avail = true;
    for (auto &out : outputs) {
        const vector<Bound> &bounds = out.schedule().bounds();
        if (bounds.size() != out.args().size()) {
            bounds_avail = false;
            break;
        }
        vector<string> vars = out.args();

        for (unsigned int i = 0; i < bounds.size(); i++) {
            if (std::find(vars.begin(), vars.end(), bounds[i].var) == vars.end()
                    || !((bounds[i].min.as<IntImm>()) &&
                        (bounds[i].extent.as<IntImm>())))  {
                bounds_avail = false;
                break;
            }
        }
    }
    return bounds_avail;
}

void schedule_advisor(const vector<Function> &outputs,
                      const vector<string> &order,
                      map<string, Function> &env,
                      const FuncValueBounds &func_val_bounds,
                      bool root_default, bool auto_inline,
                      bool auto_par, bool auto_vec) {

    if (root_default) {
        // Changing the default to compute root. This does not completely clear
        // the user schedules since the splits are already part of the domain. I
        // do not know if there is a clean way to remove them.  This also
        // touches on the topic of completing partial schedules specified by the
        // user as opposed to completely erasing them.
    	for (auto& kv : env) {
    		// Have to reset the splits as well
    		kv.second.schedule().store_level().func = "";
    		kv.second.schedule().store_level().var = "__root";
        	kv.second.schedule().compute_level().func = "";
        	kv.second.schedule().compute_level().var = "__root";
    	}
    }

    // TODO infer the bounds of each function in the pipeline based on the
    // estimates of output sizes and the parameters

    // TODO explain strcuture
    map<string, Box> pipeline_bounds;

    // TODO Method for estimating cost when reductions are involved
    // TODO explain structure
    std::map<string, std::vector<pair<int, int> > > func_cost;
    for (auto& kv : env) {
        // std::cout << kv.first << ":" << std::endl;
        assert(func_cost.find(kv.first) == func_cost.end());
        for (auto& e: kv.second.values()) {
            ExprCostEarly cost_visitor;
            e.accept(&cost_visitor);
            auto p = make_pair(cost_visitor.ops, cost_visitor.loads);
            func_cost[kv.first].push_back(p);
            /*
            std::cout << e << " loads:" << cost_visitor.loads << " ops:"
                      << cost_visitor.ops << std::endl;
            */
        }
    }

    // TODO explain structure
    map<string, vector<const Call*> > all_calls;
    map<string, vector<string> > consumers;
    for (auto& kv:env) {

    	FindCallArgs call_args;
    	kv.second.accept(&call_args);
    	//std::cout << kv.second.name() << ":" << std::endl;
    	for (auto& fcalls: call_args.calls){
            consumers[fcalls.first].push_back(kv.first);
    		all_calls[fcalls.first].insert(all_calls[fcalls.first].end(),
    								  	   fcalls.second.begin(),
                                           fcalls.second.end());
            /*
    		for (auto& call: fcalls.second){
    			std::cout << fcalls.first << "(";
    			for(auto& arg: call->args){
    				std::cout << arg << ",";
    			}
    			std::cout << "),";
    		}
            */
    	}
    	//std::cout << std::endl;
    }

    // Make obvious inline decisions early. Partitioning downstream may end up with
    // a function that is inlined as the representative of the group. This will
    // result in a conflicting schedule.
    map<string, string> inlines;
    if (auto_inline)
        inlines = simple_inline(all_calls, consumers, env);

    std::cout << "Inlining:" << std::endl;
    disp_inlines(inlines);
    std::cout << std::endl;

    int vec_len = 8;
    bool overlap_tile = true;
    auto_vec = true;
    auto_par = true;
    unsigned int tile_dims = 2;
    int tile_size = 128;

    if (overlap_tile) {

        // Dependence analysis

        // For each function compute all the regions of upstream functions
        // required to compute a region of the function

        DependenceAnalysis analy(env, func_val_bounds);

        for (auto &reg: analy.func_dep_regions) {
            disp_regions(reg.second);
            std::cout << std::endl;
        }

        bool bounds_avail = check_bounds_on_outputs(outputs);
        // std::cout << "output bounds:" << bounds_avail << std::endl;

        if (bounds_avail) {
            for (auto &out: outputs) {
                vector<pair<int, int> > bounds;
                vector<bool> eval;
                vector<string> vars = out.args();
                for (unsigned int i = 0; i < vars.size(); i++) {
                    bool found = false;
                    for (auto &b: out.schedule().bounds())
                        if (b.var == vars[i]) {
                            const IntImm * bmin = b.min.as<IntImm>();
                            const IntImm * bextent = b.extent.as<IntImm>();
                            pair<int, int> p = make_pair(bmin->value, bmin->value
                                                         + bextent->value - 1);
                            bounds.push_back(p);
                            eval.push_back(true);
                            found = true;
                        }
                    if(!found) {
                        bounds.push_back(make_pair(-1, -1));
                        eval.push_back(false);
                    }
                }

                map<string, Box> regions =
                        analy.concrete_dep_regions(out.name(), eval, bounds);

                // Add the output region to the pipeline bounds as well
                Box out_box;
                for (unsigned int i = 0; i < bounds.size(); i++)
                    out_box.push_back(Interval(bounds[i].first,
                                               bounds[i].second));
                regions[out.name()] = out_box;

                for (auto& reg: regions) {
                    // Merge region with an existing region for the function in
                    // the global map
                    if (pipeline_bounds.find(reg.first) == pipeline_bounds.end())
                        pipeline_bounds[reg.first] = reg.second;
                    else
                        merge_boxes(pipeline_bounds[reg.first], reg.second);
                }
            }
        }

        //disp_regions(pipeline_bounds);

        // Grouping
        map<string, vector<Function> > groups;

        Partitioner part(pipeline_bounds, inlines, analy, func_cost);
        groups = part.overlap_tile();

        //disp_grouping(groups);

        // Schedule generation based on grouping
        for (auto& g: groups) {
            // Create a tiled traversal for the output of the group
            Function &g_out = env[g.first];
            // Choose which dimensions should be tiled. For now tile all
            // dimensions
            assert(inlines.find(g_out.name()) == inlines.end());
            vector<string> vars;
            vector<Dim> &dims = g_out.schedule().dims();
            if (dims.size() <= 0 || !g_out.is_pure())
                continue;
            for(int i = 0; i < (int)dims.size() - 1; i++) {
                // Restricting tiling to 2D
                if (i < (int)tile_dims) {
                    // Check if dimension is large enough to tile
                    if (check_dim_size(g_out, i, tile_size, pipeline_bounds))
                        vars.push_back(dims[i].var);
                }
            }

            // TODO Eventually this weird step should be changed into something
            // that actually gives proper ordering within a tile

            // Find the level at which the tiles should be computed
            // int min_overlap = 0;
            /*
            map<string, Function> calls = find_direct_calls(g_out);
            vector<Function> prods;

            // Captures the reads even if there are no members in the group
            for(auto &c: calls)
                prods.push_back(c.second);
            for(auto &m: g.second)
                if (calls.find(m.name()) == calls.end())
                    prods.push_back(m);

            int zero_reuse_dim = -1;
            for(int i = 0; i < (int)dims.size() - 1; i++) {
            // Skip the variables that are chosen for tiling and are above the
            // tiling
                if (i >= tile_dims) {
                    int cost = overlap_cost(g_out.name(), prods, func_overlaps,
                                            func_cost, i);
                    if (cost == 0)
                        zero_reuse_dim = i;
                    std::cout << g_out.name() << "," << dims[i].var << ","
                              << cost << std::endl;
                }
            }
            string zero_reuse_var;
            if (zero_reuse_dim != -1)
                zero_reuse_var = dims[zero_reuse_dim].var;
            else
                zero_reuse_var = dims[dims.size() - 1].var;
            */

            // TODO use the bounds
            /*
            vector<Bound> &bounds = g_out.schedule().bounds();
            for(unsigned int i = 0; i < bounds.size(); i++) {
                std::cout << g_out.name() << " " << bounds[i].var << "("
                          << bounds[i].min  << "," << bounds[i].extent << ")"
                          << std::endl;
            }
            */

            //int inner_tile_dim = 0;
            int num_tile_dims = 0;
            //string inner_tile_var = dims[dims.size() - 1].var;
            for(auto &v: vars) {
                int index = -1;
                for (int i = 0; i < (int)dims.size() - 1; i++)
                    if (dims[i].var == v) {
                        index = i;
                        break;
                    }
                assert(index!=-1);
                split_dim(g_out, index, tile_size, false);
                /*
                if (inner_tile_dim < (int)dims.size() - 1) {
                    swap_dim(g_out, index, inner_tile_dim);
                    inner_tile_dim++;
                }*/
                move_dim_to_outermost(g_out, index + 1);
                /*
                move_dim_to_var(g_out, index + 1, zero_reuse_var);
                if (num_tile_dims == 0)
                    for (int i = 0; i < (int)dims.size(); i++)
                        if (dims[i].var == zero_reuse_var) {
                            inner_tile_var = dims[i - 1].var;
                            break;
                        }
                */
                num_tile_dims++;
            }
            // int num_par_dims = std::min(2, (int)dims.size() - 1);
            int num_fused_dims = 0;
            if (g_out.is_pure()) {
                // Fuse all the tile dims?
                int outer_dim = dims.size() - 2;
                //for (int i = 0; i < num_par_dims - 1; i++) {
                /*
                for (int i = 0; i < num_tile_dims - 1; i++) {
                    if (outer_dim > 0) {
                        fuse_dim(g_out, outer_dim, outer_dim - 1);
                        outer_dim = dims.size() - 2;
                        num_fused_dims++;
                    }
                }*/
                vector<int> levels;
                levels.push_back(outer_dim);
                if (auto_par)
                    parallelize_dim(g_out, levels);
                if (auto_vec) {
                    if (check_dim_size(g_out, 0, vec_len, pipeline_bounds))
                        simple_vectorize(g_out, 0, vec_len);
                    //inner_tile_dim++;
                }
            }
            for (auto &m: g.second) {
                int outer_dim = dims.size() - 2;
                /* int inner_tile_dim = dims.size() - 1;
                for (unsigned int i = 0; i < dims.size(); i++)
                    if (dims[i].var == inner_tile_var) {
                        inner_tile_dim = i;
                        break;
                    }
                */
                if (m.name() != g_out.name() &&
                   inlines.find(m.name()) == inlines.end() && num_tile_dims > 0) {
                    //int compute_level = inner_tile_dim;
                    int compute_level = outer_dim - num_tile_dims +
                                                    num_fused_dims + 1;
                    m.schedule().store_level().func = g_out.name();
                    //m.schedule().store_level().var = dims[compute_level+1].var;
                    m.schedule().store_level().var = dims[compute_level].var;
                    m.schedule().compute_level().func = g_out.name();
                    m.schedule().compute_level().var = dims[compute_level].var;
                    if (m.is_pure() && auto_vec)
                        if (check_dim_size(m, 0, vec_len, pipeline_bounds))
                            simple_vectorize(m, 0, vec_len);
                }
            }
        }

    } else {

    // TODO Integrating prior analysis and code generation with the grouping
    // algorithm to do inlining, vectorization and parallelism

    // TODO Method for reordering and unrolling based on reuse across iterations

        if (auto_par || auto_vec) {
            // Parallelize and vectorize
            // Vectorization can be quite tricky it is hard to determine when
            // exactly it will benefit. The simple strategy is to check if the
            // arguments to the loads have the proper stride.
            for (auto& kv:env) {
                // Skipping all the functions for which the choice is inline
                if (inlines.find(kv.first) != inlines.end())
                    continue;
                // If a function is pure all the dimensions are parallel
                if (kv.second.is_pure()) {
                    // Parallelize the outer most dimension
                    // Two options when the number of iterations are small
                    // -- Collapse the two outer parallel loops
                    // -- If there is only a single dimension just vectorize
                    int outer_dim = kv.second.dimensions() - 1;
                    vector<int> levels;
                    levels.push_back(outer_dim);
                    if (auto_par)
                        parallelize_dim(kv.second, levels);
                    // The vector width also depends on the type of the operation
                    // and on the machine characteristics. For now just doing a
                    // blind 8 width vectorization.
                    if (kv.second.dimensions() > 1 && auto_vec)
                        if (check_dim_size(kv.second, 0, vec_len, pipeline_bounds))
                            simple_vectorize(kv.second, 0, vec_len);
                } else {
                    // Parallelism in reductions can be tricky
                    std::cout << std::endl;
                }
            }
        }
    }

    if (root_default || auto_vec || auto_par || auto_inline)
        disp_schedule_and_storage_mapping(env);

	return;
}

}
}
