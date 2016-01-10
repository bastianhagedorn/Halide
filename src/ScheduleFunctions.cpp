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
#include "RealizationOrder.h"

#include <algorithm>
#include <limits>

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

class FindAllCalls : public IRVisitor {
    public:
        set<string> calls;
        using IRVisitor::visit;

        void visit(const Call *call) {
            // See if images need to be included
            if (call->call_type == Call::Halide ||
                call->call_type == Call::Image) {
                calls.insert(call->name);
            }
            for (size_t i = 0; (i < call->args.size()); i++)
                call->args[i].accept(this);
        }
};

long long get_func_out_size(Function &f) {
    long long size = 0;
    const vector<Type> &types = f.output_types();
    for(unsigned int i = 0; i < types.size(); i++)
        size += types[i].bytes();
    if (size == 0)
        // Hack to over come weirdness for inputs to the pipeline
        size = 4;
    return size;
}

class VectorExprCheck : public IRVisitor {
    public:
        bool can_vec;
        string var;

        VectorExprCheck(string _var): var(_var) {
            can_vec = true;
        }

        using IRVisitor::visit;

        void visit(const IntImm *) {}
        void visit(const UIntImm *) {}
        void visit(const FloatImm *) { can_vec = false; }
        void visit(const StringImm *) { can_vec = false; }
        void visit(const Cast *) { can_vec = false; }
        void visit(const Variable * v) { can_vec = (can_vec) && (v->name == var); }

        template<typename T>
            void visit_binary_operator(const T *op) {
                op->a.accept(this);
                op->b.accept(this);
            }

        void visit(const Add *op) {visit_binary_operator(op);}
        void visit(const Sub *op) {visit_binary_operator(op);}

        void visit(const Mul *op) {
            visit_binary_operator(op);
            Expr a = simplify(op->a);
            Expr b = simplify(op->b);
            if ( !(a.as<UIntImm>() || a.as<IntImm>()) &&
                     !(b.as<UIntImm>() || b.as<IntImm>()) )
                can_vec = false;
        }

        void visit(const Div *op) {
            visit_binary_operator(op);
            Expr b = simplify(op->b);
            if(!(b.as<UIntImm>() || b.as<IntImm>()))
                can_vec = false;

        }

        void visit(const Mod *op) {
            visit_binary_operator(op);
            Expr b = simplify(op->b);
            if(!(b.as<UIntImm>() || b.as<IntImm>()))
                can_vec = false;
        }

        void visit(const Min *op) { can_vec = false;}
        void visit(const Max *op) { can_vec = false;}
        void visit(const EQ *op) { can_vec = false;}
        void visit(const NE *op) { can_vec = false;}
        void visit(const LT *op) { can_vec = false;}
        void visit(const LE *op) { can_vec = false;}
        void visit(const GT *op) { can_vec = false;}
        void visit(const GE *op) { can_vec = false;}
        void visit(const And *op) { can_vec = false;}
        void visit(const Or *op) { can_vec = false;}

        void visit(const Not *op) {
            op->a.accept(this);
        }

        void visit(const Select *op) { can_vec = false; }

        void visit(const Call * call) { can_vec = false; }

        void visit(const Let * let) { assert(0); }
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
        void visit(const Cast * op) {
            op->value.accept(this);
            ops+=1;
        }
        void visit(const Variable *) {}

        template<typename T>
            void visit_binary_operator(const T *op, int cost) {
                op->a.accept(this);
                op->b.accept(this);
                ops += cost;
            }

        void visit(const Add *op) {visit_binary_operator(op, 1);}
        void visit(const Sub *op) {visit_binary_operator(op, 1);}
        void visit(const Mul *op) {visit_binary_operator(op, 2);}
        void visit(const Div *op) {visit_binary_operator(op, 4);}
        void visit(const Mod *op) {visit_binary_operator(op, 2);}
        void visit(const Min *op) {visit_binary_operator(op, 2);}
        void visit(const Max *op) {visit_binary_operator(op, 2);}
        void visit(const EQ *op) {visit_binary_operator(op, 1);}
        void visit(const NE *op) {visit_binary_operator(op, 1);}
        void visit(const LT *op) {visit_binary_operator(op, 1);}
        void visit(const LE *op) {visit_binary_operator(op, 1);}
        void visit(const GT *op) {visit_binary_operator(op, 1);}
        void visit(const GE *op) {visit_binary_operator(op, 1);}
        void visit(const And *op) {visit_binary_operator(op, 1);}
        void visit(const Or *op) {visit_binary_operator(op, 1);}

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
            // TODO figure out the call types and how to distinguish between
            // them
            if (call->call_type == Call::Halide) {
                loads+=1;
            } else if (call->call_type == Call::Extern) {
                ops+=5;
            } else if (call->call_type == Call::Image) {
                loads+=1;
            } else if (call->call_type == Call::Intrinsic) {
                ops+=1;
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

/* Compute the regions of producers required to compute a region of the function
   'f' given symbolic sizes of the tile in each dimension. */
map<string, Box> regions_required(Function f,
                                  const vector< pair<Expr, Expr> > &sym_bounds,
                                  map<string, Function> &env,
                                  const FuncValueBounds &func_val_bounds){
    // Define the bounds for each variable of the function
    std::vector<Interval> bounds;
    int num_args = f.args().size();

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
        for (auto &val: curr_f.values()) {
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
        // Currently handling only a single update which covers simple
        // reductions which we want to handle
        assert(curr_f.updates().size() <= 1);
        for (auto &update: curr_f.updates()) {
            for (auto &val: update.values) {
                map<string, Box> curr_regions;
                Scope<Interval> curr_scope;
                int interval_index = 0;
                vector<Expr> exprs;
                exprs.push_back(val);
                for (auto &arg: update.args) {
                    Interval simple_bounds = Interval(simplify(curr_bounds[interval_index].min),
                                                      simplify(curr_bounds[interval_index].max));
                    // Check for a pure variable
                    const Variable *v = arg.as<Variable>();
                    if (!v)
                        // Need to evaluate boxes required on args that are not pure
                        // for potenial calls to other functions
                        exprs.push_back(arg);
                    else
                        curr_scope.push(v->name, simple_bounds);
                    interval_index++;
                }

                if (update.domain.defined()) {
                    for (auto &rvar: update.domain.domain()) {
                        Interval simple_bounds = Interval(rvar.min, rvar.min + rvar.extent - 1);
                        curr_scope.push(rvar.var, simple_bounds);
                    }
                }

                for (auto &e: exprs) {
                    curr_regions = boxes_required(e, curr_scope, func_val_bounds);
                    for (auto& reg: curr_regions) {
                        // Merge region with an existing region for the function in
                        // the global map
                        if(reg.first != curr_f.name()) {
                            if (regions.find(reg.first) == regions.end())
                                regions[reg.first] = reg.second;
                            else
                                merge_boxes(regions[reg.first], reg.second);
                            f_queue.push_back(make_pair(env[reg.first], reg.second.bounds));
                        }
                    }
                }
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

class ExprClone : public IRVisitor {

public:
    Expr e;
    Expr clone;
    map<Expr, Expr, ExprCompare> subs;
    ExprClone(Expr _e) : e(_e) {
        e.accept(this);
        clone = subs[e];
    }

    using IRVisitor::visit;

    template<typename T>
        void clone_binary_operator(const T *op) {
            op->a.accept(this);
            op->b.accept(this);
            Expr e = T::make(subs[op->a], subs[op->b]);
            subs[op] = e;
        }

    void visit(const Add *op) {clone_binary_operator(op);}
    void visit(const Sub *op) {clone_binary_operator(op);}
    void visit(const Mul *op) {clone_binary_operator(op);}
    void visit(const Div *op) {clone_binary_operator(op);}
    void visit(const Mod *op) {clone_binary_operator(op);}
    void visit(const Min *op) {clone_binary_operator(op);}
    void visit(const Max *op) {clone_binary_operator(op);}
    void visit(const EQ *op)  {clone_binary_operator(op);}
    void visit(const NE *op)  {clone_binary_operator(op);}
    void visit(const LT *op)  {clone_binary_operator(op);}
    void visit(const LE *op)  {clone_binary_operator(op);}
    void visit(const GT *op)  {clone_binary_operator(op);}
    void visit(const GE *op)  {clone_binary_operator(op);}
    void visit(const And *op) {clone_binary_operator(op);}
    void visit(const Or *op)  {clone_binary_operator(op);}

    void visit(const IntImm *op) { subs[op] = op;}
    void visit(const UIntImm *op) { subs[op] = op;}
    void visit(const FloatImm *op) { subs[op] = op;}
    void visit(const StringImm *op) { subs[op] = op;}
    void visit(const Variable *op)  { subs[op] = Variable::make(op->type,
                                                                op->name);}

    void visit(const Cast *op) {
        op->value.accept(this);
        Expr e = Cast::make(op->type, subs[op->value]);
        subs[op] = e;
    }

    void visit(const Not *op) {
        op->a.accept(this);
        Expr e = Not::make(subs[op->a]);
        subs[op] = e;
    }

    void visit(const Select *op)  {
        op->condition.accept(this);
        op->true_value.accept(this);
        op->false_value.accept(this);
        Expr e = Select::make(subs[op->condition], subs[op->true_value],
                              subs[op->false_value]);
        subs[op] = e;
    }

    void visit(const Load *op) {
        op->index.accept(this);
        Expr e = Load::make(op->type, op->name, subs[op->index],
                            op->image, op->param);
        subs[op] = e;
    }

    void visit(const Ramp *op) {
        op->base.accept(this);
        op->stride.accept(this);
        Expr e = Ramp::make(subs[op->base], subs[op->stride], op->lanes);
        subs[op] = e;
    }

    void visit(const Broadcast *op) {
        op->value.accept(this);
        Expr e = Broadcast::make(subs[op->value], op->lanes);
        subs[op] = e;
    }

    void visit(const Call *op) {
        vector<Expr > new_args(op->args.size());

        for (size_t i = 0; i < op->args.size(); i++) {
            op->args[i].accept(this);
            new_args[i] = subs[op->args[i]];
        }

        Expr e = Call::make(op->type, op->name, new_args, op->call_type,
                            op->func, op->value_index, op->image, op->param);
        subs[op] = e;
    }

    void visit(const Let *op) {
        op->value.accept(this);
        op->body.accept(this);
        Expr e = Let::make(op->name, subs[op->value], subs[op->body]);
        subs[op] = e;
    }

    void visit(const LetStmt *op) { assert(0); }
    void visit(const AssertStmt *op) { assert(0); }
    void visit(const ProducerConsumer *op) { assert(0); }
    void visit(const For *op) { assert(0); }
    void visit(const Store *op) { assert(0); }
    void visit(const Provide *op) { assert(0); }
    void visit(const Allocate *op) { assert(0); }
    void visit(const Free *op) { assert(0); }
    void visit(const Realize *op) { assert(0); }
    void visit(const Block *op) { assert(0); }
    void visit(const IfThenElse *op) { assert(0);}
    void visit(const Evaluate *op) { assert(0); }

};

map<string, Box> sym_to_concrete_bounds(vector< pair<Var, Var> > &sym,
                                        vector< pair<int, int> > &bounds,
                                        vector<bool> &eval,
                                        map<string, Box> &sym_regions,
                                        map<string, Function> &env) {

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
            //ExprClone cmin(r.second[i].min);
            //ExprClone cmax(r.second[i].max);
            Expr lower = simplify(substitute(replacements, r.second[i].min));
            Expr upper = simplify(substitute(replacements, r.second[i].max));

            // Use the bounds if the lower and upper bounds cannot be
            // determined
            if (!lower.as<IntImm>()) {
                for (auto &b: env[r.first].schedule().bounds())
                    if (b.var == env[r.first].args()[i])
                        lower = Expr(b.min.as<IntImm>()->value);

            }

            if (!upper.as<IntImm>()) {
                for (auto &b: env[r.first].schedule().bounds())
                    if (b.var == env[r.first].args()[i]) {
                        const IntImm * bmin = b.min.as<IntImm>();
                        const IntImm * bextent = b.extent.as<IntImm>();
                        upper = Expr(bmin->value + bextent->value - 1);
                    }
            }

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
                                      func_dep_regions[name], env);
    }

    vector< map<string, Box> > concrete_overlap_regions(
                                             string name, vector<bool> &eval,
                                             vector<pair<int, int> > &bounds) {
        vector< map<string, Box> > conc_overlaps;
        for (auto & dir: func_overlaps[name]) {
            map<string, Box> conc_reg =
                sym_to_concrete_bounds(func_sym[name], bounds, eval,
                                       dir, env);
            conc_overlaps.push_back(conc_reg);
        }
        return conc_overlaps;
    }

};

int get_min(const Interval &i) {

    if ((i.min.as<IntImm>())) {
        const IntImm * bmin = i.min.as<IntImm>();
        return bmin->value;
    }
    return std::numeric_limits<int>::max();
}

int get_extent(const Interval &i) {

    if ((i.min.as<IntImm>()) && (i.max.as<IntImm>())) {
        const IntImm * bmin = i.min.as<IntImm>();
        const IntImm * bmax = i.max.as<IntImm>();
        // Count only if the overlap makes sense
        if (bmin->value <= bmax->value)
            return (bmax->value - bmin->value + 1);
        else
            return 0;
    }
    /* TODO Check if this is necessary at some point
    else {
        Expr diff = simplify(i.max - i.min);
        std::cout << diff << std::endl;
        if (diff.as<IntImm>())
            return diff.as<IntImm>()->value;
    } */
    return -1;
}

pair<int, int> get_bound(const Interval &i) {

    if ((i.min.as<IntImm>()) && (i.max.as<IntImm>())) {
        const IntImm * bmin = i.min.as<IntImm>();
        const IntImm * bmax = i.max.as<IntImm>();
        return make_pair(bmin->value, bmax->value);
    }
    return make_pair(std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::min());
}

long long box_area(Box &b) {
    long long box_area = 1;
    for(unsigned int i = 0; i < b.size(); i++) {
        // Maybe should check for unsigned integers and floats too
        int extent = get_extent(b[i]);
        if (extent > 0 && box_area > 0)
            box_area = box_area * extent;
        else if (extent == 0) {
            box_area = 0;
            break;
        } else {
            box_area = -1;
        }
    }
    return box_area;
}

long long region_size(string func, Box &region, map<string, Function> &env) {
    Function &f = env[func];
    long long area = box_area(region);
    if (area < 0)
        // Area could not be determined
        return -1;
    long long size = get_func_out_size(f);
    return area * size;
}

long long region_size(map<string, Box> &regions, map<string, Function> &env,
                      map<string, map<string, Box> > &func_dep_regions) {

    map<string, int> num_consumers;
    for(auto &f: regions)
        num_consumers[f.first] = 0;

    for(auto &f: regions) {
        map<string, Box> &prods = func_dep_regions[f.first];
        for(auto &p: prods) {
            if (regions.find(p.first) != regions.end())
                num_consumers[p.first] += 1;
        }
    }

    vector<Function> outs;
    for(auto &f: num_consumers)
        if (f.second  == 0) {
            outs.push_back(env[f.first]);
        }

    // This assumption should hold for now
    assert(outs.size() == 1);

    // Realization order
    vector<string> order = realization_order(outs, env);

    long long working_set_size = 0;
    long long curr_size = 0;

    map<string, long long> func_sizes;
    for(auto &f: regions) {
        long long size = region_size(f.first, f.second, env);
        if (size < 0)
            return -1;
        else
            func_sizes[f.first] = size;
    }

    for(auto &f: order) {
        if (regions.find(f) != regions.end()) {
            // Skip functions that have been inlined
            curr_size += func_sizes[f];
        }
        working_set_size = std::max(curr_size, working_set_size);
        map<string, Box> &prods = func_dep_regions[f];
        for(auto &p: prods) {
            if (num_consumers.find(p.first) != num_consumers.end())
                num_consumers[p.first] -= 1;
            if (num_consumers[p.first] == 0) {
                curr_size -= func_sizes[p.first];
                assert(curr_size >= 0);
            }
        }
    }

    return working_set_size;
    // Computing total size
    /*
    int total_size = 0;
    for(auto &f: funcs) {
        int size = region_size(f.first, f.second, env);
        if (size < 0)
            return -1;
        else
            total_size += size;
    }
    return total_size;
    */
}

long long data_from_group(string func, map<string, Function> &env,
                          map<string, map<string, int> > &func_calls,
                          map<string, long long> &func_sizes,
                          vector<string> &prods) {
    long long data = 0;
    for (auto&c: func_calls[func]) {
        if (std::find(prods.begin(), prods.end(), c.first) != prods.end()) {
            int num_calls = c.second;
            int prod_size_per_ele = get_func_out_size(env[c.first]);
            data += std::min(num_calls * func_sizes[func], func_sizes[c.first])
                    * prod_size_per_ele;
        }
    }
    return data;
}

long long region_cost_inline(string func, vector<string> &inline_reg,
                             map<string, map<string, int> > &func_calls,
                             map<string, pair<long long, long long> > &func_cost) {

    map<string, int> calls;
    for (auto&c: func_calls[func])
        calls[c.first] = c.second;

    // Find the total number of calls to functions outside the inline region
    bool fixpoint = false;
    long long total_cost = 0;
    while(!fixpoint) {
        fixpoint = true;
        for (auto& p: inline_reg) {
            if (calls.find(p) != calls.end()) {
                long long num_calls = calls[p];
                assert(num_calls > 0);
                long long op_cost = func_cost[p].first;
                total_cost += num_calls * op_cost;
                for (auto &c: func_calls[p]) {
                    if (calls.find(c.first) != calls.end())
                        calls[c.first] += num_calls * c.second;
                    else
                        calls[c.first] = num_calls * c.second;
                }
                calls.erase(p);
                fixpoint = false;
            }
        }
    }
    std::cout << func << " " << total_cost << std::endl;
    assert(total_cost >= 0);
    return total_cost;
}

long long region_cost(string func, Box &region,
                      map<string, pair<long long, long long> > &func_cost) {
    long long area = box_area(region);
    if (area < 0) {
        // Area could not be determined
        return -1;
    }
    long long op_cost = func_cost[func].first;

    long long cost = area * (op_cost);
    assert(cost >= 0);
    return cost;
}

long long region_cost(map<string, Box> &regions,
                      map<string, pair<long long, long long> > &func_cost) {

    long long total_cost = 0;
    for(auto &f: regions) {
        long long cost = region_cost(f.first, f.second, func_cost);
        if (cost < 0) {
            return -1;
        }
        else
            total_cost += cost;
    }
    assert(total_cost >= 0);
    return total_cost;
}

long long overlap_cost(string cons, Function prod, vector<map<string, Box> > &overlaps,
                       map<string, pair<long, long> > &func_cost, int dim=-1) {
    long long total_area = 0;
    assert((int)overlaps.size() > dim);
    for (unsigned int d = 0; d < overlaps.size(); d++) {
        // Overlap area
        if (overlaps[d].find(prod.name()) != overlaps[d].end()
                && (dim==-1 || dim == (int)d) ) {
            long long area = box_area(overlaps[d][prod.name()]);
            if (area >= 0)
                total_area += area;
            else
                // Area could not be determined
                return -1;
        }
    }
    long long op_cost = func_cost[prod.name()].first;
    long long overlap_cost = total_area * (op_cost);
    return overlap_cost;
}

long long overlap_cost(string cons, vector<Function> &prods,
                       vector<map<string, Box> > &overlaps,
                       map<string, pair<long, long> > &func_cost,
                       int dim=-1) {

    long long total_cost = 0;
    for(auto& p: prods) {
        if (p.name()!=cons) {
            long long cost = overlap_cost(cons, p, overlaps, func_cost, dim);
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
                  map<string, vector<string> > &inlines, string func) {
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

int get_extent_estimate(Function &f, map<string, Box> &bounds, int dim) {

    vector<string> vars = f.args();
    int estimate = -1;
    for (auto &b: f.schedule().bounds())
        if (b.var == vars[dim]) {
            const IntImm * bmin = b.min.as<IntImm>();
            const IntImm * bextent = b.extent.as<IntImm>();
            estimate = bmin->value + bextent->value - 1;
        }

    if (bounds.find(f.name()) != bounds.end()) {
        Interval &I = bounds[f.name()][dim];
        int extent = get_extent(I);
        if (extent > 0 && estimate > 0)
            estimate = std::min(estimate, extent);
        else
            estimate = extent;
    }

    return estimate;
}

int get_min_estimate(Function &f, map<string, Box> &bounds, int dim) {

    vector<string> vars = f.args();
    int estimate = std::numeric_limits<int>::max();
    for (auto &b: f.schedule().bounds())
        if (b.var == vars[dim]) {
            const IntImm * bmin = b.min.as<IntImm>();
            estimate = bmin->value;
        }

    if (bounds.find(f.name()) != bounds.end()) {
        Interval &I = bounds[f.name()][dim];
        int lower = get_min(I);
        estimate = std::max(estimate, lower);
    }

    return estimate;
}

pair<int, int> get_bound_estimates(Function &f, map<string, Box> &bounds,
                                   int dim) {
    vector<string> vars = f.args();
    int est_lower = std::numeric_limits<int>::max();
    int est_upper = std::numeric_limits<int>::min();
    for (auto &b: f.schedule().bounds())
        if (b.var == vars[dim]) {
            const IntImm * bmin = b.min.as<IntImm>();
            const IntImm * bextent = b.extent.as<IntImm>();
            est_lower = bmin->value;
            est_upper = bmin->value + bextent->value - 1;
        }

    if (bounds.find(f.name()) != bounds.end()) {
        Interval &I = bounds[f.name()][dim];
        pair<int, int> b = get_bound(I);
        est_lower = std::max(est_lower, b.first);
        est_upper = std::max(est_upper, b.second);
    }

    return make_pair(est_lower, est_upper);
}

void disp_func_calls(map<string, map<string, int> > &func_calls) {
    for (auto &f: func_calls) {
        std::cout << "Calls in function " << f.first << std::endl;
        for (auto &c: f.second)
            std::cout << c.first << " " << c.second << std::endl;
    }
}

struct Partitioner {

    struct Option {
        // Option encodes the possibility of the prod_group being merged with
        // the cons_group at the granularity of the tile given by tile_sizes
        string prod_group;
        string cons_group;
        // Tile sizes of along dimensions of the output of the child group
        // A tile size of -1 indicates no tiling along the dimension
        vector<int> tile_sizes;
        // A score indicating the benefit of the option
        float benefit;
        // Estimate of extra aritmetic introduced
        float redundant_work;
        // Estimate of mem accesses saved
        float saved_mem;

        Option() {
            prod_group = "";
            cons_group = "";
            benefit = -1;
            redundant_work = -1;
            saved_mem = -1;
        }
    };

    // Levels that are targetted by the grouping algorithm
    enum Level {INLINE, FAST_MEM};

    struct GroupSched {
        vector<int> tile_sizes;
        // A score indicating the benefit of the scheduling choice
        float benefit;
        // Estimate of extra aritmetic introduced
        float redundant_work;
        // Estimate of mem accesses saved
        float saved_mem;

        GroupSched() {
            benefit = 0;
            redundant_work = 0;
            saved_mem = 0;
        }
    };

    struct MachineParams {
        int parallelism;
        int vec_len;
        long long fast_mem_size;
        int balance;
    };

    map<string, Box> &pipeline_bounds;
    map<string, vector<string> > &inlines;
    DependenceAnalysis &analy;
    map<string, pair<long long, long long> > &func_cost;

    map<string, vector<Function> > groups;
    map<string, GroupSched> group_sched;
    map<string, set<string> > children;

    map<string, vector<int> > func_dim_estimates;
    map<string, long long > func_op;
    map<string, long long > func_size;
    map<string, map<string, int> > func_calls;

    map<pair<string, string>, Option> option_cache;

    MachineParams arch_params;

    Partitioner(map<string, Box> &_pipeline_bounds,
                map<string, vector<string> > &_inlines, DependenceAnalysis &_analy,
                map<string, pair<long long, long long> > &_func_cost):
                pipeline_bounds(_pipeline_bounds), inlines(_inlines),
                analy(_analy), func_cost(_func_cost) {

        // Place each function in its own group
        for (auto &kv: analy.env) {
            vector<Dim> &dims = kv.second.schedule().dims();
            if (dims.size() > 0)
                groups[kv.first].push_back(kv.second);
        }

        // Find consumers of each function relate groups with their children
        for (auto &kv: analy.env) {
            map<string, Function> calls = find_direct_calls(kv.second);
            for (auto &c: calls)
                if (c.first != kv.first)
                    children[c.first].insert(kv.first);
        }

        //disp_children(children);

        // Add inlined functions to their child group
        for (auto &in: inlines) {
            for (auto &dest: in.second) {
                if (groups.find(dest) == groups.end()) {
                    for (auto &g: groups)
                        for (auto &m: g.second)
                            if (m.name() == dest)
                                dest = g.first;
                }
                merge_groups(in.first, dest);
            }
        }

        for (auto &g: groups) {
            Function output = analy.env[g.first];
            const vector<string> &args = output.args();

            GroupSched sched;

            // From the outer to the inner most argument
            for (int i = (int)args.size() - 1; i >= 0; i --)
                sched.tile_sizes.push_back(-1);

            group_sched[g.first] = sched;
        }

        // Build a table of num_calls to each internal Halide function when
        // each function
        for (auto &f: analy.env) {
            map<string, int> num_calls;
            FindCallArgs find;
            f.second.accept(&find);
            for(auto &c: find.calls) {
                num_calls[c.first] = c.second.size();
            }

            for (auto &u: f.second.updates()) {
                FindCallArgs find_update;

                for (auto &e: u.values)
                    e.accept(&find_update);
                for (auto &arg: u.args)
                    arg.accept(&find_update);

                if (u.domain.defined()) {
                    Box b;
                    for (auto &rvar: u.domain.domain()) {
                        b.push_back(Interval(simplify(rvar.min),
                                    simplify(rvar.min + rvar.extent - 1)));
                    }
                    long long area = box_area(b);

                    if (area != -1) {
                        for(auto &c: find.calls) {
                            num_calls[c.first] -= c.second.size();
                            num_calls[c.first] += c.second.size() * area;
                        }
                    }
                }
            }

            func_calls[f.first] = num_calls;
        }

        disp_func_calls(func_calls);

        for (auto &f: analy.env) {
            const vector<string> &args = f.second.args();
            vector<int> dim_estimates;
            long long size = 1;
            for (unsigned int i = 0; i < args.size(); i++) {
                int estimate = get_extent_estimate(f.second,
                                                   pipeline_bounds, i);
                dim_estimates.push_back(estimate);
                if (estimate != -1 && size != -1)
                    size *= estimate;
                else
                    size = -1;
            }
            long long work = size;
            if(size != -1) {
                work = func_cost[f.first].first * work;
            }
            func_op[f.first] = work;
            func_size[f.first] = size;
            func_dim_estimates[f.first] = dim_estimates;

        }

        // Initialize machine params
        arch_params.parallelism = 8;
        arch_params.vec_len = 8;
        arch_params.balance = 10;
        arch_params.fast_mem_size = 32 * 1024 * 8;
        // L1 = 32K
        // L2 = 256K
        // L3 = 8192K
    }

    void merge_groups(string cand_group, string child_group) {
        assert(groups.find(child_group) != groups.end());
        vector<Function> cand_funcs = groups[cand_group];

        groups.erase(cand_group);
        group_sched.erase(cand_group);

        groups[child_group].insert(groups[child_group].end(),
                cand_funcs.begin(), cand_funcs.end());

        // Update the children mapping
        children.erase(cand_group);
        for (auto &f: children) {
            set<string> &cons = f.second;
            if (cons.find(cand_group) != cons.end()) {
                cons.erase(cand_group);
                cons.insert(child_group);
            }
        }
    }

    void merge_group_all_children(string cand_group) {

        set<string> cand_group_children = children[cand_group];
        for (auto &cg: cand_group_children) {
            assert(groups.find(cg) != groups.end());
            vector<Function> cand_funcs = groups[cand_group];

            groups[cg].insert(groups[cg].end(),
                    cand_funcs.begin(), cand_funcs.end());
        }
        groups.erase(cand_group);
        group_sched.erase(cand_group);

        // Update the children mapping
        for (auto &f: children) {
            set<string> &cons = f.second;
            if (cons.find(cand_group) != cons.end()) {
                cons.erase(cand_group);
                cons.insert(cand_group_children.begin(),
                            cand_group_children.end());
            }
        }
        children.erase(cand_group);
    }

    void disp_grouping() {
        for (auto& g: groups) {
            std::cout << "Group " <<  g.first  << " :"<< std::endl;
            for (auto& m: g.second)
                std::cout << m.name() << std::endl;
            std::cout << std::endl;
        }
    }

    void disp_costs() {
        for (auto &f: analy.env) {
            std::cout << f.first << " Cost " <<
                func_cost[f.first].first  << " " <<
                func_cost[f.first].second  <<
                std::endl;
        }
    }

    void disp_option(Option &opt) {
        std::cout << opt.prod_group << "->" << opt.cons_group << std::endl;
        std::cout << "[";
        for (unsigned int i = 0; i < opt.tile_sizes.size(); i++) {
            std::cout << opt.tile_sizes[i] << ",";
        }
        std::cout << "]" << std::endl;
        std::cout << "Benefit:" << opt.benefit << std::endl;
        std::cout << "Redundant work:" << opt.redundant_work << std::endl;
        std::cout << "Memory accesses saved:" << opt.saved_mem << std::endl;
    }

    Option choose_candidate(const vector< pair<string, string > > &cand_pairs);
    pair<float, vector<Option> >
        choose_candidate_inline(const vector< pair<string, string > > &cand_pairs);
    void group(Partitioner::Level level);
    void clear_schedules();
    void initialize_groups_fast_mem();
    void initialize_groups_inline();
    void update_function_costs();
    void evaluate_option(Option &opt, Partitioner::Level level);
    void reorder_for_input_locality();
    pair<float, float> evaluate_reuse(string, vector<string> &group_inputs,
                                      vector<int> &tile_sizes, bool check_cache);
};

void Partitioner::clear_schedules() {
    for (auto &s: group_sched) {
        // Do not clear the benefit from inlining phase
        s.second.benefit = s.second.saved_mem * arch_params.balance;
        s.second.redundant_work = 0;
        s.second.saved_mem = 0;

        for (unsigned int i = 0; i < s.second.tile_sizes.size(); i++)
            s.second.tile_sizes[i] = -1;
    }
}

void Partitioner::initialize_groups_inline() {
    for (auto &g: groups) {
        Option opt;
        opt.prod_group = "";
        opt.cons_group = g.first;

        Function output = analy.env[g.first];
        const vector<string> &args = output.args();

        for (unsigned int i = 0; i < args.size(); i++)
            opt.tile_sizes.push_back(1);

        evaluate_option(opt, Partitioner::INLINE);

        GroupSched sched;
        sched.saved_mem = opt.saved_mem;
        sched.redundant_work = opt.redundant_work;
        sched.benefit = opt.benefit;
        sched.tile_sizes = opt.tile_sizes;

        group_sched[g.first] = sched;
    }
}

void Partitioner::initialize_groups_fast_mem() {
    option_cache.clear();
    clear_schedules();
    update_function_costs();
}

void Partitioner::update_function_costs() {
    for (auto &g: groups) {
        vector<string> prod_funcs;
        for (auto &f: g.second)
            if (f.name() != g.first)
                prod_funcs.push_back(f.name());

        long long work_per_ele = region_cost_inline(g.first, prod_funcs,
                                                    func_calls, func_cost);
        assert(work_per_ele >= 0);
        func_cost[g.first].first += work_per_ele;
    }
    for (auto &f: analy.env) {
        const vector<string> &args = f.second.args();
        long long size = 1;
        for (unsigned int i = 0; i < args.size(); i++) {
            long long  estimate = get_extent_estimate(f.second,
                                                      pipeline_bounds, i);
            if (estimate != -1 && size != -1)
                size *= estimate;
            else
                size = -1;
        }
        long long work = size;
        if(size != -1) {
            work = func_cost[f.first].first * work;
        }
        func_op[f.first] = work;
    }
}

void Partitioner::group(Partitioner::Level level) {
    // Partition the pipeline by iteratively merging groups until a fixpoint
    bool fixpoint = false;
    while(!fixpoint) {
        fixpoint = true;
        vector< pair<string, string> > cand;
        for (auto &g: groups) {
            if (children.find(g.first) != children.end()) {
                // TODO be careful about inputs and outputs to the pipeline
                int num_children = children[g.first].size();
                // Find all the groups which have a single child
                if (num_children == 1 && level == Partitioner::FAST_MEM) {
                    cand.push_back(make_pair(g.first,
                                             *children[g.first].begin()));
                } else if(num_children > 0  && level == Partitioner::INLINE) {
                    cand.push_back(make_pair(g.first, ""));
                }
            }
        }
        for (auto &p: cand) {
            std::cout << "[" << p.first << "," <<  p.second << "]";
        }
        std::cout << std::endl;

        vector<pair<string, string> > invalid_keys;
        if (level == Partitioner::INLINE) {
            pair<float, vector<Option> > best;
            best = choose_candidate_inline(cand);
            if (best.first >= 0) {
                string prod = best.second[0].prod_group;

                std::cout << "Choice Inline:" << std::endl;
                std::cout << prod << std::endl;

                for (auto &o: best.second)
                    assert(o.prod_group == prod);

                assert(best.second.size() == children[prod].size());

                analy.env[prod].schedule().store_level().var = "";
                analy.env[prod].schedule().compute_level().var = "";

                int i = 0;
                for (auto &c: children[prod]) {
                    assert(best.second[i].cons_group == c);

                    inlines[prod].push_back(c);
                    GroupSched sched;

                    sched.tile_sizes = best.second[i].tile_sizes;
                    sched.benefit = best.second[i].benefit;
                    sched.redundant_work = best.second[i].redundant_work;
                    assert(best.second[i].saved_mem >= 0);
                    sched.saved_mem = best.second[i].saved_mem;
                    group_sched[c] = sched;

                    for (auto& opt: option_cache) {
                        if (opt.first.first == c ||
                                opt.first.second == c)
                            invalid_keys.push_back(opt.first);
                    }
                    i++;
                }
                merge_group_all_children(prod);
                fixpoint = false;
            }

        } else {
            Option best;
            best = choose_candidate(cand);
            if (best.benefit >= 0) {

                std::cout << "Choice Fuse:" << std::endl;
                std::cout << best.prod_group << " "
                          << best.cons_group << std::endl;
                std::cout << "[";
                for (auto s: best.tile_sizes)
                    std::cout << s << ",";
                std::cout << "]"  << std::endl;

                for (auto& opt: option_cache) {
                    if (opt.first.second == best.cons_group
                            || opt.first.first == best.cons_group)
                        invalid_keys.push_back(opt.first);
                }

                GroupSched sched;
                sched.tile_sizes = best.tile_sizes;
                sched.benefit = best.benefit;
                sched.redundant_work = best.redundant_work;
                assert(best.saved_mem >= 0);
                sched.saved_mem = best.saved_mem;

                group_sched[best.cons_group] = sched;

                merge_groups(best.prod_group, best.cons_group);
                fixpoint = false;
            }
        }

        // Invalidate the option cache
        for (auto& key: invalid_keys)
            option_cache.erase(key);
    }
}

void disp_regions(map<string, Box> &regions) {
    for (auto& reg: regions) {
        std::cout << reg.first;
        disp_box(reg.second);
        std::cout << std::endl;
    }
}

map<string, int> get_dim_estimates(string f, map<string, Box> &pipeline_bounds,
                                   map<string, Function> &env) {
    map<string, int> dim_estimates;
    const vector<string> &args = env[f].args();
    vector<Dim> &dims = env[f].schedule().dims();
    for (unsigned int i = 0; i < args.size(); i++) {
        int estimate = get_extent_estimate(env[f], pipeline_bounds, i);
        dim_estimates[dims[i].var] = estimate;
    }
    // Add the estimates for RDom dimensions
    for (auto &u: env[f].updates()) {
        if (u.domain.defined()) {
            Box b;
            for (auto &rvar: u.domain.domain()) {
                Interval I = Interval(simplify(rvar.min),
                                       simplify(rvar.min + rvar.extent - 1));
                dim_estimates[rvar.var] = get_extent(I);
            }
        }
    }
    return dim_estimates;
}

void Partitioner::evaluate_option(Option &opt, Partitioner::Level l) {

    std::cout << std::endl;
    std::cout << "Evaluating benefit " << opt.prod_group << "->"
                                       << opt.cons_group << ":" << std::endl;
    disp_option(opt);

    map<string, Box> conc_reg;

    // For each function in the prod and child group that is not the
    // output figure out the concrete bounds

    vector<string> prod_funcs;
    if (opt.prod_group != "") {
        for (auto &f: groups[opt.prod_group])
            if (!f.is_lambda())
                prod_funcs.push_back(f.name());
    }
    for (auto &f: groups[opt.cons_group]) {
        if (f.name() != opt.cons_group && !f.is_lambda())
            prod_funcs.push_back(f.name());
    }

    vector<pair<int, int> > bounds;
    vector<bool> eval;

    const vector<string> &args = analy.env[opt.cons_group].args();
    assert(opt.tile_sizes.size() == args.size());

    vector<int> &dim_estimates_cons = func_dim_estimates[opt.cons_group];

    long long out_size = 1;
    for (unsigned int i = 0; i < args.size(); i++) {
        if (dim_estimates_cons[i] == -1) {
            // This option cannot be evaluated so discaring the option
            opt.benefit = -1;
            opt.redundant_work = -1;
            return;
        }
        else {
            out_size *= dim_estimates_cons[i];
        }
    }

    Box cons_box;
    long long tile_size = 1;
    for (unsigned int i = 0; i < args.size(); i++) {
        if (opt.tile_sizes[i] != -1) {
            // Check if the bounds allow for tiling with the given tile size
            // Ensure atleast 2 tiles
            if (dim_estimates_cons[i] >= 2 * opt.tile_sizes[i]) {
                bounds.push_back(make_pair(0, opt.tile_sizes[i] - 1));
                tile_size = tile_size * (opt.tile_sizes[i]);
                cons_box.push_back(Interval(0, opt.tile_sizes[i] - 1));
            }
            else {
                // If the dimension is too small do not tile it and set the
                // extent of the bounds to that of the dimension estimate
                opt.tile_sizes[i] = -1;
                bounds.push_back(make_pair(0, dim_estimates_cons[i] - 1));
                tile_size = tile_size * (dim_estimates_cons[i]);
                cons_box.push_back(Interval(0, dim_estimates_cons[i] - 1));
            }
        }
        else {
            bounds.push_back(make_pair(0, dim_estimates_cons[i] - 1));
            tile_size = tile_size * (dim_estimates_cons[i]);
            cons_box.push_back(Interval(0, dim_estimates_cons[i] - 1));
        }

        eval.push_back(true);
    }

    // Count the number of tiles
    long long estimate_tiles = 1;
    float partial_tiles = 1;
    for (unsigned int i = 0; i < args.size(); i++) {
        if (opt.tile_sizes[i] != -1) {
            estimate_tiles *= std::ceil((float)dim_estimates_cons[i]/opt.tile_sizes[i]);
            partial_tiles *= (float)dim_estimates_cons[i]/opt.tile_sizes[i];
        }

    }

    conc_reg = analy.concrete_dep_regions(opt.cons_group, eval, bounds);

    //disp_regions(conc_reg);

    // Cost model

    // We currently assume a two level memory model. The fast_mem_size field in
    // the arch parameters gives the size of the fast memory. Additionally, the
    // ratio of load from fast memory vs slow memory is encoded in the machine
    // parameters.

    // Computing the cost the function regions required for the group that are
    // not computed within the group are considered as loads from slow memory.
    // We compute the size of the intermediate buffers that are required to
    // compute the output of the group.

    // inter_s = size of the intermediates in the fused group
    // M = fast memory size
    // s_c = the cost of loading from slow memory
    // f_c = the cost of loading from fast memory
    // op_c = the cost of computing an op

    // The benefit of an option is the reduction in the number of operations
    // that read/write to slow memory and the benefit is calculated per tile
    //
    // if inter_s fits in fast memory then
    //    inter_s * s_c - (inter_s * f_c + (redundant_ops) * op_c)
    //    => inter_s * (s_c - f_c) - (redundant_ops) * op_c
    // else
    //    hit = max(2M - inter_s, 0) assuming LRU
    //    inter_s * s_c - (hit * f_c + (inter_s - hit) * s_c + (redundant_ops)
    //                     * op_c)
    //    => hit * (s_c - f_c) - (redundant_ops) * op_c

    // disp_regions(conc_reg);
    map<string, Box> mem_reg;
    map<string, Box> prod_comp;

    // Determine size of intermediates

    // Do not count inlines while accounting for intermediate storage when
    // grouping for fast mem
    long long original_work = 0;
    for (auto &f: prod_funcs) {
        if (inlines.find(f) == inlines.end() || (l == Partitioner::INLINE)) {
            mem_reg[f] = conc_reg[f];
            prod_comp[f] = conc_reg[f];
            original_work += func_op[f];
        }
    }

    mem_reg[opt.cons_group] = cons_box;

    vector<Function> prods;
    for (auto &f: prod_funcs)
        prods.push_back(analy.env[f]);

    //for (auto &o: conc_overlaps)
    //    disp_regions(o);

    long long work_per_tile = 0;
    long long inter_s = 0;

    if (l == Partitioner::INLINE) {
        work_per_tile = region_cost_inline(opt.cons_group, prod_funcs,
                                           func_calls, func_cost);
    } else {
        work_per_tile = region_cost(prod_comp, func_cost);
        assert(work_per_tile >= 0);
        inter_s = region_size(mem_reg, analy.env, analy.func_dep_regions);
    }

    long long saved_mem = 0;

    vector<string> out_of_cache_prods;
    for (auto &p: prod_funcs) {
        if(func_size[p] > arch_params.fast_mem_size || l == Partitioner::INLINE)
            out_of_cache_prods.push_back(p);
    }

    for (auto &f: prod_funcs) {
        if (func_op[f] != -1) {
            long long data = data_from_group(f, analy.env, func_calls,
                                             func_size, out_of_cache_prods);
            saved_mem += data;
        }
    }

    //float total_work = work_per_tile * partial_tiles;

    // This is more accurate since partial tiles are handled by shifting
    // and computing a full tile.
    float total_work = work_per_tile * estimate_tiles;

    if (saved_mem != -1) {
        long long data = data_from_group(opt.cons_group, analy.env,
                                         func_calls, func_size,
                                         out_of_cache_prods);
        saved_mem += data;
    }

    disp_regions(prod_comp);

    std::cout << "Work per tile:" << work_per_tile << std::endl;
    std::cout << "Num tiles:" << estimate_tiles << std::endl;
    std::cout << "Partial tiles:" << partial_tiles << std::endl;
    std::cout << "Total work:" << total_work << std::endl;
    std::cout << "Original work:" << original_work << std::endl;
    std::cout << "Saved mem:" << saved_mem << std::endl;

    std::cout << "Intermediate size:" << inter_s << std::endl;
    std::cout << "Redundant work:" <<
                    (total_work - original_work) << std::endl;

    opt.redundant_work = total_work - original_work;
    opt.saved_mem = saved_mem;

    if (prod_comp.size() > 0)
        assert(total_work > 0);

    if (l == Partitioner::INLINE) {
        opt.benefit = (saved_mem) * (arch_params.balance)
                                  - opt.redundant_work;
    } else {
        if (inter_s <= arch_params.fast_mem_size) {
            opt.benefit = (saved_mem) * (arch_params.balance)
                           - opt.redundant_work;
        }
        else if (inter_s <= 2 * arch_params.fast_mem_size) {
            float hit = (float)std::max(2 * arch_params.fast_mem_size - inter_s, 0LL)/inter_s;
            float loads_saved = hit * saved_mem;
            opt.benefit = loads_saved * (arch_params.balance)
                          - opt.redundant_work;
        }
    }

    std::cout << "Estimated benefit:" << opt.benefit << std::endl;

    if ((arch_params.parallelism > estimate_tiles) && opt.prod_group != "") {
        // Option did not satisfy the parallelism constraint
        opt.benefit = -1;
    }

    if (opt.prod_group != "") {
        std::cout << std::endl << "Producer group:" << std::endl;
        for (auto &f: groups[opt.prod_group])
            std::cout << f.name() << std::endl;
        std::cout << "Saved mem:" << group_sched[opt.prod_group].saved_mem
            << std::endl;
        std::cout << "Redundant work:" << group_sched[opt.prod_group].redundant_work
            << std::endl;
        std::cout << "Producer benefit:" << group_sched[opt.prod_group].benefit << std::endl;
    }

    std::cout << std::endl << "Consumer group:" << std::endl;
    for (auto &f: groups[opt.cons_group])
        std::cout << f.name() << std::endl;
    std::cout << "Saved mem:" << group_sched[opt.cons_group].saved_mem
              << std::endl;
    std::cout << "Redundant work:" << group_sched[opt.cons_group].redundant_work
              << std::endl;
    std::cout << "Consumer benefit:" << group_sched[opt.cons_group].benefit
              << std::endl;

    if (opt.prod_group != "")  {
        //assert(group_sched[opt.cons_group].benefit >= 0 &&
        //        group_sched[opt.prod_group].benefit >= 0 );

        assert(group_sched[opt.cons_group].saved_mem >= 0 &&
                group_sched[opt.prod_group].saved_mem >= 0 );

        if (group_sched[opt.cons_group].benefit +
                group_sched[opt.prod_group].benefit > opt.benefit) {
            opt.benefit = -1;
        }
    }
    std::cout << std::endl << "Final benefit:" << opt.benefit << std::endl;
}

pair<float, vector<Partitioner::Option> >
    Partitioner::choose_candidate_inline(
                    const vector< pair<string, string> > &cand_pairs) {

    pair<float, vector<Partitioner::Option> > best;
    best.first = -1;
    for (auto &p: cand_pairs) {
        // Compute the aggregate benefit for inlining into all the children
        float overall_benefit = 0;
        vector<Option> options;
        for (auto &c: children[p.first]) {

            // Get the output function of the child group
            Function output = analy.env[c];
            const vector<string> &args = output.args();

            Option cand_opt;
            cand_opt.prod_group = p.first;
            cand_opt.cons_group = c;

            // Check if the pair has been evaluated before
            pair<string, string> key = make_pair(p.first, c);
            if (option_cache.find(key) != option_cache.end()) {

                cand_opt = option_cache[key];

            } else {
                // If the pair has not been evaluated before evaluate
                // the option with tile size 1 in all dimensions

                for (unsigned int i = 0; i < args.size(); i++)
                    cand_opt.tile_sizes.push_back(1);

                evaluate_option(cand_opt, Partitioner::INLINE);

                // Cache the result of the evaluation for the pair
                option_cache[key] = cand_opt;
            }

            if (cand_opt.benefit < 0) {
                overall_benefit = -1;
                break;
            } else {
                options.push_back(cand_opt);
                overall_benefit += cand_opt.benefit;
            }
        }

        if (best.first < overall_benefit) {
            assert(children[p.first].size() == options.size());
            best.first = overall_benefit;
            best.second = options;
        }

    }
    return best;
}

Partitioner::Option Partitioner::choose_candidate(
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
    // needs to account for early frees and the high watermark of intermediate
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
    // latency of the entire pipeline is 3 tiles. So either the number of tiles
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

    vector<Option> options;
    vector<int> size_variants = {256, 128, 64, 32, 16, 8, 4};

    Option best_opt;

    for (auto &p: cand_pairs) {
        pair<string, string> key = make_pair(p.first, p.second);
        Option cand_best_opt;
        // Check if the pair has been evaluated before
        if (option_cache.find(key) != option_cache.end()) {
            //std::cout << "Hit:" << p.first << "," << p.second << std::endl;
            cand_best_opt = option_cache[key];
            if (best_opt.benefit < cand_best_opt.benefit)
                best_opt = cand_best_opt;
            continue;
        }

        // If the pair has not been evaluated before create all the options
        // and evaluate them

        // Get the output function of the child group
        Function output = analy.env[p.second];
        const vector<string> &args = output.args();

        bool invalid = false;
        vector<int> &dim_estimates_prod = func_dim_estimates[p.first];

        const vector<string> &args_prod = analy.env[p.first].args();
        for (unsigned int i = 0; i < args_prod.size(); i++) {
            if (dim_estimates_prod[i] == -1) {
                // This option cannot be evaluated so discaring the option
                invalid = true;
            }
        }

        cand_best_opt.prod_group = p.first;
        cand_best_opt.cons_group = p.second;

        if (!invalid) {
            // Find the dimensions with zero reuse/redundant work
            vector<float> reuse;
            for (unsigned int i = 0; i < args.size(); i++)
                reuse.push_back(-1);
            for (unsigned int i = 0; i < args.size(); i++) {
                Option opt;
                opt.prod_group = p.first;
                opt.cons_group = p.second;
                for (unsigned int j = 0; j < args.size(); j++) {
                    if (i!=j)
                        opt.tile_sizes.push_back(-1);
                    else
                        opt.tile_sizes.push_back(1);
                }
                evaluate_option(opt, Partitioner::FAST_MEM);
                reuse[i] = opt.redundant_work;
            }
            std::cout << "Analyzing dims for reuse" << std::endl;
            for (unsigned int i = 0; i < args.size(); i++) {
                std::cout << args[i] << " Reuse/Redundant Work " << reuse[i]
                          << std::endl;
            }


            // From the outer to the inner most argument
            //for (int i = (int)args.size() - 1; i >= 1; i--) {
            for (int i = (int)args.size() - 1; i >= 0; i--) {
                for (auto s: size_variants) {
                    Option opt;
                    opt.prod_group = p.first;
                    opt.cons_group = p.second;

                    for (int j = 0; j < i; j++) {
                        if (reuse[j] > 0 || j == 0)
                            opt.tile_sizes.push_back(-1);
                        else
                            opt.tile_sizes.push_back(1);
                    }

                    for (unsigned int j = i; j < args.size(); j++) {
                        int curr_size;
                        if (reuse[j] > 0 || j == 0)
                            curr_size = s;
                        else
                            curr_size = 1;

                        if (j == 0)
                            opt.tile_sizes.push_back(std::max(curr_size, 64));
                        else
                            opt.tile_sizes.push_back(curr_size);
                    }

                    if (i == 0)
                        evaluate_option(opt, Partitioner::FAST_MEM);
                    else
                        evaluate_option(opt, Partitioner::FAST_MEM);


                    if (cand_best_opt.benefit < opt.benefit)
                        cand_best_opt = opt;
                }
            }
        }

        // Cache the result of the evaluation for the pair
        option_cache[key] = cand_best_opt;
        if (best_opt.benefit < cand_best_opt.benefit)
            best_opt = cand_best_opt;
    }
    return best_opt;
}

pair<float, float>
    Partitioner::evaluate_reuse(string group, vector<string> &group_inputs,
                                vector<int> &tile_sizes, bool check_cache) {

    const vector<string> &args = analy.env[group].args();
    vector<pair<int, int> > bounds;
    vector<bool> eval;

    vector<int> &dim_estimates = func_dim_estimates[group];
    Box cons_box;

    long long tile_size = 1;
    for (unsigned int i = 0; i < args.size(); i++) {
        if (tile_sizes[i] != -1) {
            // Check if the bounds allow for tiling with the given tile size
            if (dim_estimates[i] >= tile_sizes[i]) {
                bounds.push_back(make_pair(0, tile_sizes[i] - 1));
                cons_box.push_back(Interval(0, tile_sizes[i] -1));
                tile_size = tile_size * (tile_sizes[i]);
            }
            else {
                // If the dimension is too small do not tile it and set the
                // extent of the bounds to that of the dimension estimate
                tile_sizes[i] = -1;
                bounds.push_back(make_pair(0, dim_estimates[i] - 1));
                cons_box.push_back(Interval(0, dim_estimates[i] - 1));
                tile_size = tile_size * (dim_estimates[i]);
            }
        }
        else {
            bounds.push_back(make_pair(0, dim_estimates[i] - 1));
            cons_box.push_back(Interval(0, dim_estimates[i] - 1));
            tile_size = tile_size * (dim_estimates[i]);
        }
        eval.push_back(true);
    }

    // Count the number of tiles
    long long estimate_tiles = 1;
    for (unsigned int i = 0; i < args.size(); i++) {
        if (tile_sizes[i] != -1)
            estimate_tiles *= std::ceil((float)dim_estimates[i]/tile_sizes[i]);
    }

    map<string, Box> conc_reg =
        analy.concrete_dep_regions(group, eval, bounds);

    map<string, Box> group_mem_reg;
    map<string, Box> input_mem_reg;

    for (auto &m: groups[group]) {
        if (inlines.find(m.name()) == inlines.end()) {
            group_mem_reg[m.name()] = conc_reg[m.name()];
        }
    }

    group_mem_reg[group] = cons_box;

    float input_inter = 0;
    for (auto &f: group_inputs) {
        float size = region_size(f, conc_reg[f], analy.env);
        input_mem_reg[f] = conc_reg[f];
        if (size > -1) {
            input_inter += size;
        } else {
            input_inter = -1;
            break;
        }
    }

    float group_inter = region_size(group_mem_reg, analy.env,
                                    analy.func_dep_regions);

    float total_inter = 0;
    if (group_inter < 0 || input_inter < 0)
        return make_pair(-1, -1);
    else
        total_inter = group_inter + input_inter;

    float unit_input_data = 0;
    // Evalute the intermediate storage for computing in unit tiles
    if (tile_size > 1) {
        disp_regions(group_mem_reg);
        disp_regions(input_mem_reg);
        std::cout << "Config :[";
        for (auto &s: tile_sizes)
            std::cout << s << ",";
        std::cout << "]" << std::endl;
        std::cout << "Total intermediate size:" << total_inter  << std::endl;
        std::cout << "Input intermediate size:" << input_inter  << std::endl;
        vector<int> unit_sizes;

        for (unsigned int i = 0; i < args.size(); i++)
            unit_sizes.push_back(1);
        pair<float, float> unit = evaluate_reuse(group, group_inputs,
                                                 unit_sizes, true);
        assert(unit.first < 1);
        unit_input_data = unit.second;
    } else {
        std::cout << "Unit input size:" << input_inter << std::endl;
        unit_input_data = input_inter;
    }

    // Compute the reuse within a tile
    float reuse =  estimate_tiles * (unit_input_data * tile_size - input_inter);
    float realized_reuse = -1;
    if (check_cache) {
        if (total_inter <= arch_params.fast_mem_size) {
            realized_reuse = reuse;
        }
    } else {
        realized_reuse = reuse;
    }

    if (tile_size > 1)
        std::cout << "Reuse:" << realized_reuse << std::endl << std::endl;

    return make_pair(realized_reuse, input_inter);
}

void Partitioner::reorder_for_input_locality() {

    // Do this for each of the groups
    for(auto &g: groups) {

        bool tiled_grouping = false;
        for (auto &s: group_sched[g.first].tile_sizes) {
            if (s > 1) {
                tiled_grouping = true;
                break;
            }
        }

        if (tiled_grouping)
            continue;

        vector<string> group_inputs;
        set<string> group_mem;
        for(auto &f: g.second)
            group_mem.insert(f.name());

        for(auto &f: g.second) {
            FindAllCalls find;
            f.accept(&find);
            for(auto &c: find.calls) {
                if (group_mem.find(c) == group_mem.end())
                        group_inputs.push_back(c);
            }
        }

        std::cout << "Inputs for group " << g.first << ":" << std::endl;
        for(auto &in: group_inputs)
            std::cout << in << std::endl;
        std::cout << std::endl;

        // For the dimensions with reuse along multiple dimensions tile
        // the dimensions in such a way that the reuse is maximized and
        // the porition of inputs fit in fast memory
        vector<int> size_variants = {256, 128, 64, 32, 16, 8, 4};

        // If the pair has not been evaluated before create all the options
        // and evaluate them

        // Get the output function of the child group
        Function output = analy.env[g.first];
        const vector<string> &args = output.args();

        bool invalid = false;
        for(auto &in: group_inputs) {
            vector<int> &dim_estimates_prod = func_dim_estimates[in];

            const vector<string> &args_prod = analy.env[in].args();
            for (unsigned int i = 0; i < args_prod.size(); i++) {
                if (dim_estimates_prod[i] == -1) {
                    // This option cannot be evaluated so discaring the option
                    invalid = true;
                }
            }
        }

        if (!invalid) {
            // Find the dimensions with zero reuse/redundant work
            vector<float> reuse;
            for (unsigned int i = 0; i < args.size(); i++)
                reuse.push_back(-1);
            for (unsigned int i = 0; i < args.size(); i++) {

                vector<pair<int, int> > bounds;
                vector<bool> eval;

                vector<int> &dim_estimates = func_dim_estimates[g.first];

                for (unsigned int j = 0; j < args.size(); j++) {
                    if (j==i) {
                        bounds.push_back(make_pair(0, 1));
                    }
                    else {
                        bounds.push_back(make_pair(0, dim_estimates[i] - 1));
                    }
                    eval.push_back(true);
                }

                vector< map<string, Box> > conc_overlaps =
                    analy.concrete_overlap_regions(g.first, eval, bounds);

                float input_overlap = 0;
                for (auto &in: group_inputs) {
                    assert(conc_overlaps[i].find(in) != conc_overlaps[i].end());
                    float area = box_area(conc_overlaps[i][in]);
                    assert(area >= 0);
                    input_overlap += area * get_func_out_size(analy.env[in]);
                }
                reuse[i] = input_overlap;
            }

            std::cout << "Analyzing dims for locality" << std::endl;
            for (unsigned int i = 0; i < args.size(); i++) {
                std::cout << args[i] << " Reuse " << reuse[i]
                          << std::endl;
            }

            // From the outer to the inner most argument
            float best_reuse = 0;
            vector<int> best_tiling;
            for (int i = (int)args.size() - 1; i >= 0; i--) {
                for(auto &s: size_variants) {
                    vector<int> tile_sizes;

                    for (int j = 0; j < i; j++) {
                        if (reuse[j] > arch_params.fast_mem_size || j == 0)
                            tile_sizes.push_back(-1);
                        else
                            tile_sizes.push_back(1);
                    }

                    for (unsigned int j = i; j < args.size(); j++) {
                        int curr_size;
                        if (reuse[j] > arch_params.fast_mem_size || j == 0)
                            curr_size = s;
                        else
                            curr_size = 1;

                        if (j == 0)
                            tile_sizes.push_back(std::max(curr_size, 64));
                        else
                            tile_sizes.push_back(curr_size);
                    }

                    pair<float, float>  eval;
                    eval = evaluate_reuse(g.first, group_inputs, tile_sizes,
                                          true);
                    if (eval.first > best_reuse) {
                        best_reuse = eval.first;
                        best_tiling = tile_sizes;
                    }
                }
            }

            if (best_reuse > 0) {
                group_sched[g.first].tile_sizes = best_tiling;
            }
        }
    }
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

void disp_inlines(map<string, vector<string> > &inlines) {
    for (auto& in: inlines) {
        std::cout << in.first << "-> [";
        for (auto& c: in.second)
            std::cout << c << " ";
        std::cout << "]" << std::endl;
    }
}

map<string, vector<string>>
simple_inline(map<string, vector<const Call*>> &all_calls,
              map<string, vector<string> > &consumers,
              map<string, Function> &env) {
    map<string, vector<string> > inlines;
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
            inlines[fcalls.first].push_back(consumers[fcalls.first][0]);
            env[fcalls.first].schedule().store_level().var = "";
            env[fcalls.first].schedule().compute_level().var = "";
        }
        if (env[fcalls.first].is_boundary() || env[fcalls.first].is_lambda()) {
            assert(consumers[fcalls.first].size() == 1);
            inlines[fcalls.first].push_back(consumers[fcalls.first][0]);
            env[fcalls.first].schedule().store_level().var = "";
            env[fcalls.first].schedule().compute_level().var = "";
        }
    }
    return inlines;
}

// Helpers for schedule surgery

// Parallel
void parallelize_dim(Schedule &sched, int dim) {
    vector<Dim> &dims = sched.dims();
    dims[dim].for_type = ForType::Parallel;
}

void move_dim_to_outermost(Schedule &sched, int dim) {
    vector<Dim> &dims = sched.dims();
    dims.insert(dims.end() - 1, dims[dim]);
    dims.erase(dims.begin() + dim);
}

void move_dim_to_innermost(Schedule &sched, int dim) {
    vector<Dim> &dims = sched.dims();
    dims.insert(dims.begin(), dims[dim]);
    dims.erase(dims.begin() + dim + 1);
}

void move_dim_to_var(Schedule& sched, int dim, string var) {

    vector<Dim> &dims = sched.dims();
    int cand_dim = -1;
    for (unsigned int i = 0;  i < dims.size(); i++)
        if (dims[i].var == var)
            cand_dim = i;
    assert(cand_dim != -1);
    dims.insert(dims.begin() + cand_dim, dims[dim]);
    dims.erase(dims.begin() + dim);
}

void swap_dim(Schedule &sched, int dim1, int dim2) {

    vector<Dim> &dims = sched.dims();

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
void split_dim(Schedule &sched, int dim, int split_size,
               map<string, int> &dim_estimates, string prefix, bool partial) {

    vector<Dim> &dims = sched.dims();
    // Vectorization is not easy to insert in a Function object
    // have to revisit if this is the cleanest way to do it
    string old = dims[dim].var;
    string inner_name, outer_name, old_name;

    old_name = dims[dim].var;
    inner_name = old_name + "." + prefix + "." + "in";
    outer_name = old_name + "." + prefix + "." + "out";
    dims.insert(dims.begin() + dim, dims[dim]);
    dims[dim].var = inner_name;
    dims[dim+1].var = outer_name;
    dims[dim+1].pure = dims[dim].pure;
    dims[dim+1].for_type = dims[dim].for_type;

    // Add the split to the splits list
    Split split = {old_name, outer_name, inner_name, split_size,
                   false, partial, Split::SplitVar};
    sched.splits().push_back(split);

    // Updating the estimates to reflect the splitting
    dim_estimates[inner_name] = split_size;
    if (dim_estimates[old_name] != -1) {
        dim_estimates[outer_name] =
            std::ceil((float)dim_estimates[old_name]/split_size);
    } else {
        dim_estimates[inner_name] = -1;
    }
    dim_estimates.erase(old_name);
}

string fuse_dim(Schedule &sched, int dim1, int dim2,
                map<string, int> &dim_estimates) {
    // Add the fuse to the splits list
    string inner_name, outer_name, fused_name;
    vector<Dim> &dims = sched.dims();

    outer_name = dims[dim1].var;
    bool outer_pure = dims[dim1].pure;
    dims.erase(dims.begin() + dim1);

    inner_name = dims[dim2].var;
    fused_name = inner_name + "." + outer_name;
    dims[dim2].var = fused_name;
    dims[dim2].pure &= outer_pure;

    int out_estimate = dim_estimates[outer_name];
    int in_estimate = dim_estimates[inner_name];

    if (in_estimate > 0 && out_estimate > 0)
        dim_estimates[fused_name] = out_estimate * in_estimate;
    else
        dim_estimates[fused_name] = -1;

    dim_estimates.erase(outer_name);
    dim_estimates.erase(inner_name);

    Split split = {fused_name, outer_name, inner_name, Expr(),
                   true, false, Split::FuseVars};
    sched.splits().push_back(split);
    return fused_name;
}

// Vectorization
void vectorize_dim(Schedule &sched, map<string, int> &dim_estimates,
                   int dim, int vec_width) {
    vector<Dim> &dims = sched.dims();
    if (vec_width != -1) {
        split_dim(sched, dim, vec_width, dim_estimates, "vec", false);
        dims[dim].for_type = ForType::Vectorized;
    } else {
        dims[dim].for_type = ForType::Vectorized;
    }
}

bool check_dim_size(Schedule &sched, int dim, int min_size,
                    map<string, int> &dim_estimates) {
    vector<Dim> &dims = sched.dims();
    int extent = dim_estimates[dims[dim].var];
    bool can_vec = false;
    if (extent >= 0)
        can_vec = extent >= min_size;
    return can_vec;
}

void simple_vectorize(Function &func, map<string, int> &dim_estimates,
                      int inner_dim, int vec_width=-1) {
    // Collect all the load args
    FindCallArgs find;
    func.accept(&find);
    // For all the loads find the stride of the innermost loop
    bool vec = true;
    for(auto& larg: find.load_args) {
        Expr diff = simplify(finite_difference(larg[inner_dim],
                             func.args()[inner_dim]));

        //std::cout << "Diff expr" << std::endl;
        //std::cout << diff << std::endl;
        VectorExprCheck vec_check(func.args()[inner_dim]);
        diff.accept(&vec_check);

        vec = vec && ( is_simple_const(diff) ||
                       vec_check.can_vec );
        //std::cout << vec_check.can_vec << std::endl;
    }
    if (vec)
        vectorize_dim(func.schedule(), dim_estimates, inner_dim, vec_width);
}

void vectorize_update(Function &func, int stage,
                      map<string, int> &dim_estimates, int vec_len,
                      set<string> &par_vars) {
    Schedule &s = func.update_schedule(stage);
    const UpdateDefinition &u = func.updates()[stage];
    vector<Dim> &dims = s.dims();
    // Vectorize the inner most loop that can be vectorized
    for (unsigned int dim = 0; dim < dims.size(); dim++) {
        bool dim_par = can_parallelize_rvar(dims[dim].var, func.name(), u);
        dim_par = dim_par || (par_vars.find(dims[dim].var) != par_vars.end());
        if(check_dim_size(s, dim, vec_len, dim_estimates) && dim_par) {
            //move_dim_to_innermost(s, dim);
            //vectorize_dim(s, dim_estimates, 0, vec_len);
            vectorize_dim(s, dim_estimates, dim, vec_len);
            par_vars.insert(dims[dim+1].var);
            break;
        }
    }
}

bool pick_dim_to_parallelize(Function &f, map<string, int> &dim_estimates,
                             int parallelism, Partitioner::GroupSched &sched,
                             int &outer_dim, int& num_fused_dims) {
    // TODO Check which is better fusing the dimensions or moving
    // the right dimension out and parallelizing it
    //std::cout << "Parallel Dim Choice " << f.name() << std::endl;
    vector<Dim> &dims = f.schedule().dims();
    //for (auto &d: dims)
    //    std::cout << d.var << ",";
    //std::cout << std::endl;
    outer_dim = dims.size() - 2;
    int num_tile_dims = 0;
    for (auto &d: sched.tile_sizes) {
       if (d >= 1)
           num_tile_dims++;
    }

    if (num_tile_dims > 0) {
        for (int i = 0; i < num_tile_dims; i++) {
            if (dim_estimates[dims[outer_dim].var] > parallelism)
                return true;
            else {
                fuse_dim(f.schedule(), outer_dim, outer_dim - 1, dim_estimates);
                outer_dim = dims.size() - 2;
                num_fused_dims++;
            }
        }
    } else {
        for (int i = outer_dim; i > 0; i--) {
            //std::cout << dims[i].var << " Num Iter "
            //          << dim_estimates[dims[i].var] << std::endl;
            if (dim_estimates[dims[i].var] > parallelism) {
                move_dim_to_outermost(f.schedule(), i);
                return true;
            }
        }
    }
    return false;
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

    // TODO explain structure
    std::map<string, pair<long long, long long> > func_cost;
    for (auto& kv : env) {
        //std::cout << kv.first << ":" << std::endl;
        assert(func_cost.find(kv.first) == func_cost.end());

        func_cost[kv.first].first = 1;
        func_cost[kv.first].second = 0;

        if (kv.second.is_boundary())
            continue;

        for (auto &e: kv.second.values()) {
            ExprCostEarly cost_visitor;
            e.accept(&cost_visitor);
            func_cost[kv.first].first += cost_visitor.ops;
            func_cost[kv.first].second += cost_visitor.loads;
        }

        // Estimating cost when reductions are involved
        // Only considering functions with a single update covers most of the
        // cases we want to tackle
        assert(kv.second.updates().size() <= 1);
        for (auto &u: kv.second.updates()) {
            int ops = 1;
            int loads = 0;
            for (auto &e: u.values) {
                ExprCostEarly cost_visitor;
                e.accept(&cost_visitor);
                ops += cost_visitor.ops;
                loads += cost_visitor.loads;
            }
            for (auto &arg: u.args) {
                ExprCostEarly cost_visitor;
                arg.accept(&cost_visitor);
                ops += cost_visitor.ops;
                loads += cost_visitor.loads;
            }

            if (u.domain.defined()) {
                Box b;
                for (auto &rvar: u.domain.domain()) {
                    b.push_back(Interval(simplify(rvar.min),
                                         simplify(rvar.min + rvar.extent - 1)));
                    //std::cout << rvar.min << std::endl;
                    //std::cout << rvar.min + rvar.extent - 1 << std::endl;
                }
                long long area = box_area(b);
                // Fixed size RDom
                assert(area!=-1);
                func_cost[kv.first].first += ops * area;
                func_cost[kv.first].second += loads * area;
            }
        }
    }

    // Make obvious inline decisions early
    map<string, vector<string> > inlines;

    // TODO explain structure
    map<string, vector<const Call*> > all_calls;
    map<string, vector<string> > consumers;
    for (auto& kv:env) {
    	FindCallArgs call_args;
    	kv.second.accept(&call_args);
    	for (auto& fcalls: call_args.calls){
            consumers[fcalls.first].push_back(kv.first);
    		all_calls[fcalls.first].insert(all_calls[fcalls.first].end(),
    								  	   fcalls.second.begin(),
                                           fcalls.second.end());
    	}
    }

    /*
    if (auto_inline)
        inlines = simple_inline(all_calls, consumers, env);
    */

    std::cout << "Inlining:" << std::endl;
    for (auto &f: env) {
        if (env[f.first].is_lambda()) {
            //assert(consumers[f.first].size() == 1);
            inlines[f.first].push_back(consumers[f.first][0]);
            env[f.first].schedule().store_level().var = "";
            env[f.first].schedule().compute_level().var = "";
        }
    }

    disp_inlines(inlines);
    std::cout << std::endl;

    bool group = true;
    auto_vec = true;
    auto_par = true;

    if (group) {
        // Dependence analysis

        // For each function compute all the regions of upstream functions
        // required to compute a region of the function

        DependenceAnalysis analy(env, func_val_bounds);

        for (auto &reg: analy.func_dep_regions) {
            disp_regions(reg.second);
            std::cout << std::endl;
        }

        bool bounds_avail = check_bounds_on_outputs(outputs);
        std::cout << "output bounds:" << bounds_avail << std::endl;

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

        disp_regions(pipeline_bounds);

        // Grouping
        Partitioner part(pipeline_bounds, inlines, analy, func_cost);
        std::cout << std::endl << "Function costs pre Inlining" << std::endl;
        part.disp_costs();
        std::cout << std::endl;

        part.initialize_groups_inline();
        std::cout << std::endl << "Groups Pre Inlining" << std::endl;
        part.disp_grouping();
        std::cout << std::endl;
        part.group(Partitioner::INLINE);
        std::cout << std::endl << "Groups Inlining" << std::endl;
        part.disp_grouping();
        std::cout << std::endl;
        // Clear the option cache

        std::cout << std::endl << "Function costs post Inlining" << std::endl;
        part.disp_costs();
        std::cout << std::endl;

        part.initialize_groups_fast_mem();
        part.group(Partitioner::FAST_MEM);
        std::cout << std::endl << "Groups Fast Mem" << std::endl;
        part.disp_grouping();
        std::cout << std::endl;
        //part.disp_grouping();

        //part.reorder_for_input_locality();

        int vec_len = part.arch_params.vec_len;

        // Schedule generation based on grouping
        for (auto& g: part.groups) {
            // Create a tiled traversal for the output of the group
            Function &g_out = env[g.first];

            assert(inlines.find(g_out.name()) == inlines.end());
            // The dimension names that will be tiled
            vector<string> vars;
            vector<Dim> &dims = g_out.schedule().dims();

            Partitioner::GroupSched sched = part.group_sched[g.first];

            map<string, int> tile_sizes;
            for(int i = 0; i < (int)dims.size() - 1; i++) {
                if (sched.tile_sizes[i] != -1) {
                    vars.push_back(dims[i].var);
                    tile_sizes[dims[i].var] = sched.tile_sizes[i];
                }
            }

            // Get estimates of pipeline bounds
            map<string, int> org_out_estimates =
                          get_dim_estimates(g_out.name(), pipeline_bounds, env);
            map<string, int> out_estimates = org_out_estimates;

            // Realizing the tiling and updating the dimension estimates
            int num_tile_dims = 0;
            for(auto &v: vars) {
                int index = -1;
                for (int i = 0; i < (int)dims.size() - 1; i++)
                    if (dims[i].var == v) {
                        index = i;
                        break;
                    }
                assert(index!=-1);
                if (tile_sizes[v] > 1) {
                    split_dim(g_out.schedule(), index, tile_sizes[v],
                              out_estimates, "tile", false);
                    move_dim_to_outermost(g_out.schedule(), index + 1);
                } else if (tile_sizes[v] == 1) {
                    move_dim_to_outermost(g_out.schedule(), index);
                }
                num_tile_dims++;
            }

            int num_fused_dims = 0;
            int parallelism = part.arch_params.parallelism;

            {
                // Vectorize first
                Schedule &s = g_out.schedule();
                if (auto_vec) {
                    if (check_dim_size(s, 0, vec_len, out_estimates))
                        simple_vectorize(g_out, out_estimates, 0, vec_len);
                }
                int outer_dim = -1;
                bool can_par = pick_dim_to_parallelize(g_out, out_estimates,
                                                       parallelism, sched,
                                                       outer_dim, num_fused_dims);

                if (auto_par && outer_dim !=-1 && can_par)
                    parallelize_dim(g_out.schedule(), outer_dim);
            }

            if (!g_out.is_pure()) {

                int num_updates = g_out.updates().size();
                for (int i = 0; i < num_updates; i ++) {
                    // Start with fresh bounds estimates for each update
                    map<string, int> out_up_estimates = org_out_estimates;

                    Schedule &s = g_out.update_schedule(i);
                    vector<Dim> &dims = s.dims();

                    // Use the same tiling as the pure dimensions
                    const UpdateDefinition &u = g_out.updates()[i];
                    set<string> par_vars;
                    for(auto &v: vars) {
                        int index = -1;
                        for (int i = 0; i < (int)dims.size() - 1; i++)
                            if (dims[i].var == v) {
                                index = i;
                                break;
                            }
                        assert(index!=-1);
                        if (tile_sizes[v] > 1) {
                            split_dim(s, index, tile_sizes[v],
                                      out_up_estimates, "tile", false);
                            move_dim_to_outermost(s, index + 1);
                            if (can_parallelize_rvar(v, g_out.name(), u)) {
                                int o_dim = s.dims().size() - 2;
                                par_vars.insert(s.dims()[o_dim].var);
                                par_vars.insert(s.dims()[index].var);
                            }
                        } else if (tile_sizes[v] == 1) {
                            move_dim_to_outermost(s, index);
                        }
                    }

                    // Vectorization of update definitions
                    vectorize_update(g_out, i, out_up_estimates, vec_len,
                                     par_vars);

                    int curr_par = 1;
                    for (int i = (int)dims.size() - 2; i > 0 ; i--) {
                        bool dim_par = can_parallelize_rvar(dims[i].var,
                                                            g_out.name(), u);
                        dim_par = dim_par ||
                                  (par_vars.find(dims[i].var) != par_vars.end());
                        if (dim_par) {
                            curr_par = curr_par * out_up_estimates[dims[i].var];
                            parallelize_dim(s, i);
                            if (curr_par > parallelism)
                                break;
                        } else {
                            break;
                        }

                        /*
                        if (dim_par && out_up_estimates[dims[i].var] > parallelism) {
                            move_dim_to_outermost(s, i);
                            int outer_dim = dims.size() - 2;
                            parallelize_dim(s, outer_dim);
                            break;
                        }
                        */
                    }
                }
            }

            for (auto &m: g.second) {
                int outer_dim = dims.size() - 2;
                map<string, int> org_mem_estimates =
                          get_dim_estimates(m.name(), pipeline_bounds, env);
                map<string, int> mem_estimates = org_mem_estimates;
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
                    if (auto_vec)
                        if (check_dim_size(m.schedule(), 0, vec_len, mem_estimates))
                            simple_vectorize(m, mem_estimates, 0, vec_len);
                    if (!m.is_pure()) {
                        int num_updates = m.updates().size();
                        for (int i = 0; i < num_updates; i ++) {
                            // Start with fresh bounds estimates for each update
                            map<string, int> mem_up_estimates =
                                                org_mem_estimates;
                            set<string> par_vars;
                            vectorize_update(m, i, mem_up_estimates, vec_len,
                                             par_vars);
                        }
                    }
                }
            }
        }
    }
    // TODO Method for reordering and unrolling based on reuse across iterations

    if (root_default || auto_vec || auto_par || auto_inline)
        disp_schedule_and_storage_mapping(env);

	return;
}

}
}
