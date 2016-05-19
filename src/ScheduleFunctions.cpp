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
#include "CodeGen_GPU_Dev.h"
#include "IRPrinter.h"

#include "FindCalls.h"
#include "ParallelRVar.h"
#include "RealizationOrder.h"

#include <cstdlib>
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
// A structure representing a containing LetStmt, IfThenElse, or For
// loop. Used in build_provide_loop_nest below.
struct Container {
    enum Type {For, Let, If};
    Type type;
    // If it's a for loop, the index in the dims list.
    int dim_idx;
    string name;
    Expr value;
};
}

class ContainsImpureCall : public IRVisitor {
    using IRVisitor::visit;

    void visit(const Call *op) {
        if (!op->is_pure()) {
            result = true;
        } else {
            IRVisitor::visit(op);
        }
    }

public:
    bool result = false;
    ContainsImpureCall() {}
};

bool contains_impure_call(const Expr &expr) {
    ContainsImpureCall is_not_pure;
    expr.accept(&is_not_pure);
    return is_not_pure.result;
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
    for (const Split &split : splits) {
        Expr outer = Variable::make(Int(32), prefix + split.outer);
        Expr outer_max = Variable::make(Int(32), prefix + split.outer + ".loop_max");
        if (split.is_split()) {
            Expr inner = Variable::make(Int(32), prefix + split.inner);
            Expr old_max = Variable::make(Int(32), prefix + split.old_var + ".loop_max");
            Expr old_min = Variable::make(Int(32), prefix + split.old_var + ".loop_min");
            Expr old_extent = Variable::make(Int(32), prefix + split.old_var + ".loop_extent");

            known_size_dims[split.inner] = split.factor;

            Expr base = outer * split.factor + old_min;
            string base_name = prefix + split.inner + ".base";
            Expr base_var = Variable::make(Int(32), base_name);
            string old_var_name = prefix + split.old_var;
            Expr old_var = Variable::make(Int(32), old_var_name);

            map<string, Expr>::iterator iter = known_size_dims.find(split.old_var);

            if (is_update) {
                user_assert(split.tail != TailStrategy::ShiftInwards)
                    << "When splitting Var " << split.old_var
                    << " ShiftInwards is not a legal tail strategy for update definitions, as"
                    << " it may change the meaning of the algorithm\n";
            }

            if (split.exact) {
                user_assert(split.tail == TailStrategy::Auto ||
                            split.tail == TailStrategy::GuardWithIf)
                    << "When splitting Var " << split.old_var
                    << " the tail strategy must be GuardWithIf or Auto. "
                    << "Anything else may change the meaning of the algorithm\n";
            }

            TailStrategy tail = split.tail;
            if (tail == TailStrategy::Auto) {
                if (split.exact) {
                    tail = TailStrategy::GuardWithIf;
                } else if (is_update) {
                    tail = TailStrategy::RoundUp;
                } else {
                    tail = TailStrategy::ShiftInwards;
                }
            }

            if ((iter != known_size_dims.end()) &&
                is_zero(simplify(iter->second % split.factor))) {
                // We have proved that the split factor divides the
                // old extent. No need to adjust the base or add an if
                // statement.
                known_size_dims[split.outer] = iter->second / split.factor;
            } else if (is_one(split.factor)) {
                // The split factor trivially divides the old extent,
                // but we know nothing new about the outer dimension.
            } else if (tail == TailStrategy::GuardWithIf) {
                // It's an exact split but we failed to prove that the
                // extent divides the factor. Use predication.

                // Make a var representing the original var minus its
                // min. It's important that this is a single Var so
                // that bounds inference has a chance of understanding
                // what it means for it to be limited by the if
                // statement's condition.
                Expr rebased = outer * split.factor + inner;
                string rebased_var_name = prefix + split.old_var + ".rebased";
                Expr rebased_var = Variable::make(Int(32), rebased_var_name);
                stmt = substitute(prefix + split.old_var, rebased_var + old_min, stmt);

                // Tell Halide to optimize for the case in which this
                // condition is true by partitioning some outer loop.
                Expr cond = likely(rebased_var < old_extent);
                stmt = IfThenElse::make(cond, stmt, Stmt());
                stmt = LetStmt::make(rebased_var_name, rebased, stmt);

            } else if (tail == TailStrategy::ShiftInwards) {
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
            } else {
                internal_assert(tail == TailStrategy::RoundUp);
            }

            // Substitute in the new expression for the split variable ...
            stmt = substitute(old_var_name, base_var + inner, stmt);
            // ... but also define it as a let for the benefit of bounds inference.
            stmt = LetStmt::make(old_var_name, base_var + inner, stmt);
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
        Container c = {Container::For, i, prefix + dim.var, Expr()};
        nest.push_back(c);
    }

    // Strip off the lets into the containers vector.
    while (const LetStmt *let = stmt.as<LetStmt>()) {
        Container c = {Container::Let, 0, let->name, let->value};
        nest.push_back(c);
        stmt = let->body;
    }

    // Put all the reduction domain predicate into the containers vector.
    int n_predicates = 0;
    if (rdom.defined()) {
        vector<Expr> predicates = rdom.split_predicate();
        n_predicates = predicates.size();
        for (Expr pred : predicates) {
            pred = qualify(prefix, pred);
            Container c = {Container::If, 0, "", pred};
            nest.push_back(c);
        }
    }

    // Resort the containers vector so that lets are as far outwards
    // as possible. Use reverse insertion sort. Start at the first letstmt.
    for (int i = (int)s.dims().size(); i < (int)nest.size() - n_predicates; i++) {
        // Only push up LetStmts.
        internal_assert(nest[i].value.defined());
        internal_assert(nest[i].type == Container::Let);

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

    // Sort the predicate guards so they are as far outwards as possible.
    for (int i = (int)nest.size() - n_predicates; i < (int)nest.size(); i++) {
        // Only push up LetStmts.
        internal_assert(nest[i].value.defined());
        internal_assert(nest[i].type == Container::If);

        // Cannot lift out the predicate guard if it contains call to non-pure function
        if (contains_impure_call(nest[i].value)) {
            continue;
        }

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
        if (nest[i].type == Container::Let) {
            internal_assert(nest[i].value.defined());
            stmt = LetStmt::make(nest[i].name, nest[i].value, stmt);
        } else if (nest[i].type == Container::If) {
            internal_assert(nest[i].value.defined());
            stmt = IfThenElse::make(likely(nest[i].value), stmt, Stmt());
        } else {
            internal_assert(nest[i].type == Container::For);
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
            Expr inner_extent = split.factor;
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
        stmt = LetStmt::make(o + ".loop_max", 0, stmt);
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
                    Expr buffer = Variable::make(type_of<struct buffer_t *>(), buf_name);
                    extern_call_args.push_back(buffer);
                }
            } else if (arg.is_buffer()) {
                Buffer b = arg.buffer;
                Parameter p(b.type(), true, b.dimensions(), b.name());
                p.set_buffer(b);
                Expr buf = Variable::make(type_of<struct buffer_t *>(), b.name() + ".buffer", p);
                extern_call_args.push_back(buf);
            } else if (arg.is_image_param()) {
                Parameter p = arg.image_param;
                Expr buf = Variable::make(type_of<struct buffer_t *>(), p.name() + ".buffer", p);
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
                Expr buffer = Variable::make(type_of<struct buffer_t *>(), buf_name);
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

                Expr output_buffer_t = Call::make(type_of<struct buffer_t *>(), Call::create_buffer_t,
                                                  buffer_args, Call::Intrinsic);

                string buf_name = f.name() + "." + std::to_string(j) + ".tmp_buffer";
                extern_call_args.push_back(Variable::make(type_of<struct buffer_t *>(), buf_name));
                lets.push_back(make_pair(buf_name, output_buffer_t));
            }
        }

        // Make the extern call
        Expr e = Call::make(Int(32), extern_name, extern_call_args,
                            f.extern_definition_is_c_plus_plus() ? Call::ExternCPlusPlus
                                                                 : Call::Extern);
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
            debug(3) << "Update value " << i << " = " << v << "\n";
        }

        for (size_t i = 0; i < r.args.size(); i++) {
            Expr s = r.args[i];
            s = qualify(prefix, s);
            site[i] = s;
            debug(3) << "Update site " << i << " = " << s << "\n";
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

    // Combine the update steps
    Stmt merged_updates = Block::make(updates);
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
            string prefix = func.name() + ".s" + std::to_string(stage) + "." + b.var;
            string min_name = prefix + ".min_unbounded";
            string max_name = prefix + ".max_unbounded";
            Expr min_var = Variable::make(Int(32), min_name);
            Expr max_var = Variable::make(Int(32), max_name);
            if (!b.min.defined()) {
                b.min = min_var;
            }

            Expr max_val = (b.extent + b.min) - 1;
            Expr min_val = b.min;

            Expr check = (min_val <= min_var) && (max_val >= max_var);
            Expr error_msg = Call::make(Int(32), "halide_error_explicit_bounds_too_small",
                                        {b.var, func.name(), min_val, max_val, min_var, max_var},
                                        Call::Extern);
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
        if (op->type.is_handle() &&
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
    bool is_output, found_store_level, found_compute_level;
    const Target &target;

    InjectRealization(const Function &f, bool o, const Target &t) :
        func(f), is_output(o),
        found_store_level(false), found_compute_level(false),
        target(t) {}

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
        if (target.has_feature(Target::NoAsserts)) {
            return s;
        } else {
            return inject_explicit_bounds(s, func);
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
        if (v->type.is_handle() &&
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

void validate_schedule(Function f, Stmt s, const Target &target, bool is_output) {

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

    // If the func is scheduled on the gpu, check that the relevant
    // api is enabled in the target.
    vector<Schedule> schedules;
    schedules.push_back(f.schedule());
    for (const UpdateDefinition &u : f.updates()) {
        schedules.push_back(u.schedule);
    }
    for (size_t i = 0; i < schedules.size(); i++) {
        for (const Specialization &s : schedules[i].specializations()) {
            schedules.push_back(s.schedule);
        }
    }
    for (const Schedule &s : schedules) {
        for (const Dim &d : s.dims()) {
            if (!target.supports_device_api(d.device_api)) {
                user_error << "Schedule for Func " << f.name()
                           << " requires " << d.device_api
                           << " but no compatible target feature is enabled in target "
                           << target.to_string() << "\n";
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

    /*
    std::cerr << "Legal sites for " << f.name() << ":\n";
    auto site = legal.sites_allowed.begin();
    while (site != legal.sites_allowed.end()) {
        if (site->loop_level.var == "__outermost") {
            site = legal.sites_allowed.erase(site);
            continue;
        }
        std::cerr << site->loop_level << "\n";
        site++;
    }
    std::cerr << "\n";

    if (store_at.is_index) {
        std::cerr << "Store: " << store_at << " -> ";
        assert(store_at.index >= 0);
        if (store_at.index >= (int)legal.sites_allowed.size()) {
            store_at.index = legal.sites_allowed.size() - 1;
        }
        store_at = legal.sites_allowed[store_at.index].loop_level;
        f.schedule().store_level() = store_at;
        std::cerr << store_at << std::endl;
    }
    if (compute_at.is_index) {
        std::cerr << "Compute: " << compute_at << " -> ";
        assert(compute_at.index >= 0);
        if (compute_at.index >= (int)legal.sites_allowed.size()) {
            compute_at.index = legal.sites_allowed.size() - 1;
        }
        compute_at = legal.sites_allowed[compute_at.index].loop_level;
        f.schedule().compute_level() = compute_at;
        std::cerr << compute_at << std::endl;
    }
    */

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
        if (ends_with(op->name, ".__outermost") && is_one(simplify(op->extent))) {
            stmt = mutate(substitute(op->name, op->min, op->body));
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const LetStmt *op) {
        if (ends_with(op->name, ".__outermost.loop_extent") ||
            ends_with(op->name, ".__outermost.loop_min") ||
            ends_with(op->name, ".__outermost.loop_max")) {
            stmt = mutate(substitute(op->name, simplify(op->value), op->body));
        } else {
            IRMutator::visit(op);
        }
    }
};

Stmt schedule_functions(const vector<Function> &outputs,
                        const vector<string> &order,
                        const map<string, Function> &env,
                        const Target &target,
                        bool &any_memoized) {

    string root_var = LoopLevel::root().func + "." + LoopLevel::root().var;
    Stmt s = For::make(root_var, 0, 1, ForType::Serial, DeviceAPI::Host, Evaluate::make(0));

    any_memoized = false;

    for (size_t i = order.size(); i > 0; i--) {
        Function f = env.find(order[i-1])->second;

        bool is_output = false;
        for (Function o : outputs) {
            is_output |= o.same_as(f);
        }

        validate_schedule(f, s, target, is_output);

        if (f.has_pure_definition() &&
            !f.has_update_definition() &&
            f.schedule().compute_level().is_inline()) {
            debug(1) << "Inlining " << order[i-1] << '\n';
            s = inline_function(s, f);
        } else {
            debug(1) << "Injecting realization of " << order[i-1] << '\n';
            InjectRealization injector(f, is_output, target);
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
                calls[call->name].push_back(call);
                load_args.push_back(call->args);
            }
            for (size_t i = 0; (i < call->args.size()); i++)
                call->args[i].accept(this);
        }
};

class FindAllCallArgs : public IRVisitor {
    public:
        map<string, std::vector<const Call*> > calls;
        vector<vector<Expr>> load_args;

        using IRVisitor::visit;

        void visit(const Call *call) {
            // See if images need to be included
            if (call->call_type == Call::Halide ||
                call->call_type == Call::Image) {
                calls[call->name].push_back(call);
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

class UsesVarCheck : public IRVisitor {
    public:
        bool uses_var;
        string var;

        UsesVarCheck(string _var): var(_var) {
            uses_var = false;
        }
        using IRVisitor::visit;

        void visit(const Variable *v) { uses_var = uses_var
                                        || (v->name == var); }
};

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
map<string, Box> regions_required(Function f, const vector<string> &update_args,
                                  const vector< pair<Expr, Expr> > &sym_bounds,
                                  const map<string, Function> &env,
                                  const FuncValueBounds &func_val_bounds,
                                  bool include_self = false){
    // Define the bounds for each variable of the function
    std::vector<Interval> bounds;
    unsigned int num_args = f.args().size();
    unsigned int num_update_args = update_args.size();

    for (unsigned int arg = 0; arg < num_args; arg++)
        bounds.push_back(Interval(sym_bounds[arg].first,
                                  sym_bounds[arg].second));

    for (unsigned int arg = 0; arg < num_update_args; arg++) {
        unsigned int sym_index = num_args + arg;
        bounds.push_back(Interval(sym_bounds[sym_index].first,
                                  sym_bounds[sym_index].second));
    }

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

                if (env.find(reg.first) != env.end())
                    f_queue.push_back(make_pair(env.at(reg.first),
                                                reg.second.bounds));
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
                    if (!v) {
                        // Need to evaluate boxes required on args that are not pure
                        // for potenial calls to other functions
                        exprs.push_back(arg);
                    } else {
                        curr_scope.push(v->name, simple_bounds);
                    }
                    interval_index++;
                }

                if (update.domain.defined()) {
                    // Partial analysis is only done for the output function f
                    if (num_update_args > 0 && curr_f.name() == f.name()) {
                        //std::cerr << curr_f.name() << std::endl;
                        assert(update.domain.domain().size() == num_update_args);
                        //std::cerr << curr_bounds.size() << " " << num_args << " "
                        //          << num_update_args << std::endl;
                        assert(curr_bounds.size() ==
                                    num_update_args + num_args);
                        for (auto &rvar: update.domain.domain()) {
                            Interval simple_bounds =
                                Interval(simplify(curr_bounds[interval_index].min),
                                         simplify(curr_bounds[interval_index].max));
                            curr_scope.push(rvar.var, simple_bounds);
                            interval_index++;
                        }
                    } else {
                        for (auto &rvar: update.domain.domain()) {
                            Interval simple_bounds = Interval(rvar.min,
                                                              rvar.min + rvar.extent - 1);
                            curr_scope.push(rvar.var, simple_bounds);
                        }
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

                            if (env.find(reg.first) != env.end())
                                f_queue.push_back(make_pair(env.at(reg.first),
                                                            reg.second.bounds));
                        } else if (include_self) {
                            if (regions.find(reg.first) == regions.end())
                                regions[reg.first] = reg.second;
                            else
                                merge_boxes(regions[reg.first], reg.second);
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
                                   const vector<string> &update_args,
                                   const vector<pair<Expr, Expr>> &sym_bounds,
                                   const map<string, Function> &env,
                                   const FuncValueBounds &func_val_bounds){

    map<string, Box> regions = regions_required(f, update_args, sym_bounds, env,
                                                func_val_bounds, true);
    vector<pair<Expr, Expr>> shifted_bounds;
    int num_pure_args = f.args().size();
    for (int arg = 0; arg < num_pure_args; arg++) {
        if (dir == arg) {
            Expr len = sym_bounds[arg].second - sym_bounds[arg].first + 1;
            pair<Expr, Expr> bounds = make_pair(sym_bounds[arg].first + len,
                                              sym_bounds[arg].second + len);
            shifted_bounds.push_back(bounds);
        }
        else
            shifted_bounds.push_back(sym_bounds[arg]);
    }

    int num_update_args = update_args.size();
    for (int arg = 0; arg < num_update_args; arg++) {
        int sym_index = num_pure_args + arg;
        if (dir == sym_index) {
            Expr len = sym_bounds[arg].second - sym_bounds[sym_index].first + 1;
            pair<Expr, Expr> bounds = make_pair(sym_bounds[sym_index].first + len,
                                                sym_bounds[sym_index].second + len);
            shifted_bounds.push_back(bounds);
        }
        else
            shifted_bounds.push_back(sym_bounds[sym_index]);
    }

    map<string, Box> regions_shifted = regions_required(f, update_args, shifted_bounds,
                                                        env, func_val_bounds,
                                                        true);

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
            // The box used makes things complicated ignoring it for now
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
                for (auto &b: env[r.first].schedule().estimates()) {
                    unsigned int num_pure_args = env[r.first].args().size();
                    if (i < num_pure_args && b.var == env[r.first].args()[i])
                        lower = Expr(b.min.as<IntImm>()->value);
                }
            }

            if (!upper.as<IntImm>()) {
                for (auto &b: env[r.first].schedule().estimates()) {
                    unsigned int num_pure_args = env[r.first].args().size();
                    if (i < num_pure_args && b.var == env[r.first].args()[i]) {
                        const IntImm * bmin = b.min.as<IntImm>();
                        const IntImm * bextent = b.extent.as<IntImm>();
                        upper = Expr(bmin->value + bextent->value - 1);
                    }
                }
            }

            Interval concrete_bounds = Interval(lower, upper);
            concrete_box.push_back(concrete_bounds);
        }
        concrete_regions[r.first] = concrete_box;
    }
    return concrete_regions;
}

void disp_box(Box &b) {
    for (unsigned int dim = 0; dim < b.size(); dim++)
        std::cerr << "(" << b[dim].min << "," << b[dim].max << ")";
}

void disp_regions(map<string, Box> &regions) {
    for (auto& reg: regions) {
        std::cerr << reg.first;
        disp_box(reg.second);
        std::cerr << std::endl;
    }
}

struct DependenceAnalysis {

    map<string, Function> &env;
    const FuncValueBounds &func_val_bounds;
    map<string, map<string, Box> > func_dep_regions;
    map<string, map<string, Box> > func_partial_dep_regions;
    map<string, vector< map<string, Box> > > func_overlaps;
    map<string, vector< map<string, Box> > > func_partial_overlaps;
    map<string, vector< pair<Var, Var> > > func_sym;
    set<string> &reductions;
    map<string, vector<string> > &update_args;

    DependenceAnalysis(map<string, Function> &_env,
                       const FuncValueBounds &_func_val_bounds,
                       set<string> &_reductions,
                       map<string, vector<string> > &_update_args):
                       env(_env), func_val_bounds(_func_val_bounds),
                       reductions(_reductions), update_args(_update_args) {
        for (auto& kv : env) {
            // For each argument create a variables which will serve as the lower
            // and upper bounds of the interval corresponding to the argument
            const vector<string> &args = kv.second.args();
            assert(args.size() > 0);
            vector< pair<Expr, Expr> > sym_bounds;
            for (unsigned int arg = 0; arg < args.size(); arg++) {
                Var lower = Var(args[arg] + "_l");
                Var upper = Var(args[arg] + "_u");
                pair<Var, Var> sym = make_pair(lower, upper);
                pair<Expr, Expr> bounds = make_pair(Expr(lower), Expr(upper));
                func_sym[kv.first].push_back(sym);
                sym_bounds.push_back(bounds);
            }

            vector<string> u_args;
            map<string, Box> regions = regions_required(kv.second, u_args, sym_bounds,
                                                        env, func_val_bounds);
            assert(func_dep_regions.find(kv.first) == func_dep_regions.end());
            func_dep_regions[kv.first] = regions;

            /*
            std::cerr << "Function regions required for " << kv.first << ":" << std::endl;
            disp_regions(regions);
            std::cerr << std::endl;
            */

            assert(func_overlaps.find(kv.first) == func_overlaps.end());
            for (unsigned int arg = 0; arg < args.size(); arg++) {
                map<string, Box> overlaps = redundant_regions(kv.second, arg,
                                                              u_args, sym_bounds, env,
                                                              func_val_bounds);
                func_overlaps[kv.first].push_back(overlaps);

                /*
                std::cerr << "Function region overlaps for var " <<
                          args[arg]  << " " << kv.first << ":" << std::endl;
                disp_regions(overlaps);
                std::cerr << std::endl;
                */
            }

            if (reductions.find(kv.first) != reductions.end()) {
                //std::cerr << "Processing reduction " << kv.first << std::endl;
                assert(update_args.find(kv.first) != update_args.end());
                u_args = update_args[kv.first];

                // Append the required symbolic bounds
                for (unsigned int arg = 0; arg < u_args.size(); arg++) {
                    Var lower = Var(u_args[arg] + "_l");
                    Var upper = Var(u_args[arg] + "_u");
                    pair<Var, Var> sym = make_pair(lower, upper);
                    pair<Expr, Expr> bounds = make_pair(Expr(lower), Expr(upper));
                    func_sym[kv.first].push_back(sym);
                    sym_bounds.push_back(bounds);
                }

                map<string, Box> regions = regions_required(kv.second, u_args, sym_bounds,
                                                            env, func_val_bounds);
                assert(func_partial_dep_regions.find(kv.first) ==
                       func_partial_dep_regions.end());
                func_partial_dep_regions[kv.first] = regions;

                /*
                   std::cerr << "Function regions required for " << kv.first << ":" << std::endl;
                   disp_regions(regions);
                   std::cerr << std::endl;
                 */

                assert(func_partial_overlaps.find(kv.first) ==
                       func_partial_overlaps.end());

                unsigned int num_args = u_args.size() + args.size();
                for (unsigned int arg = 0; arg < num_args; arg++) {
                    map<string, Box> overlaps = redundant_regions(kv.second, arg,
                                                                  u_args, sym_bounds, env,
                                                                  func_val_bounds);
                    func_partial_overlaps[kv.first].push_back(overlaps);

                    /*
                       std::cout << "Function region overlaps for var " <<
                       args[arg]  << " " << kv.first << ":" << std::endl;
                       disp_regions(overlaps);
                       std::cout << std::endl;
                     */
                }
            }
        }
    }

    map<string, Box>
        concrete_dep_regions(string name, vector<bool> &eval,
                             map<string, map<string, Box> > &dep_regions,
                             vector<pair<int, int> > &bounds) {
        return sym_to_concrete_bounds(func_sym[name], bounds, eval,
                                      dep_regions[name], env);
    }

    vector< map<string, Box> >
        concrete_overlap_regions(string name, vector<bool> &eval,
                                 map<string, vector< map<string, Box> > > &overlaps,
                                 vector<pair<int, int> > &bounds) {
        assert(eval.size() == bounds.size());
        assert(overlaps.find(name) != overlaps.end());
        vector< map<string, Box> > conc_overlaps;
        for (auto &dir_overlap: overlaps[name]) {
            map<string, Box> conc_reg =
                sym_to_concrete_bounds(func_sym[name], bounds, eval,
                                       dir_overlap, env);
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
                      map<string, map<string, Box> > &func_dep_regions,
                      bool gpu_schedule) {
    if (gpu_schedule) {
        // Computing total size
        int total_size = 0;
        for(auto &f: regions) {
            int size = region_size(f.first, f.second, env);
            if (size < 0)
                return -1;
            else
                total_size += size;
        }
        return total_size;
    }

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
    //std::cerr << func << " " << total_cost << std::endl;
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
        std::cerr << f.first <<  ": [";
        for (auto &c: f.second)
            std::cerr << c << ",";
        std::cerr << "]" << std::endl;
    }
}

int get_extent_estimate(Function &f, map<string, Box> &bounds, int dim) {

    vector<string> vars = f.args();
    int estimate = -1;
    for (auto &b: f.schedule().estimates())
        if (b.var == vars[dim]) {
            const IntImm * bextent = b.extent.as<IntImm>();
            estimate = bextent->value;
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
    for (auto &b: f.schedule().estimates())
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
    for (auto &b: f.schedule().estimates())
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
        std::cerr << "Calls in function " << f.first << std::endl;
        for (auto &c: f.second)
            std::cerr << c.first << " " << c.second << std::endl;
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
        // Reuse
        vector<float> reuse;

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
        // Has the group been tiled for fusion
        bool fusion;
        // Has the group been tiled for locality
        bool locality;
        // Reuse estimates
        vector<float> reuse;

        GroupSched() {
            benefit = 0;
            redundant_work = 0;
            saved_mem = 0;
            fusion = false;
            locality = false;
        }
    };

    struct MachineParams {
        int parallelism;
        int vec_len;
        long long fast_mem_size;
        int balance;
        int max_threads_per_block;
        int target_threads_per_block;
    };

    map<string, Box> &pipeline_bounds;
    map<string, vector<string> > &inlines;
    DependenceAnalysis &analy;
    map<string, pair<long long, long long> > &func_cost;
    const vector<Function> &outputs;

    map<string, float > input_reuse;
    map<string, vector<Function> > groups;
    map<string, GroupSched> group_sched;
    map<string, set<string> > children;

    map<string, vector<int> > func_pure_dim_estimates;
    map<string, map<string, int> > func_dim_estimates;
    map<string, long long > func_op;
    map<string, long long > func_size;
    map<string, map<string, int> > func_calls;

    map<pair<string, string>, Option> option_cache;

    bool gpu_schedule;
    int random_seed;
    bool debug_info;

    MachineParams arch_params;

    Partitioner(map<string, Box> &_pipeline_bounds,
                map<string, vector<string> > &_inlines, DependenceAnalysis &_analy,
                map<string, pair<long long, long long> > &_func_cost,
                const vector<Function> &_outputs, bool _gpu_schedule,
                int _random_seed, bool _debug_info):
                pipeline_bounds(_pipeline_bounds), inlines(_inlines),
                analy(_analy), func_cost(_func_cost), outputs(_outputs),
                gpu_schedule(_gpu_schedule),
                random_seed(_random_seed),
                debug_info(_debug_info) {

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
            for (int i = (int)args.size() - 1; i >= 0; i --) {
                sched.tile_sizes.push_back(-1);
                sched.reuse.push_back(-1);
            }

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

        //disp_func_calls(func_calls);

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
            func_pure_dim_estimates[f.first] = dim_estimates;
            func_dim_estimates[f.first] =
                get_dim_estimates(f.first, pipeline_bounds, analy.env);
        }

        arch_params.parallelism = 12;
        arch_params.vec_len = 16;

        //arch_params.parallelism = 16 * 8;
        //arch_params.vec_len = 16;

        if (!random_seed) {
            // Initialize machine params
            arch_params.balance = 10;
            //arch_params.fast_mem_size = 8 * 1024;
            arch_params.fast_mem_size = 8 * 1024 * 32;
            // L1 = 32K
            // L2 = 256K
            // L3 = 8192K
            arch_params.max_threads_per_block = 1024;
            arch_params.target_threads_per_block = 128;

            char *var;
            var = getenv("HL_AUTO_PARALLELISM");
            if (var) {
                arch_params.parallelism = atoi(var);
            }
            var = getenv("HL_AUTO_VEC_LEN");
            if (var) {
                arch_params.vec_len = atoi(var);
            }
            var = getenv("HL_AUTO_BALANCE");
            if (var) {
                arch_params.balance = atoi(var);
            }
            var = getenv("HL_AUTO_FAST_MEM_SIZE");
            if (var) {
                arch_params.fast_mem_size = atoi(var);
            }
        } else {
            vector<int> balance = { 2,4,6,8,10,12,14,16,18,20 };
            vector<int> fast_mem_kb = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
            int balance_idx = rand() % balance.size();
            int fast_mem_kb_idx = rand() % fast_mem_kb.size();

            arch_params.balance = balance[balance_idx];
            arch_params.fast_mem_size = fast_mem_kb[fast_mem_kb_idx]*1024*8;
        }

        fprintf(stdout,
                "auto_sched_par: %d\n"
                "auto_sched_vec: %d\n"
                "auto_sched_balance: %d\n"
                "auto_sched_fast_mem_size: %lld\n",
                arch_params.parallelism, arch_params.vec_len,
                arch_params.balance, arch_params.fast_mem_size);
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
            std::cerr << "Group " <<  g.first  << " : [" ;
            for (auto& m: g.second)
                std::cerr << m.name() << ",";
            std::cerr << "]" << std::endl;
        }
    }

    void disp_costs() {
        for (auto &f: analy.env) {
            std::cerr << f.first << " Cost " <<
                func_cost[f.first].first  << " " <<
                func_cost[f.first].second << std::endl;
        }
    }

    void disp_option(Option &opt) {
        std::cerr << opt.prod_group << "->" << opt.cons_group << std::endl;
        std::cerr << "[";
        for (unsigned int i = 0; i < opt.tile_sizes.size(); i++) {
            std::cerr << opt.tile_sizes[i] << ",";
        }
        std::cerr << "]" << std::endl;
        std::cerr << "Benefit:" << opt.benefit << std::endl;
        std::cerr << "Redundant work:" << opt.redundant_work << std::endl;
        std::cerr << "Memory accesses saved:" << opt.saved_mem << std::endl;
    }

    Option choose_candidate(const vector< pair<string, string > > &cand_pairs);
    pair<float, vector<Option> >
        choose_candidate_inline(const vector< pair<string, string > > &cand_pairs);
    void group(Partitioner::Level level);
    void clear_schedules_fast_mem();
    void initialize_groups_fast_mem();
    void initialize_groups_inline();
    void update_function_costs();
    void evaluate_option(Option &opt, Partitioner::Level level);
    void tile_for_input_locality(bool init_pipeline_reuse = false);
    vector<float> get_input_reuse(Function f, vector<string> &inputs);
    pair<float, float> evaluate_reuse(string, vector<string> &group_inputs,
                                      vector<int> &tile_sizes, bool unit_tile);
};

void Partitioner::clear_schedules_fast_mem() {
    for (auto &s: group_sched) {
        // Do not clear the benefit from inlining phase
        s.second.benefit = s.second.saved_mem * arch_params.balance;
        s.second.redundant_work = 0;
        s.second.saved_mem = 0;
        s.second.fusion = false;
        s.second.locality = false;

        for (unsigned int i = 0; i < s.second.tile_sizes.size(); i++) {
            s.second.tile_sizes[i] = -1;
            s.second.reuse[i] = -1;
        }
    }
}

void Partitioner::initialize_groups_inline() {
    for (auto &g: groups) {
        Option opt;
        opt.prod_group = "";
        opt.cons_group = g.first;

        Function output = analy.env[g.first];
        const vector<string> &args = output.args();

        for (unsigned int i = 0; i < args.size(); i++) {
            opt.tile_sizes.push_back(1);
            opt.reuse.push_back(-1);
        }

        evaluate_option(opt, Partitioner::INLINE);

        GroupSched sched;
        sched.saved_mem = opt.saved_mem;
        sched.redundant_work = opt.redundant_work;
        sched.benefit = opt.benefit;
        sched.tile_sizes = opt.tile_sizes;
        sched.reuse = opt.reuse;
        sched.fusion = false;
        sched.locality = false;

        group_sched[g.first] = sched;
    }
}

void Partitioner::initialize_groups_fast_mem() {
    option_cache.clear();
    clear_schedules_fast_mem();
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

            bool is_output = false;
            for (auto &out: outputs) {
                if(out.name() == g.first) {
                    is_output = true;
                }
            }

            if (is_output)
                continue;

            if (children.find(g.first) != children.end()) {
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

        if (debug_info) {
            std::cerr << "Current grouping candidates:" << std::endl;
            for (auto &p: cand) {
                std::cerr << "[" << p.first << "," <<  p.second << "]";
            }
            std::cerr << std::endl;
        }

        vector<pair<string, string> > invalid_keys;
        if (level == Partitioner::INLINE) {
            pair<float, vector<Option> > best;
            best = choose_candidate_inline(cand);
            if (best.first >= 0) {
                string prod = best.second[0].prod_group;

                if (debug_info) {
                    std::cerr << "Choice Inline:" << prod << std::endl;
                }

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
                    sched.reuse = best.second[i].reuse;
                    sched.benefit = best.second[i].benefit;
                    sched.redundant_work = best.second[i].redundant_work;
                    assert(best.second[i].saved_mem >= 0);
                    sched.saved_mem = best.second[i].saved_mem;
                    sched.fusion = false;
                    sched.locality = false;
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

                if (debug_info) {
                    std::cerr << "Choice Fuse:";
                    std::cerr << best.prod_group << "->"
                        << best.cons_group << std::endl;
                    std::cerr << "[";
                    for (auto s: best.tile_sizes)
                        std::cerr << s << ",";
                    std::cerr << "]"  << std::endl;
                }

                for (auto& opt: option_cache) {
                    if (opt.first.second == best.cons_group
                            || opt.first.first == best.cons_group)
                        invalid_keys.push_back(opt.first);
                }

                GroupSched sched;
                sched.tile_sizes = best.tile_sizes;
                sched.reuse = best.reuse;
                sched.benefit = best.benefit;
                sched.redundant_work = best.redundant_work;
                assert(best.saved_mem >= 0);
                sched.saved_mem = best.saved_mem;
                sched.fusion = true;
                sched.locality = false;
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

void Partitioner::evaluate_option(Option &opt, Partitioner::Level l) {

    //disp_option(opt);

    map<string, Box> conc_reg;

    // For each function in the prod and child group that is not the
    // output figure out the concrete bounds

    set<string> group_mem;
    vector<string> prod_funcs;
    if (opt.prod_group != "") {
        for (auto &f: groups[opt.prod_group]) {
            if (!(f.is_lambda() && func_size[f.name()] < 0))
                prod_funcs.push_back(f.name());
            group_mem.insert(f.name());
        }
    }

    for (auto &f: groups[opt.cons_group]) {
        if (f.name() != opt.cons_group &&
            !(f.is_lambda() && func_size[f.name()] < 0)) {
            prod_funcs.push_back(f.name());
            group_mem.insert(f.name());
        }
    }

    vector<string> group_inputs;

    for(auto &f: group_mem) {
        FindAllCalls find;
        analy.env[f].accept(&find);
        for(auto &c: find.calls) {
            if (group_mem.find(c) == group_mem.end())
                group_inputs.push_back(c);
        }
    }

    vector<pair<int, int> > bounds;
    vector<bool> eval;

    const vector<string> &args = analy.env[opt.cons_group].args();
    assert(opt.tile_sizes.size() == args.size());

    vector<int> &dim_estimates_cons = func_pure_dim_estimates[opt.cons_group];

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

    // Evaluate input reuse
    pair<float, float>  eval_reuse;
    eval_reuse = evaluate_reuse(opt.cons_group, group_inputs, opt.tile_sizes, false);

    // Count the number of tiles
    long long estimate_tiles = 1;
    int num_ele_per_tile = 1;
    float partial_tiles = 1;
    for (unsigned int i = 0; i < args.size(); i++) {
        if (opt.tile_sizes[i] != -1) {
            estimate_tiles *= std::ceil((float)dim_estimates_cons[i]/opt.tile_sizes[i]);
            partial_tiles *= (float)dim_estimates_cons[i]/opt.tile_sizes[i];
            num_ele_per_tile *= opt.tile_sizes[i];
        }
    }

    conc_reg = analy.concrete_dep_regions(opt.cons_group, eval,
                                          analy.func_dep_regions, bounds);

    //disp_regions(conc_reg);

    // Cost model

    // We currently assume a two level memory model. The fast_mem_size field in
    // the arch parameters gives the size of the fast memory. Additionally, the
    // ratio of load from fast memory vs slow memory is encoded in the machine
    // parameters.

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
        inter_s = region_size(mem_reg, analy.env, analy.func_dep_regions, gpu_schedule);
    }

    long long saved_mem = 0;

    vector<string> out_of_cache_prods;
    for (auto &p: prod_funcs) {
        if (func_size[p] < 0) {
            // This option cannot be evaluated so discaring the option
            opt.benefit = -1;
            opt.redundant_work = -1;
            return;
        }
        if (func_size[p] > arch_params.fast_mem_size || l == Partitioner::INLINE)
            out_of_cache_prods.push_back(p);
    }

    for (auto &f: prod_funcs) {
        long long data = data_from_group(f, analy.env, func_calls,
                                         func_size, out_of_cache_prods);
        assert(data >= 0);
        saved_mem += data;
    }

    //float total_work = work_per_tile * partial_tiles;

    // This is more accurate since partial tiles are handled by shifting
    // and computing a full tile.
    float total_work = work_per_tile * estimate_tiles;

    long long data = data_from_group(opt.cons_group, analy.env,
                                     func_calls, func_size,
                                     out_of_cache_prods);
    assert(data >= 0);
    saved_mem += data;

    if (debug_info) {

        disp_regions(prod_comp);
        std::cerr << "Work per tile:" << work_per_tile << std::endl;
        std::cerr << "Num tiles:" << estimate_tiles << std::endl;
        std::cerr << "Partial tiles:" << partial_tiles << std::endl;
        std::cerr << "Total work:" << total_work << std::endl;
        std::cerr << "Original work:" << original_work << std::endl;
        std::cerr << "Saved mem:" << saved_mem << std::endl;

        std::cerr << "Intermediate size:" << inter_s << std::endl;
        std::cerr << "Redundant work:" <<
                            (total_work - original_work) << std::endl;
    }

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

    if (debug_info)
        std::cerr << "Estimated benefit:" << opt.benefit << std::endl;

    if ((arch_params.parallelism > estimate_tiles) && opt.prod_group != "") {
        // Option did not satisfy the parallelism constraint
        opt.benefit = -1;
    }

    //std::cerr << opt.prod_group << "," << opt.cons_group << std::endl;
    //std::cerr << saved_mem << "," << eval_reuse.first << "," << input_reuse[opt.prod_group] << std::endl;
    //if (saved_mem + eval_reuse.first < input_reuse[opt.prod_group]) {
    //    opt.benefit = -1;
    //}

    if (gpu_schedule && l != Partitioner::INLINE &&
        (((num_ele_per_tile) < arch_params.target_threads_per_block))) {
        // Constrain the number of elements in a tile to be atleast
        // target threads per block size
        opt.benefit = -1;
    }

    if (opt.prod_group != "") {
        if (l == Partitioner::INLINE) {
            assert(!group_sched[opt.prod_group].fusion);
            assert(!group_sched[opt.prod_group].locality);
        } else {
            assert(!group_sched[opt.prod_group].locality);
        }
        if (debug_info) {
            std::cerr << "Producer group:" << std::endl;
            for (auto &f: groups[opt.prod_group])
                std::cerr << f.name() << std::endl;
            std::cerr << "Saved mem:"
                      << group_sched[opt.prod_group].saved_mem << std::endl;
            std::cerr << "Redundant work:"
                      << group_sched[opt.prod_group].redundant_work << std::endl;
            std::cerr << "Producer benefit:"
                      << group_sched[opt.prod_group].benefit << std::endl;
        }
    }

    if (l == Partitioner::INLINE) {
        assert(!group_sched[opt.cons_group].fusion);
        assert(!group_sched[opt.cons_group].locality);
    } else {
        assert(!group_sched[opt.cons_group].locality);
    }

    if (debug_info) {
        std::cerr << std::endl << "Consumer group:" << std::endl;
        for (auto &f: groups[opt.cons_group])
            std::cerr << f.name() << std::endl;

        std::cerr << "Saved mem:"
            << group_sched[opt.cons_group].saved_mem << std::endl;
        std::cerr << "Redundant work:"
            << group_sched[opt.cons_group].redundant_work << std::endl;
        std::cerr << "Consumer benefit:"
            << group_sched[opt.cons_group].benefit << std::endl;
    }

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
    if (debug_info)
        std::cerr << std::endl << "Final benefit:" << opt.benefit << std::endl;
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
        // Flip a coin and skip evaluating the option. Will also make the
        // auto tuning runs faster.
        //
        // This will change the order of choices from greedy considering
        // all the chocies to greedy only considering a subset of the
        // choices at any point. With this we are going against the cost
        // model in a bounded fashion.
        //
        // This will also achive the goal of early stopping since there
        // might no benefit in fusing the subset of choice that are being
        // considered
        if (random_seed && rand()%2==0) {
            continue;
        }
        for (auto &c: children[p.first]) {

            // Get the output function of the child group
            Function output = analy.env[c];
            const vector<string> &args = output.args();

            Option cand_opt;
            cand_opt.prod_group = p.first;
            cand_opt.cons_group = c;

            // Weird to inline into boundary conditions
            if (output.is_boundary()) {
                overall_benefit = -1;
                break;
            }

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
                float prod_size = func_size[cand_opt.prod_group] *
                                  get_func_out_size(analy.env[cand_opt.prod_group]);
                float prod_data = func_size[cand_opt.prod_group] *
                                  func_cost[cand_opt.prod_group].second;

                // Check if the prod_group is a reduction. If it is a reduction
                // then by inlining it we can no longer schedule the individual
                // dimensions effectively.
                // The number of loads saved is relatively insignificant
                if (prod_size < 0.01 * prod_data) {
                    if (debug_info) {
                        std::cerr << "Inlining avoided " <<  cand_opt.prod_group
                                  << " " << cand_opt.cons_group << std::endl;
                    }
                    overall_benefit = -1;
                    break;
                }
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
    // imbalance.
    //
    // 4) Does the fusion limit vectorization. Reordering function dimensions
    // and modifying data layout have significant interactions with
    // vectorization. As a first pass the goal is to not miss any obvious
    // vectorization and does not create new oportunities.  Generating a
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

    // Restricting tile sizes when trying to emulate polymage mode
    const char *tile_size_1_var = getenv("HL_POLYMAGE_TILE_SIZE1");

    int tile_size_1 = 0;
    if (tile_size_1_var) {
        tile_size_1 = atoi(tile_size_1_var);
    }

    const char *tile_size_2_var = getenv("HL_POLYMAGE_TILE_SIZE2");

    int tile_size_2 = 0;
    if (tile_size_2_var) {
        tile_size_2 = atoi(tile_size_2_var);
    }

    vector<int> size_variants = {256, 128, 64, 32, 16, 8, 4};

    bool polymage_mode = false;
    if (tile_size_2_var && tile_size_1_var)  {
        size_variants.clear();
        size_variants.push_back(tile_size_1);
        size_variants.push_back(tile_size_2);
        polymage_mode = true;
    }

    if(random_seed) {
        vector<int> rand_variants;
        for (auto &s: size_variants) {
            if (rand()%2 == 0) {
                rand_variants.push_back(s);
            }
        }
        size_variants.clear();
        size_variants = rand_variants;
    }

    Option best_opt;

    for (auto &p: cand_pairs) {
        pair<string, string> key = make_pair(p.first, p.second);
        Option cand_best_opt;

        // Flip a coin and skip evaluating the option. Will also make the
        // auto tuning runs faster.
        //
        // This will change the order of choices from greedy considering
        // all the chocies to greedy only considering a subset of the
        // choices at any point. With this we are going against the cost
        // model in a bounded fashion.
        //
        // This will also achive the goal of early stopping since there
        // might no benefit in fusing the subset of choices that are being
        // considered
        if (random_seed && rand()%2==0) {
            continue;
        }

        // Check if the pair has been evaluated before
        if (option_cache.find(key) != option_cache.end()) {
            //std::cerr << "Hit:" << p.first << "," << p.second << std::endl;
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
        vector<int> &dim_estimates_prod = func_pure_dim_estimates[p.first];

        const vector<string> &args_prod = analy.env[p.first].args();
        for (unsigned int i = 0; i < args_prod.size(); i++) {
            if (dim_estimates_prod[i] == -1) {
                // This option cannot be evaluated so discaring the option
                invalid = true;
            }
        }

        // Weird to fuse into boundary conditions
        if (output.is_boundary() || !output.is_pure()) {
            invalid = true;
        }

        if (gpu_schedule && analy.env[p.first].is_boundary()) {
            invalid = true;
        }

        float prod_size = func_size[p.first] *
                          get_func_out_size(analy.env[p.first]);
        float prod_data = func_size[p.first] * func_cost[p.first].second;

        // Check if the prod_group is a reduction. If it is a reduction
        // then by grouping it we can no longer schedule the individual
        // dimensions effectively.
        // The number of loads saved is relatively insignificant
        // std::cerr << p.first << "," << input_reuse[p.first] << std::endl;
        if (prod_size < 0.01 * prod_data) {
            if (debug_info) {
                std::cerr << "Grouping avoided " <<  p.first
                          << " " << p.second << std::endl;
            }
            invalid = true;
        }

        cand_best_opt.prod_group = p.first;
        cand_best_opt.cons_group = p.second;

        if (!invalid && !polymage_mode) {
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

            if (debug_info) {
                std::cerr << "Analyzing dims for reuse" << std::endl;
                for (unsigned int i = 0; i < args.size(); i++) {
                    std::cerr << args[i] << " Reuse/Redundant Work " << reuse[i]
                              << std::endl;
                }
            }

            /*
            vector<int> new_variants = {1, 4, 8, 16, 32, 64, 128, 256};
            // Reuse based tiling
            for (unsigned int i = 0; i < args.size(); i++) {
                for (auto &s: new_variants) {
                    vector<int> tile_sizes(args.size());
                    Option opt;
                    opt.prod_group = p.first;
                    opt.cons_group = p.second;
                    for (unsigned int j = 0; j < args.size(); j++) {
                        unsigned int rank = 0;
                        for (unsigned int k = 0; k < args.size(); k++) {
                            // Count up the number of dimensions with reuse
                            // greater than that of j
                            if (k!=j) {
                                if (reuse[k] > reuse[j] ||
                                        (reuse[k] == reuse[j] && k < j))
                                rank++;
                            }
                        }
                        if (rank < i)
                           // All the dimensions ranked < i in the reuse order
                           // get max tile size
                            tile_sizes[j] = new_variants[new_variants.size() - 1];
                        else if (rank == i)
                            // Vary tile sizes for dimension ranked <=i
                            tile_sizes[j] = s;
                        else
                            // All the dimensions ranked > i in the reuse order
                            // get min tile size
                            tile_sizes[j] = new_variants[0];

                        if (j == 0)
                            tile_sizes[j] = std::min(64, s);

                        std::cout << args[j] << " tile size " << tile_sizes[j]
                                  << std::endl;
                    }

                    opt.tile_sizes = tile_sizes;
                    evaluate_option(opt, Partitioner::FAST_MEM);

                    if (cand_best_opt.benefit < opt.benefit) {
                        cand_best_opt = opt;
                    }
                }
            }*/

            // From the outer to the inner most argument
            for (int i = (int)args.size() - 1; i >= 0; i--) {
                for (auto s: size_variants) {
                    Option opt;
                    opt.prod_group = p.first;
                    opt.cons_group = p.second;
                    opt.reuse = reuse;

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

                        if (j == 0) {
                            if (gpu_schedule)
                                opt.tile_sizes.push_back(std::max(curr_size, arch_params.vec_len));
                            else
                                opt.tile_sizes.push_back(std::max(curr_size, 64));
                        }
                        else
                            opt.tile_sizes.push_back(curr_size);
                    }

                    evaluate_option(opt, Partitioner::FAST_MEM);

                    if (cand_best_opt.benefit < opt.benefit) {
                        cand_best_opt = opt;
                    }
                }
            }
        }

        if (polymage_mode) {
            Option opt;
            opt.prod_group = p.first;
            opt.cons_group = p.second;
            for (int i = (int)args.size() - 1; i >= 0; i--) {
                unsigned int num_tiled_dims = 0;
                if (num_tiled_dims < size_variants.size()) {
                    opt.tile_sizes.push_back(size_variants[num_tiled_dims]);
                    num_tiled_dims++;
                } else {
                    opt.tile_sizes.push_back(-1);

                }
                opt.reuse.push_back(-1);
            }
            evaluate_option(opt, Partitioner::FAST_MEM);

            if (cand_best_opt.benefit < opt.benefit) {
                cand_best_opt = opt;
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
                                vector<int> &tile_sizes, bool unit_tile) {

    const vector<string> &pure_args = analy.env[group].args();
    unsigned int num_pure_args = pure_args.size();

    bool is_update = !analy.env[group].is_pure();
    vector<pair<int, int> > bounds;
    vector<bool> eval;

    map<string, int> &dim_estimates = func_dim_estimates[group];
    Box cons_box;

    int parallel_tile_iter = 1;
    int parallel_tiles = 1;

    long long tile_size = 1;
    for (unsigned int i = 0; i < tile_sizes.size(); i++) {
        string arg_name = "";
        if (i < num_pure_args) {
            arg_name = pure_args[i];
        } else {
            vector<string> &u_args = analy.update_args[group];
            int u_index = (int)i - num_pure_args;
            arg_name = u_args[u_index];
        }
        assert(dim_estimates.find(arg_name) != dim_estimates.end());
        if (tile_sizes[i] != -1) {
            // Check if the bounds allow for tiling with the given tile size
            if (dim_estimates[arg_name] > tile_sizes[i]) {
                if (is_update) {
                    // Make the tile size the smallest multiple of dimension
                    // less than the tile size
                    if (dim_estimates[arg_name]%tile_sizes[i] != 0) {
                        for (int s = tile_sizes[i] - 1; s > 0; s--) {
                            if (dim_estimates[arg_name]%s == 0) {
                                tile_sizes[i] = s;
                                break;
                            }
                        }
                    }
                }
                assert(tile_sizes[i] > 0);
                bounds.push_back(make_pair(0, tile_sizes[i] - 1));
                if (i < num_pure_args) {
                    cons_box.push_back(Interval(0, tile_sizes[i] - 1));
                    parallel_tile_iter *= tile_sizes[i];
                }
                tile_size = tile_size * (tile_sizes[i]);
            } else {
                // If the dimension is too small do not tile it and set the
                // extent of the bounds to that of the dimension estimate
                tile_sizes[i] = -1;
                bounds.push_back(make_pair(0, dim_estimates[arg_name] - 1));
                if (i < num_pure_args) {
                    cons_box.push_back(Interval(0, dim_estimates[arg_name] - 1));
                    parallel_tile_iter *= dim_estimates[arg_name];
                }
                tile_size = tile_size * (dim_estimates[arg_name]);
            }
        } else {
            bounds.push_back(make_pair(0, dim_estimates[arg_name] - 1));
            if (i < num_pure_args) {
                cons_box.push_back(Interval(0, dim_estimates[arg_name] - 1));
                parallel_tile_iter *= dim_estimates[arg_name];
            }
            tile_size = tile_size * (dim_estimates[arg_name]);
        }
        eval.push_back(true);
    }

    // Count the number of tiles
    long long estimate_tiles = 1;
    for (unsigned int i = 0; i < tile_sizes.size(); i++) {
        string arg_name = "";
        if (i < num_pure_args) {
            arg_name = pure_args[i];
        } else {
            vector<string> &u_args = analy.update_args[group];
            int u_index = (int)i - num_pure_args;
            arg_name = u_args[u_index];
        }
        if (tile_sizes[i] != -1) {
            estimate_tiles *= std::ceil((float)dim_estimates[arg_name]/tile_sizes[i]);
            if (i < num_pure_args) {
                parallel_tiles *= std::ceil((float)dim_estimates[arg_name]/tile_sizes[i]);
            }
        }
    }

    map<string, Box> conc_reg =
        analy.concrete_dep_regions(group, eval,
                                   analy.func_partial_dep_regions, bounds);

    if (!unit_tile) {
        if (parallel_tiles < arch_params.parallelism)
            return make_pair(-1, -1);

        if (gpu_schedule &&
                (parallel_tile_iter < arch_params.target_threads_per_block ||
                 parallel_tile_iter > arch_params.max_threads_per_block)) {
            return make_pair(-1, -1);
        }
    }

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
                                    analy.func_dep_regions, false);

    float total_inter = 0;
    if (group_inter < 0 || input_inter < 0)
        return make_pair(-1, -1);
    else
        total_inter = group_inter + input_inter;

    float unit_input_data = 0;
    // Evalute the intermediate storage for computing in unit tiles
    if (tile_size > 1) {
        if (debug_info) {
            disp_regions(group_mem_reg);
            disp_regions(input_mem_reg);
            std::cerr << "Config :[";
            for (auto &s: tile_sizes)
                std::cerr << s << ",";
            std::cerr << "]" << std::endl;
            std::cerr << "Total intermediate size:" << total_inter  << std::endl;
            std::cerr << "Input intermediate size:" << input_inter  << std::endl;
        }
        vector<int> unit_sizes;
        for (unsigned int i = 0; i < tile_sizes.size(); i++)
            unit_sizes.push_back(1);
        pair<float, float> unit = evaluate_reuse(group, group_inputs,
                                                 unit_sizes, true);
        assert(unit.first < 1);
        unit_input_data = unit.second;
    } else {
        if (debug_info)
            std::cerr << "Unit input size:" << input_inter << std::endl;
        unit_input_data = input_inter;
    }

    // Compute the reuse within a tile
    float reuse =  estimate_tiles * (unit_input_data * tile_size - input_inter);
    float realized_reuse = -1;
    if (total_inter <= arch_params.fast_mem_size) {
        realized_reuse = reuse;
    }

    if (tile_size > 1 && debug_info)
        std::cerr << "Reuse:" << realized_reuse << std::endl;

    return make_pair(realized_reuse, input_inter);
}

vector<float>
Partitioner::get_input_reuse(Function f, vector<string> &inputs) {

    const vector<string> &args = f.args();
    vector<string> &u_args = analy.update_args[f.name()];
    vector<float> reuse;

    unsigned int num_args = args.size() + u_args.size();
    for (unsigned int i = 0; i < num_args; i++)
        reuse.push_back(-1);
    for (unsigned int i = 0; i < num_args; i++) {

        vector<pair<int, int> > bounds;
        vector<bool> eval;

        map<string, int> dim_estimates =
                        get_dim_estimates(f.name(), pipeline_bounds, analy.env);

        for (unsigned int j = 0; j < num_args; j++) {
            string arg_name = "";
            if (j < args.size())
                arg_name = args[j];
            else
                arg_name = u_args[j - args.size()];
            if (j==i) {
                bounds.push_back(make_pair(0, 0));
                //std::cerr << "Varying " <<  arg_name << std::endl;
            }
            else {
                assert(dim_estimates.find(arg_name) != dim_estimates.end());
                //std::cerr << arg_name << ":"  << dim_estimates[arg_name]  << std::endl;
                bounds.push_back(make_pair(0, dim_estimates[arg_name] - 1));
            }
            eval.push_back(true);
        }

        vector< map<string, Box> > conc_overlaps =
                        analy.concrete_overlap_regions(f.name(), eval,
                                                       analy.func_partial_overlaps,
                                                       bounds);
        /*
        map<string, Box> conc_reg =
                        analy.concrete_dep_regions(f.name(), eval,
                                                   analy.func_partial_dep_regions,
                                                   bounds); */

        float input_overlap = 0;
        //disp_regions(conc_overlaps[i]);
        //disp_regions(analy.func_partial_dep_regions[f.name()]);
        for (auto &in: inputs) {
            assert(conc_overlaps[i].find(in) != conc_overlaps[i].end());
            float area = box_area(conc_overlaps[i][in]);
            assert(area >= 0);
            input_overlap += area * get_func_out_size(analy.env[in]);
        }
        // Account for reuse of the reduction output buffer
        if (conc_overlaps[i].find(f.name()) != conc_overlaps[i].end()) {
            float area = box_area(conc_overlaps[i][f.name()]);
            assert(area >= 0);
            input_overlap += area * get_func_out_size(analy.env[f.name()]);
        }
        reuse[i] = input_overlap;
    }
    return reuse;
}

void Partitioner::tile_for_input_locality(bool init_pipeline_reuse) {

    for(auto &g: groups) {
        // Skip something that has been tiled for fusion
        // Skip functions that are not reductions
        if (group_sched[g.first].fusion || (analy.reductions.find(g.first) ==
                                            analy.reductions.end())) {
            if (debug_info)
                std::cerr << "Skipped " << g.first << std::endl;
            continue;
        }

        // TODO Reason about instances where things have been inlined
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

        if (debug_info) {
            std::cerr << "Inputs for group " << g.first << ":" << std::endl;
            for(auto &in: group_inputs)
                std::cerr << in << std::endl;
            std::cerr << std::endl;
        }

        // Restricting tile sizes when trying to emulate polymage mode
        const char *tile_size_1_var = getenv("HL_POLYMAGE_TILE_SIZE1");

        bool polymage_mode = false;
        int tile_size_1 = 0;
        if (tile_size_1_var) {
            tile_size_1 = atoi(tile_size_1_var);
        }

        const char *tile_size_2_var = getenv("HL_POLYMAGE_TILE_SIZE2");

        int tile_size_2 = 0;
        if (tile_size_2_var) {
            tile_size_2 = atoi(tile_size_2_var);
        }

        vector<int> size_variants = {256, 128, 64, 32, 16, 8, 4};

        if (tile_size_2_var && tile_size_1_var)  {
            size_variants.clear();
            size_variants.push_back(tile_size_1);
            size_variants.push_back(tile_size_2);
            polymage_mode = true;
        }

        if(random_seed) {
            // Truncate tile size variants randomly
            vector<int> rand_variants;
            for (auto &s: size_variants) {
                if (rand()%2 == 0) {
                    rand_variants.push_back(s);
                }
            }
            size_variants.clear();
            size_variants = rand_variants;
        }

        // For the dimensions with reuse along multiple dimensions tile
        // the dimensions in such a way that the reuse is maximized and
        // the porition of inputs fit in fast memory

        // Get the output function of the child group
        Function output = analy.env[g.first];
        const vector<string> &args = output.args();

        bool invalid = false;
        for(auto &in: group_inputs) {
            vector<int> &dim_estimates_prod = func_pure_dim_estimates[in];

            const vector<string> &args_prod = analy.env[in].args();
            for (unsigned int i = 0; i < args_prod.size(); i++) {
                if (dim_estimates_prod[i] == -1) {
                    // This option cannot be evaluated so discaring the option
                    invalid = true;
                }
            }
        }

        if (!invalid && !polymage_mode) {
            vector<float> reuse = get_input_reuse(analy.env[g.first],
                                                  group_inputs);

            vector<string> &u_args = analy.update_args[g.first];
            unsigned int num_args = args.size() + u_args.size();
            //std::cerr << "Analyzing dims for locality" << std::endl;
            for (unsigned int i = 0; i < num_args; i++) {
                string arg_name = "";
                if (i < args.size())
                    arg_name = args[i];
                else
                    arg_name = u_args[i - args.size()];
                //std::cerr << arg_name << " Reuse " << reuse[i]
                //          << std::endl;
            }

            float best_reuse = 0;
            vector<int> best_tiling;

            vector<int> new_variants = {1, 4, 8, 16, 32, 64, 128, 256, 512};

            if(random_seed) {
                // Truncate tile size variants randomly
                vector<int> rand_variants;
                for (auto &s: new_variants) {
                    if (rand()%2 == 0) {
                        rand_variants.push_back(s);
                    }
                }
                new_variants.clear();
                new_variants = rand_variants;
            }

            map<string, unsigned int> dim_rank;
            for (unsigned int j = 0; j < num_args; j++) {
                unsigned int rank = 0;
                for (unsigned int k = 0; k < num_args; k++) {
                    // Count up the number of dimensions with reuse
                    // greater than that of j
                    if (k!=j) {
                        if (reuse[k] > reuse[j] ||
                                (reuse[k] == reuse[j] && k < j))
                            rank++;
                    }
                }

                string arg_name = "";
                if (j < args.size())
                    arg_name = args[j];
                else
                    arg_name = u_args[j - args.size()];

                dim_rank[arg_name] = rank;
            }

            //for (auto &rank: dim_rank)
            //    std::cerr << rank.first << "," << rank.second << std::endl;

            // Reuse based tiling
            for (unsigned int i = 0; i < num_args; i++) {
                for (auto &s: new_variants) {
                    vector<int> tile_sizes(num_args);
                    for (unsigned int j = 0; j < num_args; j++) {

                        string arg_name = "";
                        if (j < args.size())
                            arg_name = args[j];
                        else
                            arg_name = u_args[j - args.size()];

                        unsigned int rank = dim_rank[arg_name];
                        if (rank < i)
                            // All the dimensions ranked < i in the reuse order
                            // get max tile size
                            tile_sizes[j] = -1;
                        else if (rank == i)
                            // Vary tile sizes for dimension ranked <=i
                            tile_sizes[j] = s;
                        else
                            // All the dimensions ranked > i in the reuse order
                            // get min tile size
                            tile_sizes[j] = new_variants[0];

                        if (j == 0) {
                            if (gpu_schedule) {
                                tile_sizes[j] = std::max(s, arch_params.vec_len);
                            } else {
                                tile_sizes[j] = -1;
                            }
                        }

                        //std::cerr << arg_name << " tile size " << tile_sizes[j]
                        //          << std::endl;
                    }

                    /*
                    std::cerr << g.first << " Config:" << "[";
                    for (auto &t: tile_sizes)
                        std::cerr << t << ",";
                    std::cerr <<  "]" << std::endl;
                    */

                    pair<float, float>  eval;
                    eval = evaluate_reuse(g.first, group_inputs, tile_sizes,
                                          false);
                    if (eval.first > best_reuse) {
                        best_reuse = eval.first;
                        best_tiling = tile_sizes;
                    }
                }
            }

            for (unsigned int i = 0; i < num_args; i++) {
                for (auto &s: new_variants) {
                    vector<int> tile_sizes(num_args);
                    for (unsigned int j = 0; j < num_args; j++) {

                        string arg_name = "";
                        if (j < args.size())
                            arg_name = args[j];
                        else
                            arg_name = u_args[j - args.size()];

                        unsigned int rank = dim_rank[arg_name];
                        if (rank <= i)
                            // Vary tile sizes for dimension ranked <=i
                            tile_sizes[j] = s;
                        else
                            // All the dimensions ranked > i in the reuse order
                            // get min tile size
                            tile_sizes[j] = new_variants[0];

                        if (j == 0) {
                            if (gpu_schedule) {
                                tile_sizes[j] = std::max(s, arch_params.vec_len);
                            } else {
                                tile_sizes[j] = -1;
                            }
                        }
                        //std::cerr << arg_name << " tile size " << tile_sizes[j]
                        //          << std::endl;
                    }

                    /*
                    std::cerr << g.first << " Config:" << "[";
                    for (auto &t: tile_sizes)
                        std::cerr << t << ",";
                    std::cerr <<  "]" << std::endl;
                    */

                    pair<float, float>  eval;
                    eval = evaluate_reuse(g.first, group_inputs, tile_sizes,
                                          false);
                    if (eval.first > best_reuse) {
                        best_reuse = eval.first;
                        best_tiling = tile_sizes;
                    }
                }
            }

            if (debug_info) {
                std::cerr << g.first << " Best tiling:" << "[";
                for (auto &t: best_tiling)
                    std::cerr << t << ",";
                std::cerr <<  "]" << std::endl;
                std::cerr <<  best_reuse << std::endl;
            }

            if (!init_pipeline_reuse){
                if (best_reuse > 0) {
                    group_sched[g.first].tile_sizes = best_tiling;
                    assert(!group_sched[g.first].fusion);
                    group_sched[g.first].locality = true;
                    group_sched[g.first].reuse = reuse;

                } else {
                    assert(!group_sched[g.first].fusion);
                    group_sched[g.first].locality = false;
                }
            } else {
                input_reuse[g.first] = best_reuse;
            }
        }
    }
}

void disp_function_value_bounds(const FuncValueBounds &func_val_bounds) {

	for (auto& kv: func_val_bounds) {
        std::cerr << kv.first.first << "," << kv.first.second << ":"
                  << "(" << kv.second.min  << ","  << kv.second.max << ")"
                  << std::endl;
    }
}

void disp_schedule_and_storage_mapping(map<string, Function> &env) {
    // Names of all the functions in the environment and their schedules
    for (auto& kv : env) {
        std::cerr << schedule_to_source(kv.second,
                                        kv.second.schedule().compute_level(),
                                        kv.second.schedule().store_level())
                  << std::endl;
    }
    std::cerr << std::endl;
}

void disp_inlines(map<string, vector<string> > &inlines) {
    for (auto& in: inlines) {
        std::cerr << in.first << "-> [";
        for (auto& c: in.second)
            std::cerr << c << " ";
        std::cerr << "]" << std::endl;
    }
}

map<string, vector<string>>
simple_inline(map<string, vector<const Call*>> &all_calls,
              map<string, vector<string> > &consumers,
              map<string, Function> &env,
              const vector<Function> &outputs) {
    map<string, vector<string> > inlines;
    for (auto& fcalls: all_calls) {
        // Check if all arguments to the function call over all the calls are
        // one-to-one. If this holds and the number of calls == 1 it is a good
        // candidate for inlining.

        bool is_output = false;
        for (auto &out: outputs) {
            if(out.name() == fcalls.first) {
                is_output = true;
            }
        }

        if (is_output)
            continue;

        bool all_one_to_one = false;
        int num_calls = 0;
        /* TODO one_to_one is no longer available
        for (auto& call: fcalls.second){
            num_calls++;
            for(auto& arg: call->args){
                // Skip casts to an integer there seems to be a bug lurking
                // in is_one_to_one
                bool one_to_one = (!arg.as<Cast>()) && is_one_to_one(arg);
                all_one_to_one = (call->name != fcalls.first) &&
                                  all_one_to_one && (one_to_one
                                                    || is_simple_const(arg));
            }
        }
        */
        if (consumers[fcalls.first].size() == 1 &&
            all_one_to_one && num_calls == 1) {
            inlines[fcalls.first].push_back(consumers[fcalls.first][0]);
            env[fcalls.first].schedule().store_level().var = "";
            env[fcalls.first].schedule().compute_level().var = "";
        }
        if (env[fcalls.first].is_boundary() || env[fcalls.first].is_lambda()) {
            //assert(consumers[fcalls.first].size() == 1);
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

void unroll_dim(Schedule &sched, int dim) {
    vector<Dim> &dims = sched.dims();
    dims[dim].for_type = ForType::Unrolled;
}

void reorder_dims(Schedule &sched, vector<string> &var_order) {
    vector<Dim> &dims = sched.dims();
    vector<Dim> new_dims;
    for (auto &v: var_order) {
        for (int i = 0; i < (int)dims.size() - 1; i++)
            if (v == dims[i].var)
                new_dims.push_back(dims[i]);
    }
    for (unsigned int i = 0; i < dims.size(); i++) {
        if (find(var_order.begin(), var_order.end(), dims[i].var) ==
            var_order.end()) {
            new_dims.push_back(dims[i]);
        }
    }
    dims.clear();
    for (unsigned int i = 0; i < new_dims.size(); i++) {
        dims.push_back(new_dims[i]);
    }
}

string get_spatially_coherent_innermost_dim(Function &f, const UpdateDefinition &u) {
    // Collect all the load args
    FindAllCallArgs find;
    for (auto &e: u.values)
        e.accept(&find);
    for (auto &arg: u.args)
        arg.accept(&find);

    string inner_store_var = f.schedule().storage_dims()[0].var;
    int inner_dim = 0;
    assert(f.args()[0] == inner_store_var);
    // For all the loads find the stride of the loop corresponding
    // to the innermost storage dim across inputs and outputs
    for(auto& larg: find.load_args) {
        for (unsigned int dim = 0; dim < larg.size(); dim++) {
            UsesVarCheck check(f.args()[inner_dim]);
            Expr acc = larg[dim];
            acc.accept(&check);
            //std::cout << larg[dim] << std::endl;
            if (dim > 0 && check.uses_var) {
                //std::cout << dim << " "  << f.args()[inner_dim]
                //          << " " << acc << std::endl;
                // Stick to the ordering specified in the program when there
                // is disruption in spaital locality
                return u.schedule.dims()[0].var;
            }
        }
    }
    return inner_store_var;
}

void reorder_by_reuse(Schedule &sched, map<string, float> &reuse,
                      string innermost_dim,
                      map<string, int> &dim_estimates, int unroll_len) {
    vector<string> var_order;
    vector<Dim> &dims = sched.dims();
    assert(reuse.size() + 1 == dims.size());
    map<string, int> dim_rank;
    for(int j = 0; j < (int)dims.size() - 1; j++) {
        int rank = 0;
        for (int k = 0; k < (int)dims.size() - 1; k++) {
            // Count up the number of dimensions with reuse
            // significantly less than that of j
            if (k!=j) {
                bool better_reuse =
                    (reuse[dims[j].var] > 2 * reuse[dims[k].var]);
                if (better_reuse || (j < k) )
                    rank++;
            }
        }
        dim_rank[dims[j].var] = rank;
    }

    //for (auto &rank: dim_rank)
    //    std::cerr << rank.first << "," << rank.second << std::endl;

    while (!dim_rank.empty()) {
        int largest_rank = -1;
        for (auto &rank: dim_rank) {
            if (rank.second > largest_rank)
                largest_rank = rank.second;
        }

        string var;
        for (int i = 0; i < (int)dims.size() - 1; i++) {
            if (dim_rank.find(dims[i].var) != dim_rank.end()
                    && dim_rank[dims[i].var] == largest_rank) {
                var = dims[i].var;
                dim_rank.erase(var);
                break;
            }
        }
        // If the update definition is pure in the dimension corresponding
        // to the innermost storage dimension, vectorize that to get a
        // dense vector store instead of a scatter.
        if (var != innermost_dim) {
            var_order.push_back(var);
        }
    }

    // Insert the innermost storage dim at a position where all the inner loops
    // can be unrolled
    int vector_dim = 0;
    int iterations_unrolled = 1;
    for (unsigned int i = 0; i < var_order.size(); i++) {
        if (dim_estimates[var_order[i]] * iterations_unrolled < unroll_len) {
            iterations_unrolled *= dim_estimates[var_order[i]];
            vector_dim++;
        }
        else
            break;
    }

    var_order.insert(var_order.begin() + vector_dim, innermost_dim);

    //std::cerr << "Reuse order :";
    //for (auto &v: var_order)
    //    std::cerr << v << ",";
    //std::cerr << std::endl;
    reorder_dims(sched, var_order);
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

void move_dim_before_innermost(Schedule &sched, int dim) {
    vector<Dim> &dims = sched.dims();
    assert(dims.size() > 2);
    dims.insert(dims.begin() + 1, dims[dim]);
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
               map<string, int> &dim_estimates, string prefix) {

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
                   false, Halide::TailStrategy::Auto, Split::SplitVar};
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

void rename_dim(Schedule &sched, int dim, string suffix) {

    vector<Dim> &dims = sched.dims();

    string old_name = dims[dim].var;
    string new_name = old_name + "." + suffix;

    dims[dim].var = new_name;

    bool found = false;
    for (size_t i = sched.splits().size(); i > 0; i--) {
        if (sched.splits()[i-1].is_fuse()) {
            if (sched.splits()[i-1].inner == old_name ||
                sched.splits()[i-1].outer == old_name) {
                assert(false);
            }
            if (sched.splits()[i-1].old_var == old_name) {
                sched.splits()[i-1].old_var = new_name;
                found = true;
                break;
            }
        } else {
            if (sched.splits()[i-1].inner == old_name) {
                sched.splits()[i-1].inner = new_name;
                found = true;
                break;
            }
            if (sched.splits()[i-1].outer == old_name) {
                sched.splits()[i-1].outer = new_name;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        Split split = {old_name, new_name, "", 1, false, Halide::TailStrategy::Auto, Split::RenameVar};
        sched.splits().push_back(split);
    }
}

void split_dim_gpu(Schedule &sched, int dim, int split_size,
                   map<string, int> &dim_estimates,
                   string block_name, string thread_name) {

    vector<Dim> &dims = sched.dims();
    string old_name = dims[dim].var;
    string outer_name = old_name + "." + block_name;
    string inner_name = old_name + "." + thread_name;

    dims.insert(dims.begin() + dim, dims[dim]);
    dims[dim].var = inner_name;
    dims[dim+1].var = outer_name;
    dims[dim+1].pure = dims[dim].pure;

    dims[dim+1].for_type = ForType::Parallel;
    dims[dim+1].device_api = DeviceAPI::Default_GPU;

    dims[dim].for_type = ForType::Parallel;
    dims[dim].device_api = DeviceAPI::Default_GPU;

    // Add the split to the splits list
    Split split = {old_name, outer_name, inner_name, split_size,
                   false, Halide::TailStrategy::Auto, Split::SplitVar};
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
                   true, Halide::TailStrategy::Auto, Split::FuseVars};
    sched.splits().push_back(split);
    return fused_name;
}

// Vectorization
void vectorize_dim(Schedule &sched, map<string, int> &dim_estimates,
                   int dim, int vec_width) {
    vector<Dim> &dims = sched.dims();
    if (vec_width != -1) {
        split_dim(sched, dim, vec_width, dim_estimates, "vec");
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

int get_dim_size(Schedule &sched, int dim, map<string, int> &dim_estimates) {
    vector<Dim> &dims = sched.dims();
    int extent = dim_estimates[dims[dim].var];
    return extent;
}

void simple_vectorize(Function &func, map<string, int> &dim_estimates,
                      int inner_dim, int vec_width=-1) {
    // Collect all the load args
    FindCallArgs find;
    func.accept(&find);
    // For all the loads find the stride of the innermost loop
    bool vec = true;
    /* TODO: Need to stop using finite_difference
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
    }*/
    if (vec)
        vectorize_dim(func.schedule(), dim_estimates, inner_dim, vec_width);
}

void vectorize_update(Function &func, int stage,
        map<string, int> &dim_estimates, int vec_len,
        set<string> &par_vars) {
    if (func.dimensions() == 0) {
        // Can't vectorize a reduction onto a scalar.
        return;
    }

    Schedule &s = func.update_schedule(stage);
    const UpdateDefinition &u = func.updates()[stage];
    vector<Dim> &dims = s.dims();

    int iterations_unrolled = 1;
    int unroll_limit = 16;
    for (unsigned int dim = 0; dim < dims.size(); dim++) {
        bool dim_par = can_parallelize_rvar(dims[dim].var, func.name(), u);
        dim_par = dim_par || (par_vars.find(dims[dim].var) != par_vars.end());
        if(check_dim_size(s, dim, vec_len, dim_estimates) && dim_par) {
            //move_dim_to_innermost(s, dim);
            //vectorize_dim(s, dim_estimates, 0, vec_len);
            vectorize_dim(s, dim_estimates, dim, vec_len);
            par_vars.insert(dims[dim+1].var);
            break;
        } else if(check_dim_size(s, dim, 8, dim_estimates) && dim_par) {
            vectorize_dim(s, dim_estimates, dim, 8);
            par_vars.insert(dims[dim+1].var);
            break;
        } else if(check_dim_size(s, dim, 4, dim_estimates) && dim_par) {
            vectorize_dim(s, dim_estimates, dim, 4);
            par_vars.insert(dims[dim+1].var);
            break;
        } else {
            int dim_size = get_dim_size(s, dim, dim_estimates);
            iterations_unrolled *= dim_size;
            if (dim_size < vec_len && iterations_unrolled <= unroll_limit) {
                unroll_dim(s, dim);
            }
        }
    }
}

void pick_gpu_thread_dims(Schedule &s,
                          map<string, int> &dim_estimates,
                          int max_threads_per_block,
                          vector<int> &thread_dim_size) {

    std::vector<string> thread_names = {"__thread_id_x",
                                        "__thread_id_y",
                                        "__thread_id_z"};

    vector<Dim> &dims = s.dims();
    int outer_dim = dims.size() - 2;
    int marked_block_dims = 0;
    int num_block_dim = 3;
    int num_threads = 1;
    for (int i = 0; marked_block_dims < num_block_dim && i <= outer_dim; i++) {
        //std::cerr << dims[i].var << "," << dim_estimates[dims[i].var] << "," << num_threads << std::endl;
        int dim_threads = std::max(dim_estimates[dims[i].var], thread_dim_size[marked_block_dims]);
        if (dim_threads * num_threads > max_threads_per_block || !dims[i].pure)
            continue;
        num_threads *= dim_threads;
        thread_dim_size[marked_block_dims] = dim_threads;
        rename_dim(s, i, thread_names[marked_block_dims]);
        dims[i].for_type = ForType::Parallel;
        marked_block_dims++;
    }
}

bool pick_dim_to_parallelize(Function &f, map<string, int> &dim_estimates,
                             int parallelism, int num_tile_dims,
                             int &outer_dim, int& num_fused_dims) {
    // TODO Check which is better fusing the dimensions or moving
    // the right dimension out and parallelizing it
    //std::cout << "Parallel Dim Choice " << f.name() << std::endl;
    vector<Dim> &dims = f.schedule().dims();
    //for (auto &d: dims)
    //    std::cout << d.var << ",";
    //std::cout << std::endl;
    outer_dim = dims.size() - 2;

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

bool check_estimates_on_outputs(const vector<Function> &outputs) {
    bool estimates_avail = true;
    for (auto &out : outputs) {
        const vector<Bound> &estimates = out.schedule().estimates();
        if (estimates.size() != out.args().size()) {
            estimates_avail = false;
            break;
        }
        vector<string> vars = out.args();

        for (unsigned int i = 0; i < estimates.size(); i++) {
            if (std::find(vars.begin(), vars.end(), estimates[i].var) == vars.end()
                    || !((estimates[i].min.as<IntImm>()) &&
                        (estimates[i].extent.as<IntImm>())))  {
                estimates_avail = false;
                break;
            }
        }
    }
    return estimates_avail;
}

map<string, Box> get_group_member_bounds(Partitioner &part, string group,
                                         vector<int> &tile_sizes) {

    vector<pair<int, int> > bounds;
    vector<bool> eval;
    map<string, Box> conc_reg;

    const vector<string> &args = part.analy.env[group].args();
    vector<int> &dim_estimates = part.func_pure_dim_estimates[group];

    for (unsigned int i = 0; i < args.size(); i++) {
        if (tile_sizes[i] != -1) {
            bounds.push_back(make_pair(0, tile_sizes[i] - 1));
        }
        else {
            bounds.push_back(make_pair(0, dim_estimates[i] - 1));
        }
        eval.push_back(true);
    }

    conc_reg = part.analy.concrete_dep_regions(group, eval,
                                               part.analy.func_dep_regions,
                                               bounds);
    return conc_reg;
}

void synthesize_cpu_schedule(string g_name, Partitioner &part,
        map<string, Function> &env,
        map<string, Box> &pipeline_bounds,
        map<string, vector<string> > &inlines,
        bool debug_info, bool auto_par, bool auto_vec) {

    // CPU schedule generation
    // Create a tiled traversal for the output of the group
    Function &g_out = env[g_name];
    assert(inlines.find(g_out.name()) == inlines.end());

    // std::cerr << "Start scheduling "  <<  g_out.name() << std::endl;
    // The dimension names that will be tiled
    vector<string> pure_vars;
    vector<Dim> &dims = g_out.schedule().dims();
    Partitioner::GroupSched sched = part.group_sched[g_name];

    assert(!(sched.locality && sched.fusion));

    // Get estimates of pipeline bounds
    map<string, int> org_out_estimates =
        get_dim_estimates(g_out.name(), pipeline_bounds, env);
    map<string, int> out_estimates = org_out_estimates;

    map<string, Box> group_bounds =
        get_group_member_bounds(part, g_name, sched.tile_sizes);

    map<string, int> tile_sizes_pure;
    if (sched.locality || sched.fusion) {

        for(int i = 0; i < (int)dims.size() - 1; i++) {
            if (sched.tile_sizes[i] != -1) {
                pure_vars.push_back(dims[i].var);
                tile_sizes_pure[dims[i].var] = sched.tile_sizes[i];
            }
        }

        if (debug_info) {
            std::cerr << g_out.name() << " tile sizes and reuse" << std::endl;
            std::cerr << "[";
            for(int i = 0; i < (int)dims.size() - 1; i++) {
                if (debug_info) {
                    std::cerr << "("  << dims[i].var  << "," << sched.tile_sizes[i] << ","
                        << sched.reuse[i] << ")" << ",";
                }
            }
            std::cerr << "]" << std::endl;
        }

        // Reordering by reuse
        /*
        for(int i = (int)dims.size() - 1; i >= 0; i--) {
            // Find the variable with ith most reuse and move to innermost
            for(int j = 0; j < (int)dims.size() - 1; j++) {
                int rank = 0;
                for (int k = 0; k < (int)dims.size() - 1; k++) {
                    // Count up the number of dimensions with reuse
                    // greater than that of j
                    if (k!=j) {
                        if (sched.reuse[k] > sched.reuse[j] ||
                                (sched.reuse[k] == sched.reuse[j] && k < j))
                            rank++;
                    }
                }
                if (rank == i)
                    move_dim_before_innermost(g_out.schedule(), j);
            }
        }
        */
    }

    //for (auto &e: out_estimates)
    //    std::cout << e.first << " " << e.second << std::endl;

    // Realizing the tiling and updating the dimension estimates
    int num_tile_dims = 0;
    for(auto &v: pure_vars) {
        int index = -1;
        for (int i = 0; i < (int)dims.size() - 1; i++) {
            if (dims[i].var == v) {
                index = i;
                break;
            }
        }
        assert(index!=-1);
        if (tile_sizes_pure[v] > 1) {
            split_dim(g_out.schedule(), index, tile_sizes_pure[v],
                      out_estimates, "tile");
            move_dim_to_outermost(g_out.schedule(), index + 1);
        } else if (tile_sizes_pure[v] == 1) {
            move_dim_to_outermost(g_out.schedule(), index);
        }
        num_tile_dims++;
    }

    int num_fused_dims = 0;
    int parallelism = part.arch_params.parallelism;

    //std::cerr << "Start vectorization pure dims "
    //            <<  g_out.name() << std::endl;
    {
        // Vectorize first
        Schedule &s = g_out.schedule();
        if (check_dim_size(s, 0, part.arch_params.vec_len, out_estimates))
            simple_vectorize(g_out, out_estimates, 0, part.arch_params.vec_len);
        else if (check_dim_size(s, 0, 8, out_estimates))
            simple_vectorize(g_out, out_estimates, 0, 8);
        else if (check_dim_size(s, 0, 4, out_estimates))
            simple_vectorize(g_out, out_estimates, 0, 4);

        /*
        if (auto_vec) {
            int vec_dim = 0;
            int can_vec = true;
            while (can_vec && (vec_dim < (int)dims.size() - 1)) {
                can_vec = false;
                if (check_dim_size(s, vec_dim, part.arch_params.vec_len, out_estimates)){
                    simple_vectorize(g_out, out_estimates, vec_dim, part.arch_params.vec_len);
                    break;
                }
                else {
                    int dim_size = get_dim_size(s, vec_dim, out_estimates);
                    if(dim_size < part.arch_params.vec_len) {
                        // unroll
                        unroll_dim(s, vec_dim);
                        can_vec = true;
                        vec_dim++;
                    }
                }

            }
        }
        */

        int outer_dim = -1;
        bool can_par = pick_dim_to_parallelize(g_out, out_estimates,
                                               parallelism, num_tile_dims,
                                               outer_dim, num_fused_dims);

        if (auto_par && outer_dim !=-1 && can_par)
            parallelize_dim(g_out.schedule(), outer_dim);
    }

    //std::cerr << "Finished pure dims "  <<  g_out.name() << std::endl;
    if (!g_out.is_pure()) {

        int num_updates = g_out.updates().size();
        for (int i = 0; i < num_updates; i ++) {
            // Start with fresh bounds estimates for each update
            map<string, int> out_up_estimates = org_out_estimates;

            Schedule &s = g_out.update_schedule(i);
            vector<Dim> &dims = s.dims();

            const UpdateDefinition &u = g_out.updates()[i];

            // Tiling includes reduction dimensions
            map<string, int> tile_sizes_update;
            vector<string> update_vars;

            unsigned int num_pure_dims = g_out.args().size();
            unsigned int num_red_dims = (dims.size() - 1) - num_pure_dims;

            map<string, float> var_reuse;
            if (sched.locality) {
                assert(part.analy.reductions.find(g_out.name()) !=
                        part.analy.reductions.end());
                assert(sched.tile_sizes.size() ==
                        num_pure_dims + num_red_dims);
                // The tile sizes for the reduction dimensions are at the end
                for(unsigned int i = 0; i < num_red_dims; i++) {
                    var_reuse[dims[i].var] = sched.reuse[num_pure_dims + i];
                    if (sched.tile_sizes[num_pure_dims + i] != -1) {
                        update_vars.push_back(dims[i].var);
                        tile_sizes_update[dims[i].var] =
                            sched.tile_sizes[num_pure_dims + i];
                        //std::cerr << dims[i].var << " "
                        //          << sched.tile_sizes[num_pure_dims + i]
                        //          << std::endl;
                    }
                }
            }

            if (sched.fusion || sched.locality) {
                if (sched.fusion)
                    assert(sched.tile_sizes.size() == num_pure_dims);
                for(unsigned int i = 0; i < num_pure_dims; i++) {
                    var_reuse[dims[num_red_dims + i].var] =
                        sched.reuse[i];
                    if (sched.tile_sizes[i] != -1) {
                        update_vars.push_back(dims[num_red_dims + i].var);
                        tile_sizes_update[dims[num_red_dims + i].var]
                            = sched.tile_sizes[i];
                        //std::cerr << dims[num_red_dims + i].var << " "
                        //          << sched.tile_sizes[i] << std::endl;
                    }
                }
            }

            // Determine which dimension if any can be moved inner most
            // while not disrupting spatial locality both on inputs and
            // outputs
            string inner_dim =
                get_spatially_coherent_innermost_dim(g_out, u);

            if (sched.locality) {
                reorder_by_reuse(s, var_reuse, inner_dim,
                                 out_up_estimates, 16);
                update_vars.clear();
                for (int v = 0; v < (int)dims.size() - 1; v++) {
                    if (tile_sizes_update.find(dims[v].var) != tile_sizes_update.end())
                        update_vars.push_back(dims[v].var);
                }
            }

            set<string> par_vars;
            for(auto &v: update_vars) {
                int index = -1;
                for (int i = 0; i < (int)dims.size() - 1; i++) {
                    if (dims[i].var == v) {
                        index = i;
                        break;
                    }
                }
                assert(index!=-1);
                if (tile_sizes_update[v] > 1) {
                    split_dim(s, index, tile_sizes_update[v],
                              out_up_estimates, "tile");
                    move_dim_to_outermost(s, index + 1);
                    if (can_parallelize_rvar(v, g_out.name(), u)) {
                        int o_dim = s.dims().size() - 2;
                        par_vars.insert(s.dims()[o_dim].var);
                        par_vars.insert(s.dims()[index].var);
                    }
                } else if (tile_sizes_update[v] == 1) {
                    move_dim_to_outermost(s, index);
                }
            }

            // Vectorization of update definitions
            if(auto_vec) {
                vectorize_update(g_out, i, out_up_estimates, part.arch_params.vec_len,
                                 par_vars);
            }

            if(auto_par) {
                int curr_par = 1;
                // Exploiting nested parallelism
                for (int i = (int)dims.size() - 2; i > 0 ; i--) {
                    bool dim_par = can_parallelize_rvar(dims[i].var,
                            g_out.name(), u);
                    dim_par = dim_par ||
                        (par_vars.find(dims[i].var) != par_vars.end());
                    if (dim_par) {
                        curr_par = curr_par * out_up_estimates[dims[i].var];
                        parallelize_dim(s, i);
                        move_dim_to_outermost(s, i);
                        int outer_dim = dims.size() - 2;
                        parallelize_dim(s, outer_dim);
                        if (curr_par > parallelism)
                            break;
                    }
                }
            }
        }
    }

    // std::cerr << "Finished updates "  <<  g_out.name() << std::endl;
    for (auto &m: part.groups[g_name]) {
        int outer_dim = dims.size() - 2;
        map<string, int> org_mem_estimates =
            get_dim_estimates(m.name(), group_bounds, env);
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
            if (auto_vec) {
                if (check_dim_size(m.schedule(), 0, part.arch_params.vec_len, mem_estimates))
                    simple_vectorize(m, mem_estimates, 0, part.arch_params.vec_len);
                else if (check_dim_size(m.schedule(), 0, 8, mem_estimates))
                    simple_vectorize(m, mem_estimates, 0, 8);
                else if (check_dim_size(m.schedule(), 0, 4, mem_estimates))
                    simple_vectorize(m, mem_estimates, 0, 4);
            }
            if (!m.is_pure()) {
                int num_updates = m.updates().size();
                for (int i = 0; i < num_updates; i ++) {
                    // Start with fresh bounds estimates for each update
                    map<string, int> mem_up_estimates = org_mem_estimates;
                    set<string> par_vars;
                    vectorize_update(m, i, mem_up_estimates, part.arch_params.vec_len,
                                     par_vars);
                }
            }
        }
    }
    // std::cerr << "Finished group members "  <<  g_out.name() << std::endl;
}

void mark_block_dims(vector<Dim> &dims, map<string, int> &tile_sizes,
                     vector<string> &block_dims, map<string, int> &estimates,
                     int parallelism, int vec_len, int target_threads_per_block,
                     int max_threads_per_block, set<string> &par_vars) {

    if (tile_sizes.size() == 0) {
        // Handle groups with no tiling
        int curr_par = 1;
        int num_mapped_to_block = 0;
        int inner_tiled_dim = 0;
        int num_threads = 1;

        for(int i = 0; i < (int)dims.size() - 1; i++) {
            if (par_vars.find(dims[i].var) == par_vars.end())
                continue;
            if (num_threads < target_threads_per_block && num_mapped_to_block < 3) {
                int tile_size = 0;
                int rem_threads = std::ceil(target_threads_per_block/num_threads);
                if (estimates[dims[i].var] >= std::min(vec_len, rem_threads))
                    tile_size = std::min(vec_len, rem_threads);
                else
                    tile_size = estimates[dims[i].var];

                if (num_threads * tile_size > max_threads_per_block)
                    break;

                tile_sizes[dims[i].var] = tile_size;
                curr_par *= std::ceil((float)estimates[dims[i].var]/tile_sizes[dims[i].var]);
                num_threads *= tile_sizes[dims[i].var];
                num_mapped_to_block++;
                block_dims.push_back(dims[i].var);
                inner_tiled_dim = i;
            }
        }

        for(int i = (int)dims.size() - 2; i > inner_tiled_dim; i--) {
            if (par_vars.find(dims[i].var) == par_vars.end())
                continue;
            if (num_mapped_to_block < 3 &&
                    ((curr_par < parallelism))) {
                tile_sizes[dims[i].var] = 1;
                curr_par *= std::ceil((float)estimates[dims[i].var]/tile_sizes[dims[i].var]);
                num_threads *= tile_sizes[dims[i].var];
                block_dims.push_back(dims[i].var);
                num_mapped_to_block++;
            }
        }

    } else {

        int curr_par = 1;
        int num_mapped_to_block = 0;
        int inner_tiled_dim = 0;
        int num_threads = 1;

        for(int i = 0; i < (int)dims.size() - 1; i++) {
            if (par_vars.find(dims[i].var) == par_vars.end())
                continue;
            if (num_threads < target_threads_per_block && num_mapped_to_block < 3) {
                int tile_size = 0;
                int rem_threads = std::ceil(target_threads_per_block/num_threads);
                if (tile_sizes.find(dims[i].var) == tile_sizes.end()) {
                    if (estimates[dims[i].var] >= std::min(vec_len, rem_threads))
                        tile_size = std::min(vec_len, rem_threads);
                    else
                        tile_size = estimates[dims[i].var];
                } else {
                    tile_size = tile_sizes[dims[i].var];
                }

                if (num_threads * tile_size > max_threads_per_block)
                    break;

                tile_sizes[dims[i].var] = tile_size;
                curr_par *= std::ceil((float)estimates[dims[i].var]/tile_sizes[dims[i].var]);
                num_threads *= tile_sizes[dims[i].var];
                num_mapped_to_block++;
                block_dims.push_back(dims[i].var);
                inner_tiled_dim = i;
            }
        }

        for(int i = (int)dims.size() - 2; i > inner_tiled_dim; i--) {
            if (par_vars.find(dims[i].var) == par_vars.end())
                continue;
            if (tile_sizes.find(dims[i].var) != tile_sizes.end()) {
                if (num_mapped_to_block < 3 &&
                        (curr_par < parallelism) &&
                        (num_threads * tile_sizes[dims[i].var] < max_threads_per_block)) {
                    curr_par *= std::ceil((float)estimates[dims[i].var]/ tile_sizes[dims[i].var]);
                    num_threads *= tile_sizes[dims[i].var];
                    block_dims.push_back(dims[i].var);
                    num_mapped_to_block++;
                }
            }
        }
    }
}

void realize_tiling_gpu(vector<Dim> &dims, Schedule &s,
                        vector<string> &tile_vars, map<string, int> &tile_sizes,
                        vector<string> &block_dims, map<string, int> &estimates,
                        int &num_tile_dims, vector<int> &dim_threads) {

    // Mapping tiling to GPU block and thread mapping
    std::vector<string> block_names = {"__block_id_x",
                                       "__block_id_y",
                                       "__block_id_z"};

    std::vector<string> thread_names = {"__thread_id_x",
                                        "__thread_id_y",
                                        "__thread_id_z"};

    int num_block_dim = 0;
    for(auto &v: tile_vars) {
        int index = -1;
        for (int i = 0; i < (int)dims.size() - 1; i++) {
            if (dims[i].var == v) {
                index = i;
                break;
            }
        }
        assert(index!=-1);
        if (tile_sizes[v] >= 1) {
            if (std::find(block_dims.begin(), block_dims.end(), v) == block_dims.end()) {
                if (tile_sizes[v] > 1) {
                    split_dim(s, index, tile_sizes[v],
                              estimates, "tile");
                    move_dim_to_outermost(s, index + 1);
                } else if (tile_sizes[v] == 1) {
                    move_dim_to_outermost(s, index);
                }
                num_tile_dims++;
            }
        }
    }

    for(auto &v: tile_vars) {
        if (std::find(block_dims.begin(), block_dims.end(), v) == block_dims.end())
            continue;
        int index = -1;
        for (int i = 0; i < (int)dims.size() - 1; i++) {
            if (dims[i].var == v) {
                index = i;
                break;
            }
        }
        assert(index!=-1);
        if (tile_sizes[v] >= 1) {
            string block_name = block_names[num_block_dim];
            string thread_name = thread_names[num_block_dim];
            dim_threads[num_block_dim] = std::max(dim_threads[num_block_dim], tile_sizes[v]);

            split_dim_gpu(s, index, tile_sizes[v], estimates, block_name, thread_name);
            move_dim_to_outermost(s, index + 1);
            num_block_dim++;
            num_tile_dims++;
        }
    }

    // Unrolling small inner reduction dims
    int iterations_unrolled = 1;
    int unroll_limit = 16;
    for (int i = 0; i < (int)dims.size() - 1; i++) {
        if (!dims[i].pure && (iterations_unrolled * estimates[dims[i].var]) < unroll_limit) {
            iterations_unrolled *= estimates[dims[i].var];
            unroll_dim(s, i);
        }
        else
            break;
    }
}

void synthesize_gpu_schedule(string g_name, Partitioner &part,
                             map<string, Function> &env,
                             map<string, Box> &pipeline_bounds,
                             map<string, vector<string> > &inlines,
                             bool debug_info) {

    Function &g_out = env[g_name];

    // The dimension names that will be tiled
    vector<string> pure_tile_vars;
    vector<Dim> &dims = g_out.schedule().dims();

    Partitioner::GroupSched sched = part.group_sched[g_name];

    assert(!(sched.locality && sched.fusion));

    // Get estimates of pipeline bounds
    map<string, int> org_out_estimates =
        get_dim_estimates(g_out.name(), pipeline_bounds, env);
    map<string, int> out_estimates = org_out_estimates;

    map<string, Box> group_bounds =
        get_group_member_bounds(part, g_name, sched.tile_sizes);

    map<string, int> tile_sizes_pure;
    if (sched.locality || sched.fusion) {

        for(int i = 0; i < (int)dims.size() - 1; i++) {
            if (sched.tile_sizes[i] != -1) {
                tile_sizes_pure[dims[i].var] = sched.tile_sizes[i];
            }
        }

        if (debug_info) {
            std::cerr << g_out.name() << " tile sizes and reuse" << std::endl;
            std::cerr << "[";
            for(int i = 0; i < (int)dims.size() - 1; i++) {
                std::cerr << "("  << dims[i].var  << "," << sched.tile_sizes[i] << ","
                    << sched.reuse[i] << ")" << ",";
            }
            std::cerr << "]" << std::endl;
        }
    }

    //for (auto &e: out_estimates)
    //    std::cout << e.first << " " << e.second << std::endl;

    vector<string> block_dims;
    vector<int> dim_threads = {0, 0, 0};
    set<string> pure_par_vars;

    for(int i = 0; i < (int)dims.size() - 1; i++) {
        pure_par_vars.insert(dims[i].var);
    }

    mark_block_dims(dims, tile_sizes_pure, block_dims, out_estimates,
                    part.arch_params.parallelism,
                    part.arch_params.vec_len,
                    part.arch_params.target_threads_per_block,
                    part.arch_params.max_threads_per_block,
                    pure_par_vars);

    // Populate pure_tile_vars in order
    for(int i = 0; i < (int)dims.size() - 1; i++) {
        if (tile_sizes_pure.find(dims[i].var) != tile_sizes_pure.end())
            pure_tile_vars.push_back(dims[i].var);
    }

    int num_tile_dims = 0;

    //std::cerr << g_out.name() <<std::endl;
    realize_tiling_gpu(dims, g_out.schedule(), pure_tile_vars,
                       tile_sizes_pure, block_dims, out_estimates,
                       num_tile_dims, dim_threads);

    if (!g_out.is_pure()) {

        int num_updates = g_out.updates().size();
        for (int i = 0; i < num_updates; i++) {
            // Start with fresh bounds estimates for each update
            map<string, int> out_up_estimates = org_out_estimates;

            Schedule &s = g_out.update_schedule(i);
            vector<Dim> &u_dims = s.dims();

            const UpdateDefinition &u = g_out.updates()[i];

            // Tiling includes reduction dimensions
            map<string, int> tile_sizes_update;
            vector<string> update_vars;

            unsigned int num_pure_dims = g_out.args().size();
            unsigned int num_red_dims = (u_dims.size() - 1) - num_pure_dims;

            map<string, float> var_reuse;
            if (sched.locality) {
                assert(part.analy.reductions.find(g_out.name()) !=
                        part.analy.reductions.end());
                assert(sched.tile_sizes.size() ==
                        num_pure_dims + num_red_dims);
                // The tile sizes for the reduction dimensions are at the end
                for(unsigned int d = 0; d < num_red_dims; d++) {
                    var_reuse[u_dims[d].var] = sched.reuse[num_pure_dims + d];
                    if (sched.tile_sizes[num_pure_dims + d] != -1) {
                        tile_sizes_update[u_dims[d].var] =
                            sched.tile_sizes[num_pure_dims + d];
                        //std::cerr << u_dims[d].var << " "
                        //          << sched.tile_sizes[num_pure_dims + d]
                        //          << std::endl;
                    }
                }
            }

            if (sched.fusion || sched.locality) {
                if (sched.fusion)
                    assert(sched.tile_sizes.size() == num_pure_dims);
                for(unsigned int d = 0; d < num_pure_dims; d++) {
                    var_reuse[u_dims[num_red_dims + d].var] = sched.reuse[d];
                    if (sched.tile_sizes[d] != -1) {
                        tile_sizes_update[u_dims[num_red_dims + d].var]
                                        = sched.tile_sizes[d];
                        //std::cerr << u_dims[num_red_dims + d].var << " "
                        //          << sched.tile_sizes[d] << std::endl;
                    }
                }
            }

            // Determine which dimension if any can be moved inner most
            // while not disrupting spatial locality both on inputs and
            // outputs
            string inner_dim =
                get_spatially_coherent_innermost_dim(g_out, u);

            if (sched.locality) {
                reorder_by_reuse(s, var_reuse, inner_dim,
                                 out_up_estimates, 16);
            }

            set<string> update_par_vars;
            for (int i = 0; i < (int)u_dims.size() - 1; i++) {
                if (can_parallelize_rvar(u_dims[i].var, g_out.name(), u)) {
                    update_par_vars.insert(u_dims[i].var);
                }
            }

            vector<string> block_dims_update;

            /*
            for (auto &p: update_par_vars)
                std::cerr << p << std::endl;
            */

            mark_block_dims(u_dims, tile_sizes_update, block_dims_update,
                            out_up_estimates, part.arch_params.parallelism,
                            part.arch_params.vec_len,
                            part.arch_params.target_threads_per_block,
                            part.arch_params.max_threads_per_block,
                            update_par_vars);

            for (int v = 0; v < (int)u_dims.size() - 1; v++) {
                if (tile_sizes_update.find(u_dims[v].var) != tile_sizes_update.end())
                    update_vars.push_back(u_dims[v].var);
            }

            int num_tile_dims_update = 0;

            /*
            for (auto &est: out_up_estimates)
                std::cerr << "Estimate:" << est.first << "," << est.second << std::endl;
            for (auto &t: tile_sizes_update)
                std::cerr << "Tile size:" << t.first << "," << t.second << std::endl;
            std::cerr << std::endl;
            */

            realize_tiling_gpu(u_dims, s, update_vars, tile_sizes_update,
                               block_dims_update, out_up_estimates,
                               num_tile_dims_update, dim_threads);

        }
    }

    int num_fused_dims = 0;

    for (auto &m: part.groups[g_name]) {
        int outer_dim = dims.size() - 2;
        //map<string, int> org_mem_estimates =
        //    get_dim_estimates(m.name(), pipeline_bounds, env);
        map<string, int> org_mem_estimates =
            get_dim_estimates(m.name(), group_bounds, env);
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

            // Parallelize within a tile
            //for (auto &est: mem_estimates)
            //    std::cerr << est.first << "," << est.second << std::endl;

            pick_gpu_thread_dims(m.schedule(), mem_estimates,
                                 part.arch_params.max_threads_per_block,
                                 dim_threads);
            if (!m.is_pure()) {
                int num_updates = m.updates().size();
                for (int u = 0; u < num_updates; u++) {
                    Schedule &u_s = m.update_schedule(u);
                    pick_gpu_thread_dims(u_s, mem_estimates,
                                         part.arch_params.max_threads_per_block,
                                         dim_threads);
                }
            }
        }
    }

    //for (auto &t: dim_threads)
    //    std::cerr << t << std::endl;
    // std::cerr << "Finished group members "  <<  g_out.name() << std::endl;
}

void schedule_advisor(const vector<Function> &outputs,
                      const vector<string> &order,
                      map<string, Function> &env,
                      const FuncValueBounds &func_val_bounds,
                      const Target &target,
                      bool root_default, bool auto_inline,
                      bool auto_par, bool auto_vec) {

    const char *random_seed_var = getenv("HL_AUTO_RANDOM_SEED");
    int random_seed = 0;
    if (random_seed_var) {
        random_seed = atoi(random_seed_var);
        srand(random_seed);
    }
    fprintf(stdout, "HL_AUTO_RANDOM_SEED: %d\n", random_seed);

    const char *debug_var = getenv("HL_AUTO_DEBUG");
    bool debug_info = false;
    if (debug_var)
        debug_info = true;

    const char *auto_naive_var = getenv("HL_AUTO_NAIVE");
    bool auto_naive = false;
    if (auto_naive_var)
        auto_naive = true;
    fprintf(stdout, "HL_AUTO_NAIVE: %d\n", auto_naive);

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

    // TODO explain strcuture
    map<string, Box> pipeline_bounds;
    map<string, vector<string> > update_args;
    set<string> reductions;

    // TODO explain structure
    std::map<string, pair<long long, long long> > func_cost;
    for (auto& kv : env) {
        //std::cout << kv.first << ":" << std::endl;
        assert(func_cost.find(kv.first) == func_cost.end());

        func_cost[kv.first].first = 1;
        func_cost[kv.first].second = 0;

        if (!kv.second.is_boundary()) {
            for (auto &e: kv.second.values()) {
                ExprCostEarly cost_visitor;
                e.accept(&cost_visitor);
                func_cost[kv.first].first += cost_visitor.ops;
                func_cost[kv.first].second += cost_visitor.loads;
            }
        }

        // Estimating cost when reductions are involved
        // Only considering functions with a single update covers most of the
        // cases we want to tackle
        assert(kv.second.updates().size() <= 1);
        if (!kv.second.is_pure()) {
            const UpdateDefinition &u = kv.second.updates()[0];
            bool reduction = true;

            long long ops = 1;
            long long loads = 0;
            for (auto &e: u.values) {
                ExprCostEarly cost_visitor;
                e.accept(&cost_visitor);
                ops += cost_visitor.ops;
                loads += cost_visitor.loads;
            }

            vector<string> red_args;
            int arg_pos = 0;

            for (auto &arg: u.args) {

                ExprCostEarly cost_visitor;
                arg.accept(&cost_visitor);
                ops += cost_visitor.ops;
                loads += cost_visitor.loads;

                // Check for a pure variable
                const Variable *v = arg.as<Variable>();
                if (!v || v->name != kv.second.args()[arg_pos]) {
                    reduction = false;
                }
                arg_pos++;
            }

            if (reduction) {
                for (auto &rvar: u.domain.domain()) {
                    red_args.push_back(rvar.var);
                }
                reductions.insert(kv.first);
                update_args[kv.first] = red_args;
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

    if (debug_info) {
        std::cerr << "Reductions in the pipeline:" << std::endl;
        for (auto &r: reductions) {
            std::cerr << r << std::endl;
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

    if (auto_naive) {
        inlines = simple_inline(all_calls, consumers, env, outputs);
    } else {
        for (auto &f: env) {
            if (env[f.first].is_lambda()) {
                //assert(consumers[f.first].size() == 1);
                inlines[f.first].push_back(consumers[f.first][0]);
                env[f.first].schedule().store_level().var = "";
                env[f.first].schedule().compute_level().var = "";
            }
        }
    }

    if (debug_info) {
        std::cerr << "Inlining lambda functions:" << std::endl;
        disp_inlines(inlines);
        std::cerr << std::endl;
    }

    auto_vec = true;
    auto_par = true;

    // Dependence analysis

    // For each function compute all the regions of upstream functions
    // required to compute a region of the function

    DependenceAnalysis analy(env, func_val_bounds, reductions, update_args);

    /*
    for (auto &reg: analy.func_dep_regions) {
        disp_regions(reg.second);
        std::cout << std::endl;
    }
    */

    bool estimates_avail = check_estimates_on_outputs(outputs);

    //if (debug_info) {
        std::cerr << "Estimates of pipeline output sizes:" << estimates_avail << std::endl;
    //}

    if (estimates_avail) {
        for (auto &out: outputs) {
            vector<pair<int, int> > bounds;
            vector<bool> eval;
            vector<string> vars = out.args();
            for (unsigned int i = 0; i < vars.size(); i++) {
                bool found = false;
                for (auto &b: out.schedule().estimates())
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
                    analy.concrete_dep_regions(out.name(), eval,
                                               analy.func_dep_regions, bounds);

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

    if (debug_info) {
       std::cerr << "Pipeline size estimates inferred from output estimates:" << std::endl;
       disp_regions(pipeline_bounds);
    }

    // Initialize the partitioner
    bool gpu_schedule = false;
    if ((target.has_feature(Target::CUDA) ||
                target.has_feature(Target::CUDACapability30)||
                target.has_feature(Target::CUDACapability32)||
                target.has_feature(Target::CUDACapability35)||
                target.has_feature(Target::CUDACapability50))) {
        gpu_schedule = true;
    }

    Partitioner part(pipeline_bounds, inlines, analy, func_cost, outputs,
                     gpu_schedule, random_seed, debug_info);

    if (debug_info) {
        std::cerr << "Function costs pre-inlining" << std::endl;
        part.disp_costs();
        std::cerr << std::endl;
    }

    if (!auto_naive)
        part.tile_for_input_locality(true);

    if (!auto_naive)
        part.initialize_groups_inline();

    if (debug_info) {
        std::cerr << "Groups pre-inlining" << std::endl;
        part.disp_grouping();
        std::cerr << std::endl;
    }

    if (!auto_naive)
        part.group(Partitioner::INLINE);

    if (debug_info) {
        std::cerr << "Groups post-inlining" << std::endl;
        part.disp_grouping();
        std::cerr << std::endl;

        std::cerr << "Function costs post-inlining" << std::endl;
        part.disp_costs();
        std::cerr << std::endl;
    }

    if (!auto_naive) {
        part.initialize_groups_fast_mem();
        part.group(Partitioner::FAST_MEM);
    }

    if (debug_info) {
        std::cerr << "Groups Fast Mem" << std::endl;
        part.disp_grouping();
        std::cerr << std::endl;
    }

    if (!auto_naive)
        part.tile_for_input_locality();

    // Schedule generation based on grouping
    for (auto& g: part.groups) {

        if (gpu_schedule) {
            synthesize_gpu_schedule(g.first, part, env, pipeline_bounds, inlines, debug_info);
        } else {
            synthesize_cpu_schedule(g.first, part, env, pipeline_bounds,
                                    inlines, debug_info, auto_par, auto_vec);
        }
    }

    //if (root_default || auto_vec || auto_par || auto_inline)
    //    disp_schedule_and_storage_mapping(env);

	return;
}

}
}
