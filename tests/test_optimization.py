
from optimization.constraints import (
    create_constraint_functions,
    round_to_nearest_50,
)
from optimization.optimization import (
    create_objective_function,
    create_results_dataframe,
    run_optimization,
)


def test_evaluation_and_objective(processed_data, evaluation_functions):
    a = processed_data["arrays"]
    eval_uncached = evaluation_functions["eval_uncached"]

    # Use reference unit prices as initial P0
    ref_P = a["ref_price_unit"].copy()
    P0 = ref_P.reshape(-1)

    res_eval = eval_uncached(P0)
    assert "total_MACO" in res_eval
    assert res_eval["total_MACO"] > 0
    assert res_eval["Q_own_opt"].shape == (a["M"], a["N"])


