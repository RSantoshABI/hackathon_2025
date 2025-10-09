import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path for direct module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from optimization.data_processor import (  # noqa: E402
    create_bounds,
    create_elasticity_matrix,
    create_reference_arrays,
    create_segment_size_labels,
    pre_processor,
)
from optimization.optimization import (  # noqa: E402
    create_evaluation_functions,
)


@pytest.fixture
def synthetic_raw_data():
    """Minimal synthetic dataset: 2 months, 3 own SKUs, 2 competitor SKUs."""
    months = ["2025-07", "2025-08"]

    # Own reference
    ref_rows = []
    for m in months:
        ref_rows.extend([
            {
                "year_month": m, "sku": "A", "reference_volume": 100.0,
                "reference_price": 1.0, "markup": 0.2, "discount": 0.05,
                "excise": 0.01, "vilc": 0.02, "sellout_volume": 90.0,
            },
            {
                "year_month": m, "sku": "B", "reference_volume": 120.0,
                "reference_price": 1.1, "markup": 0.2, "discount": 0.05,
                "excise": 0.01, "vilc": 0.02, "sellout_volume": 110.0,
            },
            {
                "year_month": m, "sku": "C", "reference_volume": 80.0,
                "reference_price": 0.9, "markup": 0.2, "discount": 0.05,
                "excise": 0.01, "vilc": 0.02, "sellout_volume": 70.0,
            },
        ])
    reference_df = pd.DataFrame(ref_rows)

    # Competitor reference
    comp_rows = []
    for m in months:
        comp_rows.extend([
            {
                "year_month": m, "sku": "X", "reference_volume": 200.0,
                "reference_price": 1.2,
            },
            {
                "year_month": m, "sku": "Y", "reference_volume": 150.0,
                "reference_price": 1.15,
            },
        ])
    competitor_reference_df = pd.DataFrame(comp_rows)

    # Elasticities (own to own) identity negative diag, small cross effect
    own_to_own_elasticity_df = pd.DataFrame(
        [
            {
                "target_sku": t,
                "other_sku": o,
                "elasticity": (-0.8 if t == o else 0.05),
            }
            for t in ["A", "B", "C"]
            for o in ["A", "B", "C"]
        ]
    )
    # Own to competitor
    own_to_comp_elasticity = pd.DataFrame([
        {"target_sku": t, "other_sku": o, "elasticity": 0.02}
        for t in ["A", "B", "C"] for o in ["X", "Y"]
    ])

    # Segment mapping (simple)
    seg_mapping = pd.DataFrame({
        "sku": ["A", "B", "C"],
        "segment": ["Core", "Core+", "Premium"],
        "sellin_sku": ["A", "B", "C"],
        "capacity": [330.0, 330.0, 500.0],
    })

    # SKU detail mapping (capacity etc.) mimic seg_mapping shape
    sku_detail_mapping = seg_mapping.copy()

    sku_scope_df = pd.DataFrame({"sku": ["A", "B", "C"]})

    return dict(
        reference_df=reference_df,
        competitor_reference_df=competitor_reference_df,
        own_to_own_elasticity_df=own_to_own_elasticity_df,
        own_to_comp_elasticity=own_to_comp_elasticity,
        seg_mapping=seg_mapping,
        sku_detail_mapping=sku_detail_mapping,
        sku_scope_df=sku_scope_df,
        months=months,
    )


@pytest.fixture
def processed_data(synthetic_raw_data):
    data = synthetic_raw_data
    valid_periods = data["months"]  # all valid
    processed = pre_processor(
        data["own_to_own_elasticity_df"],
        data["reference_df"],
        data["own_to_comp_elasticity"],
        data["competitor_reference_df"],
        data["seg_mapping"],
        start_period=data["months"][0],
        end_period=data["months"][-1],
        valid_periods=valid_periods,
        VILC_GR=0.0,
    )
    (
        reference_df_proc,
        competitor_reference_df_proc,
        own_to_own_elasticity_df_proc,
        own_to_comp_elasticity_df_proc,
        reference_df_padded_bound,
    ) = processed
    (
        own_products,
        competitor_products,
        months,
        M,
        N,
        K,
        E_own,
        E_comp,
    ) = create_elasticity_matrix(
        own_to_own_elasticity_df_proc,
        reference_df_proc,
        own_to_comp_elasticity_df_proc,
        competitor_reference_df_proc,
    )

    arrays = create_reference_arrays(
        reference_df_proc,
        competitor_reference_df_proc,
        months,
        own_products,
        competitor_products,
    )

    bounds = create_bounds(reference_df_padded_bound, -0.2, 0.2)
    segment_labels, size_labels = create_segment_size_labels(
        reference_df_proc, arrays["M"], arrays["N"]
    )

    return dict(
        reference_df=reference_df_proc,
        competitor_reference_df=competitor_reference_df_proc,
        own_products=own_products,
        competitor_products=competitor_products,
        months=months,
        M=M,
        N=N,
        K=K,
        E_own=E_own,
        E_comp=E_comp,
        arrays=arrays,
        bounds=bounds,
        segment_labels=segment_labels,
        size_labels=size_labels,
        raw=data,
    )


@pytest.fixture
def evaluation_functions(processed_data):
    a = processed_data["arrays"]
    (
        eval_uncached,
        eval_cached,
        clear_cache,
        base_total_MACO,
        NR_ref,
        MACO_ref,
    ) = create_evaluation_functions(
        processed_data["M"],
        processed_data["N"],
        processed_data["K"],
        processed_data["E_own"],
        processed_data["E_comp"],
        a["ref_price_liter"],
        a["ref_price_unit"],
        a["ref_vol_own_sellin"],
        a["ref_vol_own_sellout"],
        a["ref_vol_comp"],
        a["capacity"],
        a["markup"],
        a["discount"],
        a["excise"],
        a["vilc"],
        VAT=0.19,
        remaining_abi_industry_volume=50.0,
    )
    return dict(
        eval_uncached=eval_uncached,
        eval_cached=eval_cached,
        clear_cache=clear_cache,
        base_total_MACO=base_total_MACO,
        NR_ref=NR_ref,
        MACO_ref=MACO_ref,
    )
