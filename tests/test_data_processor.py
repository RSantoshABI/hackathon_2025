import numpy as np

from optimization.data_processor import (
    adjust_period,
    assign_size_group,
    check_periods_in_valid_range,
    create_bounds,
    extract_pack_type,
    generate_periods,
)


def test_generate_periods():
    periods = generate_periods("2025-01", "2025-03")
    assert periods == ["2025-01", "2025-02", "2025-03"]


def test_adjust_period():
    assert adjust_period("2025-05", 1) == "2024-05"


def test_check_periods_valid_direct():
    periods, mapping = check_periods_in_valid_range(
        ["2025-01", "2025-02"], ["2025-01", "2025-02"]
    )
    assert periods == ["2025-01", "2025-02"]
    assert mapping["2025-01"] == "2025-01"


def test_check_periods_with_shift():
    # only 2024 months exist, 2025 asks should map backwards
    periods, mapping = check_periods_in_valid_range(
        ["2025-01"], ["2024-01"]
    )
    assert periods == ["2024-01"]
    assert mapping["2025-01"] == "2024-01"


def test_extract_pack_type():
    assert extract_pack_type("FOO NRB 330") == "NRB"
    assert extract_pack_type("BAR RB 500") == "RB"
    assert extract_pack_type("BAZ CAN 473") == "CAN"
    assert extract_pack_type("UNKNOWN FORMAT") == "UNKNOWN"


def test_assign_size_group_can():
    row = {"capacity": 330, "pack_type": "CAN"}
    assert assign_size_group(row) == "Regular"


def test_assign_size_group_rb_large():
    row = {"capacity": 700, "pack_type": "RB"}
    assert assign_size_group(row) == "Large"


def test_create_bounds_present_missing():
    import pandas as pd
    df = pd.DataFrame({
        "reference_price": [1.0, 1.2, np.nan],
        "capacity": [330, 330, 330],
        "present": ["present", "present", "missing"],
    })
    bounds = create_bounds(df.copy(), -0.2, 0.2)
    assert len(bounds) == 3
    # first bound should be around reference unit price (scaled) +- 0.2
    ref_unit = 1.0 * 330 / 1000
    assert bounds[0][0] <= ref_unit + 0.2
    # missing keeps same
    assert bounds[2][0] == bounds[2][1]
