import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import Direction, NoOrder, SizeType

from core.utils_nb import clip_nb

logger = logging.getLogger(__name__)


@njit
def pre_group_func_nb(c):
    order_value_out = np.empty(c.group_len, dtype=np.float_)
    return (order_value_out,)


@njit
def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
    """Update last_val_price with today's price for use in `order_func_nb`

    Args:
        c (vbt.portfolio.enums.OrderContext): OrderContext

    Returns:
        tuple: Empty tuple
    """
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = nb.get_col_elem_nb(c, col, c.close)
    nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
    return ()


@njit
def order_func_nb(
    c,
    size,
    volume,
    size_type,
    direction,
    fees,
    fixed_fees,
    slippage,
    volume_max_size: float,
    rebalancing_buffer: float,
):
    """
    Execute long-only orders while respecting sizing constraints (% of available
    volume) and rebalancing buffers.

    Args:
        c (vbt.portfolio.enums.OrderContext): OrderContext
        size (_type_): _description_
        volume (_type_): _description_
        volume_max_size (float, optional): _description_. Defaults to 0.01.
        rebalancing_buffer (float, optional): _description_. Defaults to 0.01.

    Returns:
        vbt.portfolio.enums.Order: Order
    """
    # Select info related to this order
    # flex_select_auto_nb allows us to pass size as single number, 1-dim or 2-dim array
    # If flex_2d is True, 1-dim array will be per column, otherwise per row
    target_now = flex_select_auto_nb(
        np.asarray(size), c.i, c.col, flex_2d=True
    )  # [% of Account Size]
    account_size = c.value_now  # [$]
    # close is always 2-dim array
    price_now = c.close[c.i, c.col]  # [$ / Unit]
    last_position = nb.get_elem_nb(c, c.last_position)  # [Units of Asset]
    size_now = price_now * last_position  # [$]
    pct_now = size_now / account_size  # [% of Account Size]

    # Check if position deviation exceeds buffer percentage.
    position_deviation = np.abs(target_now - pct_now)  # [Percentage of Account Size]
    rebalance = position_deviation > rebalancing_buffer
    if not rebalance:
        return NoOrder
    # Rebalance to the edge of the buffer (not to the target), minimize trading.
    if target_now > pct_now:
        target_now -= rebalancing_buffer
    else:
        target_now += rebalancing_buffer

    # Translate target position into trade amount
    target_size = target_now * account_size  # [$]
    trade_size = target_size - size_now  # [$]
    trade_amnt = trade_size / price_now  # [Units of Asset]

    # Calculate position size based on % of available volume
    # TODO(@eugene.lo): Distinguish between short and long sizing...
    volume_now = flex_select_auto_nb(
        np.asarray(volume), c.i, c.col, flex_2d=True
    )  # [Units of Asset]
    if np.isnan(volume_now):
        volume_now = 0
    max_position_amnt = volume_max_size * volume_now  # [Units of Asset]
    trade_amnt = clip_nb(trade_amnt, -max_position_amnt, max_position_amnt)
    trade_size = trade_amnt * price_now  # [$]
    trade_pct = trade_size / account_size  # [%]

    # Get order size based on size type
    order_size_type = nb.get_elem_nb(c, size_type)
    if order_size_type == SizeType.Amount:
        order_size = trade_amnt
    elif order_size_type == SizeType.Value:
        order_size = trade_size
    elif order_size_type == SizeType.Percent:
        order_size = trade_pct
    else:
        raise ValueError("Unsupported size type!")

    # Submit order
    if order_size == 0:
        return NoOrder
    return nb.order_nb(
        size=order_size,
        price=price_now,
        size_type=order_size_type,
        direction=nb.get_elem_nb(c, direction),
        fees=nb.get_elem_nb(c, fees),
        fixed_fees=nb.get_elem_nb(c, fixed_fees),
        slippage=nb.get_elem_nb(c, slippage),
    )


def simulate(
    price: pd.DataFrame,
    positions: pd.DataFrame,
    volume: pd.DataFrame,
    volume_max_size: float,
    rebalancing_buffer: float,
    initial_capital: float,
    segment_mask: Optional[int] = None,
    direction: Direction = Direction.LongOnly,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    leverage: float = 1.0,
    leverage_mode: Optional[Any] = None,
) -> vbt.Portfolio:
    if leverage is not None:
        logger.warning("Leverage is not supported by vectorbt, ignoring!")

    size_np = positions.to_numpy()
    volume_np = volume.to_numpy()
    # fmt: off
    pf = vbt.Portfolio.from_order_func(
        price,
        order_func_nb,
        # Args for order_func_nb
        size_np, volume_np, vbt.Rep('size_type'), vbt.Rep('direction'),
        vbt.Rep('fees'), vbt.Rep('fixed_fees'), vbt.Rep('slippage'),
        volume_max_size, rebalancing_buffer,
        # Kwargs
        pre_group_func_nb=pre_group_func_nb,
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(
            size_np,
            vbt.Rep('price'),
            vbt.Rep('size_type'),
            vbt.Rep('direction')
        ),
        broadcast_named_args=dict(  # broadcast against each other
            price=price,
            size_type=SizeType.Amount,
            direction=direction,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
        ),
        init_cash=initial_capital,
        cash_sharing=True,
        group_by=True,
        segment_mask=segment_mask,
    )
    # fmt: on
    return pf
